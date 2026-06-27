import os
# os.chdir("..")
from copy import deepcopy
import csv
import time

import torch
import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from src.utils.plotting import make_matching_figure

from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter

# You can choose model type in ['full', 'opt']
model_type = 'opt' # 'full' for best quality, 'opt' for best efficiency

# You can choose numerical precision in ['fp32', 'mp', 'fp16']. 'fp16' for best efficiency
precision = 'fp32' # Enjoy near-lossless precision with Mixed Precision (MP) / FP16 computation if you have a modern GPU (recommended NVIDIA architecture >= SM_70).

# You can also change the default values like thr. and npe (based on input image size)

if model_type == 'full':
    _default_cfg = deepcopy(full_default_cfg)
elif model_type == 'opt':
    _default_cfg = deepcopy(opt_default_cfg)
    
if precision == 'mp':
    _default_cfg['mp'] = True
elif precision == 'fp16':
    _default_cfg['half'] = True
    
print(_default_cfg)
matcher = LoFTR(config=_default_cfg)

checkpoint = torch.load("./checkpoint/eloftr_outdoor.ckpt", map_location='cuda', weights_only=False)

if 'state_dict' in checkpoint:
    matcher.load_state_dict(checkpoint['state_dict'])
else:
    matcher.load_state_dict(checkpoint)

matcher = reparameter(matcher) # no reparameterization will lead to low performance

if precision == 'fp16':
    matcher = matcher.half()

matcher = matcher.eval().cuda()

# 1. Config
online_img_pth = "Online_Keyframe/R1257.png"
offline_folder = "Offline_Keyframes_Turn2-3/"
offline_imgs = [f for f in os.listdir(offline_folder) if f.endswith('.png')]

output_dir = "output_matches_fp32"
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "ELoFTR_fp32_stats.csv")

# 2. Online frame load
img0_raw = cv2.imread(online_img_pth, cv2.IMREAD_GRAYSCALE)
target_w, target_h = 960, 256 #almost half the original size (1920x500)
img0_raw = cv2.resize(img0_raw, (target_w, target_h))
img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//32*32, img0_raw.shape[0]//32*32))

if precision == 'fp16':
    img0 = torch.from_numpy(img0_raw)[None][None].half().cuda() / 255.
else:
    img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.

inference_times = []
confidences = []
inliers_number = []
csv_rows = []
inliers_geometric_number = []

# 3. Matching pipeline
for img_name in offline_imgs:
    img1_raw = cv2.imread(os.path.join(offline_folder, img_name), cv2.IMREAD_GRAYSCALE)
    if img1_raw is None: continue

    img1_raw = cv2.resize(img1_raw, (target_w, target_h))
        
    # img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//32*32, img1_raw.shape[0]//32*32))
    if precision == 'fp16':
        img1 = torch.from_numpy(img1_raw)[None][None].half().cuda() / 255.
    else:
        img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
    
    batch = {'image0': img0, 'image1': img1}
    
    torch.cuda.synchronize() 
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    
    with torch.no_grad():
        if precision == 'mp':
            with torch.autocast(enabled=True, device_type='cuda'):
                matcher(batch)
        else:
            matcher(batch)
        end_event.record()
    
    torch.cuda.synchronize()
    inference_time = start_event.elapsed_time(end_event)
    inference_times.append(inference_time)
            
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()

    # Draw
    raw_mconf_max = None
    if model_type == 'opt':
        raw_mconf_max = float(mconf.max())
        print(raw_mconf_max)
        mconf = (mconf - min(20.0, mconf.min())) / (max(30.0, mconf.max()) - min(20.0, mconf.min()))
    
    color = cm.jet(mconf)

    # normalize confidence of keypoints matches
    print(f"Min: {mconf.min()}, Max: {mconf.max()}, Mean: {mconf.mean()}")
    
    # filter keypoints
    threshold = 0.7 
    mask = mconf > threshold
    mkpts0_filtered = mkpts0[mask]
    mkpts1_filtered = mkpts1[mask]
    color_filtered = color[mask]
    
    num_inliers = 0
    if len(mkpts0_filtered) > 8:
        _, inliers = cv2.findFundamentalMat(mkpts0_filtered, mkpts1_filtered, cv2.USAC_MAGSAC, 0.5, 0.999, 1000)
        if inliers is not None:
            num_inliers = int(np.sum(inliers))
    
    inliers_geometric_number.append(num_inliers)
    inliers_number.append(len(mkpts0_filtered))
    confidences.append(mconf.mean())

    
    text = ['LoFTR', 'Matches: {}'.format(len(mkpts0_filtered))]
    fig = make_matching_figure(img0_raw, img1_raw, mkpts0_filtered, mkpts1_filtered, color_filtered, text=text)
    
    save_path = os.path.join(output_dir, f"match_{img_name}")
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.close(fig)
    
    print(f"Keyframe: {img_name} | Matches: {len(mkpts0_filtered)} | Inf Time {inference_time:.3f}ms")

    csv_rows.append({
        "image_name": img_name,
        "conf_min": float(mconf.min()),
        "conf_max": float(mconf.max()),
        "conf_mean": float(mconf.mean()),
        "matches": int(len(mkpts0_filtered)),
        "inliers": int(num_inliers),
        "inference_time_ms": float(inference_time),
        "threshold": float(threshold),
    })

summary_rows = [
    {
        "image_name": "__summary__",
        "conf_mean": float(np.mean(confidences)),
        "matches": float(np.mean(inliers_number)),
        "inliers": float(np.mean(inliers_geometric_number)),
        "inference_time_ms": float(sum(inference_times[1:])/(len(inference_times)-1)),
        "threshold": float(threshold),
        "note": "Mean values",
    }
]

fieldnames = ["image_name", "conf_min", "conf_max", "conf_mean", "matches", "inliers", "inference_time_ms", "threshold", "note"]

with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for row in csv_rows:
        writer.writerow(row)
    for row in summary_rows:
        writer.writerow(row)

print(f"Mean Inference Time: {sum(inference_times[1:])/(len(inference_times)-1):.3f}ms")
print(f"Mean Confidence: {np.mean(confidences)}")
print(f"Mean Number Inliers: {np.mean(inliers_number)} with confidence > {threshold}")
print(f"Saved CSV: {csv_path}")