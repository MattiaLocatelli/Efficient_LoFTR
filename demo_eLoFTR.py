import os
import struct
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

def read_keyframe_data(filepath, num_descriptors=256):
    """
    Legge i dati dal file .dat.
    Struttura: 
    - type (1 byte)
    - closest_idx (4 byte)
    - num_kpts (4 byte, solitamente scritto prima dei vettori)
    - keypoints (num_kpts * 8 byte)
    - descriptors (num_kpts * num_descriptors * 4 byte)
    - rotm (9 * 4 byte)
    - translation (3 * 4 byte)
    """
    with open(filepath, 'rb') as f:
        # Leggi header base
        type_val = struct.unpack('B', f.read(1))[0]
        closest_idx = struct.unpack('i', f.read(4))[0]
        num_kpts = struct.unpack('i', f.read(4))[0]
        
        # Salta i dati vettoriali
        f.seek(num_kpts * 8, os.SEEK_CUR)              # keypoints (cv::Point2f)
        f.seek(num_kpts * num_descriptors * 4, os.SEEK_CUR) # descriptors (float)
        
        # Leggi Rotazione (9 float = 36 byte) e Traslazione (3 float = 12 byte)
        rotm = np.frombuffer(f.read(36), dtype=np.float32).reshape(3, 3)
        trans = np.frombuffer(f.read(12), dtype=np.float32)
        
        return rotm, trans

# You can choose model type in ['full', 'opt']
model_type = 'opt' # 'full' for best quality, 'opt' for best efficiency

# You can choose numerical precision in ['fp32', 'mp', 'fp16']. 'fp16' for best efficiency
precision = 'fp16' # Enjoy near-lossless precision with Mixed Precision (MP) / FP16 computation if you have a modern GPU (recommended NVIDIA architecture >= SM_70).

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

output_dir = "output_matches_fp16"
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "ELoFTR_fp16_stats.csv")

# 2. Online frame load
img0_raw = cv2.imread(online_img_pth, cv2.IMREAD_GRAYSCALE)
target_w, target_h = 960, 256 #almost half the original size (1920x500)
img0_raw = cv2.resize(img0_raw, (target_w, target_h))
img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//32*32, img0_raw.shape[0]//32*32))

# Define intrinsic camera matrix K
# K = np.array([
#     [900.0, 0.0, target_w / 2.0],
#     [0.0, 900.0, target_h / 2.0],
#     [0.0, 0.0, 1.0]
# ], dtype=np.float32)

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
    threshold = 0.0
    mask = mconf > threshold
    mkpts0_filtered = mkpts0[mask]
    mkpts1_filtered = mkpts1[mask]
    color_filtered = color[mask]
    
    num_inliers = 0
    if len(mkpts0_filtered) > 8:
        _, inliers = cv2.findFundamentalMat(mkpts0_filtered, mkpts1_filtered, cv2.USAC_MAGSAC, 0.5, 0.999, 1000)
        if inliers is not None:
            mask_inliers = inliers.flatten() > 0
            num_inliers = int(np.sum(mask_inliers))
            
            # Seleziona solo gli inliers per il disegno
            mkpts0_inliers = mkpts0_filtered[mask_inliers]
            mkpts1_inliers = mkpts1_filtered[mask_inliers]
            color_inliers = color_filtered[mask_inliers]
    
    inliers_geometric_number.append(num_inliers)
    inliers_number.append(len(mkpts0_filtered))
    confidences.append(mconf.mean())
    
    text = ['Efficient LoFTR', 'Inliers: {}'.format(len(mkpts0_inliers))]
    fig = make_matching_figure(img0_raw, img1_raw, mkpts0_inliers, mkpts1_inliers, color_inliers, text=text)
    
    save_path = os.path.join(output_dir, f"match_{img_name}")
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.close(fig)
    
    print(f"Keyframe: {img_name} | Matches: {len(mkpts0_filtered)} | Inf Time {inference_time:.3f}ms")

    csv_rows.append({
        "image_name": img_name,
        "conf_min": float(mconf.min()),
        "conf_max": float(mconf.max()),
        "conf_mean": float(mconf.mean()),
        "percentage_matches": float(len(mkpts0_filtered) / len(mconf) if len(mconf) > 0 else 0),
        "matches": int(len(mkpts0_filtered)),
        "percentage_inliers": float(num_inliers / len(mkpts0_filtered) if len(mkpts0_filtered) > 0 else 0),
        "inliers": int(num_inliers),
        "inference_time_ms": float(inference_time),
        "threshold": float(threshold),
    })

summary_rows = [
    {
        "image_name": "__summary__",
        "conf_mean": float(np.mean(confidences)),
        "percentage_matches": float(np.mean([row["percentage_matches"] for row in csv_rows])),
        "matches": float(np.mean(inliers_number)),
        "percentage_inliers": float(np.mean([row["percentage_inliers"] for row in csv_rows])),
        "inliers": float(np.mean(inliers_geometric_number)),
        "inference_time_ms": float(sum(inference_times[1:])/(len(inference_times)-1)),
        "threshold": float(threshold),
        "note": "Mean values",
    },
    {
        "image_name": "__summary__",
        "conf_mean": float(np.std(confidences)**2),
        "percentage_matches": float(np.std([row["percentage_matches"] for row in csv_rows])**2),
        "matches": float(np.std(inliers_number)),
        "percentage_inliers": float(np.std([row["percentage_inliers"] for row in csv_rows])**2),
        "inliers": float(np.std(inliers_geometric_number)),
        "inference_time_ms": float(np.std(sum(inference_times[1:])/(len(inference_times)-1))**2),
        "threshold": float(threshold),
        "note": "Variance values",
    }
]

fieldnames = ["image_name", "conf_min", "conf_max", "conf_mean", "percentage_matches", "matches", "percentage_inliers", "inliers", "inference_time_ms", "threshold", "note"]

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