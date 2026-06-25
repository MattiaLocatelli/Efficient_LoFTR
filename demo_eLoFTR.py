import os
# os.chdir("..")
from copy import deepcopy
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

# 1. Configurazione per la pipeline
online_img_pth = "Online_Keyframe/R1257.png"
offline_folder = "Offline_Keyframes_Turn2-3/"
offline_imgs = [f for f in os.listdir(offline_folder) if f.endswith('.png')]

output_dir = "output_matches"
os.makedirs(output_dir, exist_ok=True)

# 2. Caricamento immagine online (una volta sola)
img0_raw = cv2.imread(online_img_pth, cv2.IMREAD_GRAYSCALE)
target_w, target_h = 960, 256 #almost half the original size (1920x500)
img0_raw = cv2.resize(img0_raw, (target_w, target_h))
img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//32*32, img0_raw.shape[0]//32*32))
img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.

inference_times = []

# 3. Pipeline di matching
for img_name in offline_imgs:
    img1_raw = cv2.imread(os.path.join(offline_folder, img_name), cv2.IMREAD_GRAYSCALE)
    if img1_raw is None: continue
    
    # img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//32*32, img1_raw.shape[0]//32*32))
    img1_raw = cv2.resize(img1_raw, (target_w, target_h))
    img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
    
    batch = {'image0': img0, 'image1': img1}
    
    torch.cuda.synchronize() 
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    
    with torch.no_grad():
        matcher(batch)
        
        end_event.record()
    
    torch.cuda.synchronize()
    inference_time = start_event.elapsed_time(end_event)
    inference_times.append(inference_time)
            
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()

    # Draw
    if model_type == 'opt':
        print(mconf.max())
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
    
    text = ['LoFTR', 'Matches: {}'.format(len(mkpts0_filtered))]
    fig = make_matching_figure(img0_raw, img1_raw, mkpts0_filtered, mkpts1_filtered, color_filtered, text=text)
    
    save_path = os.path.join(output_dir, f"match_{img_name}")
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.close(fig)
    
    print(f"Keyframe: {img_name} | Matches: {len(mkpts0_filtered)} | Inf Time {inference_time:.3f}ms")

print(f"Mean Inference Time: {np.mean(inference_times):.3f}")