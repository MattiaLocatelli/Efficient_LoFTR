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

def rotm2rpy(R):
    # Returns [roll, pitch, yaw] in radians
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

def rotm2rpy_candidates(R):
    rpy_0 = rotm2rpy(R)
    rpy_1 = np.array([
        rpy_0[0] + np.pi,
        np.pi - rpy_0[1],
        rpy_0[2] + np.pi,
    ])
    rpy_0 = (rpy_0 + np.pi) % (2 * np.pi) - np.pi
    rpy_1 = (rpy_1 + np.pi) % (2 * np.pi) - np.pi
    return rpy_0, rpy_1

def calculate_rot_error(R_est, R_gt):
    R_rel = R_est @ R_gt.T
    val = (np.trace(R_rel) - 1) / 2.0
    return np.degrees(np.arccos(np.clip(val, -1.0, 1.0)))

def calculate_trans_error(t_est, t_gt):
    t_est_n = t_est.flatten() / (np.linalg.norm(t_est) + 1e-9)
    t_gt_n = t_gt.flatten() / (np.linalg.norm(t_gt) + 1e-9)
    return np.degrees(np.arccos(np.clip(np.dot(t_est_n, t_gt_n), -1.0, 1.0)))

def read_keyframe_data(filepath, num_descriptors=256):
    """
    Read keyframe data from a .dat file.
    Structure: 
    - type (1 byte)
    - closest_idx (4 byte)
    - num_kpts (4 byte)
    - keypoints (num_kpts * 8 byte)
    - descriptors (num_kpts * num_descriptors * 4 byte)
    - rotm (9 * 4 byte)
    - translation (3 * 4 byte)
    """
    with open(filepath, 'rb') as f:
        # Read the base header
        type_val = struct.unpack('B', f.read(1))[0]
        closest_idx = struct.unpack('i', f.read(4))[0]
        num_kpts = struct.unpack('i', f.read(4))[0]
        
        # Skip vector data
        f.seek(num_kpts * 8, os.SEEK_CUR)              # keypoints (cv::Point2f)
        f.seek(num_kpts * num_descriptors * 4, os.SEEK_CUR) # descriptors (float)
        
        # Read the pose from the tail of the file to stay robust to extra fields before it.
        f.seek(-48, os.SEEK_END)
        rotm = np.frombuffer(f.read(36), dtype=np.float32).reshape(3, 3, order='F')
        trans = np.frombuffer(f.read(12), dtype=np.float32)
        
        return rotm, trans
    
def shortest_angular_distance_deg(angle_a, angle_b):
    return np.degrees(np.arctan2(np.sin(angle_a - angle_b), np.cos(angle_a - angle_b)))

# You can choose model type in ['full', 'opt']
model_type = 'full' # 'full' for best quality, 'opt' for best efficiency

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

output_dir = "output_matches_fp32"
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "ELoFTR_fp32_stats.csv")

# 2. Online frame load
img0_raw = cv2.imread(online_img_pth, cv2.IMREAD_GRAYSCALE)
target_w, target_h = 960, 256 #almost half the original size (1920x500)
img0_raw = cv2.resize(img0_raw, (target_w, target_h))
img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//32*32, img0_raw.shape[0]//32*32))

# Define intrinsic camera matrix K
# Original calibration
fx_orig, fy_orig = 1593.4, 1587.3
cx_orig, cy_orig = 962.8, 369.6
w_orig, h_orig = 1928, 500

# Resize factors
scale_x = target_w / w_orig
scale_y = target_h / h_orig

# Scaled K
K = np.array([
    [fx_orig * scale_x, 0, cx_orig * scale_x],
    [0, fy_orig * scale_y, cy_orig * scale_y],
    [0, 0, 1]
], dtype=np.float32)

# Distortion coefficients (dist_coeffs)
dist_coeffs = np.array([-0.3860, 0.2234, -0.0009666, -0.00026557, -0.0785])

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
    mkpts0_inliers, mkpts1_inliers, color_inliers = [], [], []
    
    dat_path = os.path.join("Offline_Keyframes_Turn2-3/", img_name.replace('.png', '.dat'))
    online_dat_path = os.path.join(os.path.dirname(online_img_pth), os.path.splitext(os.path.basename(online_img_pth))[0] + ".dat")

    kf_rot = None
    kf_trans = None
    approx_rot = None
    approx_trans = None
    
    if os.path.exists(dat_path):
        kf_rot, kf_trans = read_keyframe_data(dat_path)

    if os.path.exists(online_dat_path):
        approx_rot, approx_trans = read_keyframe_data(online_dat_path)
    else:
        print(f"Warning: online pose file not found: {online_dat_path}")

    # FIVE-POINTS ALGORITHM
    # 1. Undistort keypoints using K and dist_coeffs
    mkptsF_undistorted = cv2.undistortPoints(mkpts0_filtered, K, dist_coeffs, P=K)
    mkptsK_undistorted = cv2.undistortPoints(mkpts1_filtered, K, dist_coeffs, P=K)

    # 2. Use Essential Matrix with undistorted points
    ptsF = mkptsF_undistorted.reshape(-1, 2)
    ptsK = mkptsK_undistorted.reshape(-1, 2)

    rot_err, trans_err = 0.0, 0.0
    roll_err = 0.0
    pitch_err = 0.0
    yaw_err = 0.0

    if len(ptsF) >= 5 and kf_rot is not None and kf_trans is not None and approx_rot is not None and approx_trans is not None:
        E, mask = cv2.findEssentialMat(ptsK, ptsF, K, cv2.LMEDS, 0.999, 1.0)
        if E is not None:
            # Recover relative pose: keyframe -> online frame, matching the ROS convention.
            _, R_KF, t_KF, mask_pose = cv2.recoverPose(E, ptsK, ptsF, K)
            
            # Transform the recovered relative rotation into the world frame, matching camera_worker.cpp.
            R_est_world = kf_rot @ R_KF.T

            # Match the ROS-style Euler ambiguity handling used in camera_worker.cpp.
            rpy_hypothesis_0, rpy_hypothesis_1 = rotm2rpy_candidates(R_est_world)
            rpy_ref = rotm2rpy(approx_rot)

            yaw_err_0 = abs(shortest_angular_distance_deg(rpy_hypothesis_0[2], rpy_ref[2]))
            yaw_err_1 = abs(shortest_angular_distance_deg(rpy_hypothesis_1[2], rpy_ref[2]))
            if yaw_err_0 <= yaw_err_1:
                chosen_rpy = rpy_hypothesis_0
            else:
                chosen_rpy = rpy_hypothesis_1

            roll_err = shortest_angular_distance_deg(chosen_rpy[0], rpy_ref[0])
            pitch_err = shortest_angular_distance_deg(chosen_rpy[1], rpy_ref[1])
            yaw_err = shortest_angular_distance_deg(chosen_rpy[2], rpy_ref[2])
            
            # t_est_world = R_kf * R_KF.T * (-t_KF)
            t_est_world = kf_rot @ R_KF.T @ (-t_KF.flatten())
            
            # Direction vectors for translation error.
            dir_est = t_est_world / (np.linalg.norm(t_est_world) + 1e-9)
            dir_true = (approx_trans.flatten() - kf_trans.flatten()) / (np.linalg.norm(approx_trans.flatten() - kf_trans.flatten()) + 1e-9)
            
            # Calculate angular errors.
            dot_val = np.clip(np.dot(dir_est, dir_true), -1.0, 1.0)
            trans_err = np.degrees(np.arccos(dot_val))
            rot_err = calculate_rot_error(R_est_world, approx_rot)

            # Check the four equivalent recoverPose hypotheses, as in camera_worker.
            rec_pose_R = [R_KF, R_KF, -R_KF, -R_KF]
            rec_pose_t = [t_KF.flatten(), -t_KF.flatten(), t_KF.flatten(), -t_KF.flatten()]
            rec_pose_ang_err = []
            for rel_R, rel_t in zip(rec_pose_R, rec_pose_t):
                dir_kf = kf_rot @ rel_R.T @ rel_t
                dir_kf = dir_kf / (np.linalg.norm(dir_kf) + 1e-9)
                rec_pose_ang_err.append(np.degrees(np.arccos(np.clip(np.dot(dir_kf, dir_true), -1.0, 1.0))))
            
            # Inlier mask kept for geometric filtering, but the CSV stores only pose errors.
            mask_inliers = mask_pose.flatten() > 0
            
            print(f"Roll Err: {roll_err:.2f} deg | Pitch Err: {pitch_err:.2f} deg | Yaw Err: {yaw_err:.2f} deg | Trans Err: {trans_err:.2f} deg")
            print(f"RecoverPose angle checks: {[round(v, 2) for v in rec_pose_ang_err]}")

            num_inliers = int(np.sum(mask_inliers))
            mkpts0_inliers = mkpts0_filtered[mask_inliers]
            mkpts1_inliers = mkpts1_filtered[mask_inliers]
            color_inliers = color_filtered[mask_inliers]
            
            # print(f"Estimated R:\n{R_est}")
            # print(f"Ground Truth R:\n{gt_rot}")
            # print(f"Estimated t:\n{t_est.flatten()}")
            # print(f"Ground Truth t:\n{gt_trans}")

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
        "roll_error_deg": float(roll_err),
        "pitch_error_deg": float(pitch_err),
        "yaw_error_deg": float(yaw_err),
        "trans_error_deg": float(trans_err),
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
        "roll_error_deg": float(np.mean(np.abs([row["roll_error_deg"] for row in csv_rows]))),
        "pitch_error_deg": float(np.mean(np.abs([row["pitch_error_deg"] for row in csv_rows]))),
        "yaw_error_deg": float(np.mean(np.abs([row["yaw_error_deg"] for row in csv_rows]))),
        "trans_error_deg": float(np.mean(np.abs([row["trans_error_deg"] for row in csv_rows]))),
        "note": "Mean absolute values",
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
        "roll_error_deg": float(np.std([row["roll_error_deg"] for row in csv_rows])**2),
        "pitch_error_deg": float(np.std([row["pitch_error_deg"] for row in csv_rows])**2),
        "yaw_error_deg": float(np.std([row["yaw_error_deg"] for row in csv_rows])**2),
        "trans_error_deg": float(np.std([row["trans_error_deg"] for row in csv_rows])**2),
        "note": "Variance values",
    }
]

fieldnames = ["image_name", 
              "conf_min", 
              "conf_max", 
              "conf_mean", 
              "percentage_matches",
              "matches",
              "percentage_inliers",
              "inliers",
              "inference_time_ms",
              "threshold",
              "roll_error_deg",
              "pitch_error_deg",
              "yaw_error_deg",
              "trans_error_deg",
              "note"
              ]

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