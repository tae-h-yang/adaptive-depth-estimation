import os
import cv2
import numpy as np
import torch
from timeit import default_timer as timer
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

# Dataset paths
dataset_path = "datasets/TUM_RGBD/rgbd_dataset_freiburg1_desk"
rgb_path = os.path.join(dataset_path, "rgb")
depth_path = os.path.join(dataset_path, "depth")

# Function to read timestamps from txt files
def read_timestamps(file_path):
    timestamps = []
    with open(file_path, "r") as f:
        for line in f:
            if not line.startswith("#"):  # Ignore comment lines
                timestamps.append(float(line.split()[0]))  # Extract timestamp
    return np.array(timestamps)

# Function to match RGB and depth timestamps
def match_timestamps(rgb_timestamps, depth_timestamps, tolerance=0.02):
    """ Finds the closest depth timestamp for each RGB timestamp within a 20ms tolerance """
    matched_pairs = []
    for rgb_ts in rgb_timestamps:
        idx = np.argmin(np.abs(depth_timestamps - rgb_ts))  # Find closest depth timestamp
        closest_depth_ts = depth_timestamps[idx]

        if abs(closest_depth_ts - rgb_ts) < tolerance:  # Ensure within tolerance
            matched_pairs.append((rgb_ts, closest_depth_ts))

    return matched_pairs

# Load timestamps from files
rgb_timestamps = read_timestamps(os.path.join(dataset_path, "rgb.txt"))
depth_timestamps = read_timestamps(os.path.join(dataset_path, "depth.txt"))

# Match timestamps within 20ms tolerance
matched_timestamps = match_timestamps(rgb_timestamps, depth_timestamps, tolerance=0.02)

print(f"Total matched image pairs: {len(matched_timestamps)}")

# Generate valid file paths
rgb_images = [os.path.join(rgb_path, f"{ts[0]:.6f}.png") for ts in matched_timestamps]
depth_images = [os.path.join(depth_path, f"{ts[1]:.6f}.png") for ts in matched_timestamps]

assert len(rgb_images) == len(depth_images), "Mismatch after timestamp alignment!"

# Load model
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder = 'vits'
dataset = 'hypersim'  # 'hypersim' for indoor, 'vkitti' for outdoor
max_depth = 20

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
# model = DepthAnythingV2(**{'encoder': encoder, 'max_depth': max_depth})
model.to(device)
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location=device))
model.eval()

# Metrics initialization
abs_rel_errors, rmse_errors, delta1_scores = [], [], []
inference_times = []  # Store inference times

def compute_metrics(pred, gt):
    """ Compute AbsRel, RMSE, and Delta1 for one pair of prediction and ground truth """
    valid_mask = (gt > 0)  # Ignore invalid depth values
    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]

    abs_rel = np.mean(np.abs(pred_valid - gt_valid) / gt_valid)
    rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))
    delta1 = np.mean(np.maximum(pred_valid / gt_valid, gt_valid / pred_valid) < 1.25)

    return abs_rel, rmse, delta1

# Function to compute average inference time and standard deviation
def compute_inference_time_stats(times):
    avg_time = np.mean(times)
    std_dev = np.std(times)
    return avg_time, std_dev

# Process images (limit to 1000 pairs)
for i, (rgb_file, depth_file) in enumerate(zip(rgb_images[:1000], depth_images[:1000])):
    rgb_img = cv2.imread(rgb_file)
    depth_gt = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED) / 5000.0  # Convert depth to meters

    start = timer()
    depth_pred = model.infer_image(rgb_img)
    end = timer()
    
    inference_time = end - start
    inference_times.append(inference_time)

    print(f"Inference time for {i+1}: {inference_time:.6f} seconds")

    # Compute metrics
    abs_rel, rmse, delta1 = compute_metrics(depth_pred, depth_gt)
    abs_rel_errors.append(abs_rel)
    rmse_errors.append(rmse)
    delta1_scores.append(delta1)

# Compute and print inference time statistics
avg_time, std_dev = compute_inference_time_stats(inference_times)
print(f"\n Inference Time Stats:")
print(f"   - Average Inference Time: {avg_time:.6f} seconds")
print(f"   - Standard Deviation: {std_dev:.6f} seconds")

# Print final results
print(f"\n Evaluation Metrics on TUM RGBD (1000 samples):")
print(f"   - Mean AbsRel: {np.mean(abs_rel_errors):.4f}")
print(f"   - Mean RMSE: {np.mean(rmse_errors):.4f}")
print(f"   - Mean Delta1: {np.mean(delta1_scores):.4f}")

''' Result: Vitl, 592 images
Inference Time Stats:
   - Average Inference Time: 0.787652 seconds
   - Standard Deviation: 0.054266 seconds

Evaluation Metrics on TUM RGBD (1592 samples):
   - Mean AbsRel: 0.2771
   - Mean RMSE: 0.3117
   - Mean Delta1: 0.4928
'''

''' Result: Vits, 592 images
Inference Time Stats:
   - Average Inference Time: 0.123127 seconds
   - Standard Deviation: 0.028356 seconds

 Evaluation Metrics on TUM RGBD (592 samples):
   - Mean AbsRel: 0.4565
   - Mean RMSE: 0.4706
   - Mean Delta1: 0.2547
'''