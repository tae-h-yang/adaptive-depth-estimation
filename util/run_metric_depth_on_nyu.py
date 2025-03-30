import os
import cv2
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
from metric_depth_estimator import MetricDepthEstimator

# === Define Dataset Path ===
dataset_path = "/Users/tyang/repos/cs229-project/datasets/sync/office_0003"

# === Get Sorted List of RGB and Depth Files ===
rgb_files = sorted([f for f in os.listdir(dataset_path) if f.startswith("rgb_") and f.endswith(".jpg")])
depth_files = sorted([f for f in os.listdir(dataset_path) if f.startswith("sync_depth_") and f.endswith(".png")])

assert len(rgb_files) == len(depth_files), "Mismatch in RGB and Depth files!"

# === Load DA2 Model (Depth Anything V2) ===
model = MetricDepthEstimator(encoder='vits', dataset='hypersim', max_depth=20)

# === Initialize Metrics ===
abs_rel_errors, rmse_errors, delta1, delta2, delta3 = [], [], [], [], []
inference_times = []  # Track inference times

# === Process Each Image Pair ===
for i, (rgb_file, depth_file) in enumerate(tqdm(zip(rgb_files, depth_files), total=len(rgb_files), desc="Evaluating metric depth", unit="pair")):
    # Load RGB and Depth Images
    rgb_path = os.path.join(dataset_path, rgb_file)
    depth_path = os.path.join(dataset_path, depth_file)

    rgb = cv2.imread(rgb_path)
    depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if rgb is None or depth_gt is None:
        continue  # Skip if images not found

    # === Convert Depth to Meters (If Necessary) ===
    if depth_gt.max() > 10:  # Heuristic check (if depth is in mm, convert to meters)
        depth_gt = depth_gt.astype(np.float32) / 1000.0  # Convert mm → meters

    # === Run DA2 on RGB Image ===
    start = timer()
    depth_pred = model.infer_image(rgb)  # Ensure output is in meters
    end = timer()

    inference_times.append(end - start)  # Track inference time

    if depth_pred is None or depth_pred.shape != depth_gt.shape:
        continue  # Skip invalid predictions

    # === Compute Metrics ===
    valid_mask = (depth_gt > 0) & (depth_pred > 0)  # Ignore invalid pixels

    abs_rel = np.mean(np.abs(depth_pred[valid_mask] - depth_gt[valid_mask]) / depth_gt[valid_mask])
    rmse = np.sqrt(np.mean((depth_pred[valid_mask] - depth_gt[valid_mask])**2))

    # Accuracy Thresholds
    ratio = np.maximum(depth_pred[valid_mask] / depth_gt[valid_mask], depth_gt[valid_mask] / depth_pred[valid_mask])
    delta1.append(np.mean(ratio < 1.25))
    delta2.append(np.mean(ratio < 1.25 ** 2))
    delta3.append(np.mean(ratio < 1.25 ** 3))

    # Store Errors
    abs_rel_errors.append(abs_rel)
    rmse_errors.append(rmse)

    print(f"{i+1}/{len(rgb_files)} - AbsRel: {abs_rel:.4f}, RMSE: {rmse:.4f}, δ1: {delta1[-1]*100:.2f}%")

# === Print Final Metrics ===
print("\n=== Depth Estimation Evaluation Results ===")
print(f"Mean Absolute Relative Error (AbsRel): {np.mean(abs_rel_errors):.4f}")
print(f"Root Mean Squared Error (RMSE): {np.mean(rmse_errors):.4f}")
print(f"Accuracy δ1 (<1.25): {np.mean(delta1) * 100:.2f}%")
print(f"Accuracy δ2 (<1.25²): {np.mean(delta2) * 100:.2f}%")
print(f"Accuracy δ3 (<1.25³): {np.mean(delta3) * 100:.2f}%")

# === Compute Inference Time Statistics ===
avg_time = np.mean(inference_times)
std_time = np.std(inference_times)
p95_time = np.percentile(inference_times, 95)
max_time = np.max(inference_times)

# === Print Inference Time Evaluation ===
print("\n=== Inference Time Analysis ===")
print(f"Average Inference Time: {avg_time:.4f} sec per image")
print(f"Standard Deviation: {std_time:.4f} sec")
print(f"95th Percentile Latency: {p95_time:.4f} sec")
print(f"Max Inference Time (Worst Case): {max_time:.4f} sec")
