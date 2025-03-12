import cv2
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
from metric_depth_estimator import MetricDepthEstimator
from util import associate_time_stamps, evaluate_matched_pairs

# === Load Matched RGB and Depth Image Pairs ===
dataset_path = "datasets/TUM_RGBD/rgbd_dataset_freiburg1_desk"
rgb_file = f"{dataset_path}/rgb.txt"
depth_file = f"{dataset_path}/depth.txt"

# max_difference is reduced to 0.01 from default 0.02
matched_pairs = associate_time_stamps(rgb_file, depth_file, max_difference=0.01)
# matched_pairs = associate_time_stamps(rgb_file, depth_file, max_difference=0.01)

# === Load DA2 Model ===
model = MetricDepthEstimator(encoder='vitl', dataset='hypersim', max_depth=13)

# === Initialize Metrics ===
abs_rel_errors, rmse_errors, delta1, delta2, delta3 = [], [], [], [], []
inference_times = []  # Track inference times

# evaluate_matched_pairs(matched_pairs)

# === Process Each Matched Image Pair ===
for _, rgb_path, _, depth_path in tqdm(matched_pairs, desc="Evaluting metric depth estimation", unit="pair"):
    # Load RGB and Depth Images
    # Extract file paths from lists
    rgb = cv2.imread(f"{dataset_path}/{rgb_path[0]}")
    depth_gt = cv2.imread(f"{dataset_path}/{depth_path[0]}", cv2.IMREAD_UNCHANGED)

    if rgb is None or depth_gt is None:
        continue  # Skip if images not found

    # Convert depth to meters (confirm correct scaling factor)
    depth_gt = depth_gt.astype(np.float32) / 5000.0  # Convert to meters

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
