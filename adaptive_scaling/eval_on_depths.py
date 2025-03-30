import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from timeit import default_timer as timer
from metric_depth_estimator import MetricDepthEstimator
from train import MaxDepthPredictor  # Load trained model
from torchvision import transforms

# Define transformation to match training preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# === Define Dataset Path ===
dataset_path = "datasets/sync/office_0026"

# === Load True Scaling Factors ===
true_scale_factors = dict(np.load("adaptive_depth/training_data_office_0026.npy", allow_pickle=True))

# === Load Trained Model for Scaling Factor Prediction ===
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
scale_model = MaxDepthPredictor().to(device)
scale_model.load_state_dict(torch.load("adaptive_depth/max_depth_predictor2.pth", map_location=device)["model_state"])
scale_model.eval()

# === Get Sorted List of RGB and Depth Files ===
rgb_files = sorted([f for f in os.listdir(dataset_path) if f.startswith("rgb_") and f.endswith(".jpg")])
depth_files = sorted([f for f in os.listdir(dataset_path) if f.startswith("sync_depth_") and f.endswith(".png")])

assert len(rgb_files) == len(depth_files), "Mismatch in RGB and Depth files!"

# === Load DA2 Model (Depth Anything V2) ===
initial_max_depth = 20.0
model = MetricDepthEstimator(encoder='vits', dataset='hypersim', max_depth=initial_max_depth)

# === Initialize Metrics ===
abs_rel_errors_true, rmse_errors_true = [], []
abs_rel_errors_pred, rmse_errors_pred = [], []
delta1_true, delta2_true, delta3_true = [], [], []
delta1_pred, delta2_pred, delta3_pred = [], [], []
inference_times = []

# === Process Each Image Pair ===
for i, (rgb_file, depth_file) in enumerate(tqdm(zip(rgb_files, depth_files), total=len(rgb_files), desc="Evaluating metric depth", unit="pair")):
    # Load RGB and Depth Images
    rgb_path = os.path.join(dataset_path, rgb_file)
    depth_path = os.path.join(dataset_path, depth_file)

    rgb = cv2.imread(rgb_path)
    depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if rgb is None or depth_gt is None:
        continue  # Skip if images not found

    # Convert Depth to Meters (If Necessary)
    if depth_gt.max() > 10:
        depth_gt = depth_gt.astype(np.float32) / 1000.0  # Convert mm → meters

    # === Run Depth Estimation ===
    start = timer()
    depth_pred = model.infer_image(rgb)
    end = timer()

    inference_times.append(end - start)

    if depth_pred is None or depth_pred.shape != depth_gt.shape:
        continue

    # === Get True & Predicted Scaling Factors ===
    true_scale_factor = true_scale_factors.get(rgb_path, np.log(1.0))  # Ensure we undo the log
    true_scale_factor = float(true_scale_factor)  # Ensure it's a float
    true_scale_factor = np.exp(true_scale_factor)  # Undo log-scaling
    
    with torch.no_grad():
        image = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        image_tensor = transform(image).unsqueeze(0).to(device)  # Ensure it has batch dimension
        predicted_log_scale = scale_model(image_tensor).item()
        predicted_scale_factor = np.exp(predicted_log_scale)

    # Rescale depth predictions
    depth_pred_true = depth_pred * true_scale_factor
    depth_pred_pred = depth_pred * predicted_scale_factor

    # === Compute Metrics ===
    valid_mask = (depth_gt > 0) & (depth_pred > 0)

    def compute_metrics(depth_rescaled):
        abs_rel = np.mean(np.abs(depth_rescaled[valid_mask] - depth_gt[valid_mask]) / depth_gt[valid_mask])
        rmse = np.sqrt(np.mean((depth_rescaled[valid_mask] - depth_gt[valid_mask]) ** 2))
        ratio = np.maximum(depth_rescaled[valid_mask] / depth_gt[valid_mask], depth_gt[valid_mask] / depth_rescaled[valid_mask])
        delta1 = np.mean(ratio < 1.25)
        delta2 = np.mean(ratio < 1.25 ** 2)
        delta3 = np.mean(ratio < 1.25 ** 3)
        return abs_rel, rmse, delta1, delta2, delta3

    abs_rel_t, rmse_t, d1_t, d2_t, d3_t = compute_metrics(depth_pred_true)
    abs_rel_p, rmse_p, d1_p, d2_p, d3_p = compute_metrics(depth_pred_pred)

    # Store Metrics
    abs_rel_errors_true.append(abs_rel_t)
    rmse_errors_true.append(rmse_t)
    delta1_true.append(d1_t)
    delta2_true.append(d2_t)
    delta3_true.append(d3_t)

    abs_rel_errors_pred.append(abs_rel_p)
    rmse_errors_pred.append(rmse_p)
    delta1_pred.append(d1_p)
    delta2_pred.append(d2_p)
    delta3_pred.append(d3_p)

    print(f"{i+1}/{len(rgb_files)} - True: AbsRel={abs_rel_t:.4f}, RMSE={rmse_t:.4f}, δ1={d1_t*100:.2f}% | Predicted: AbsRel={abs_rel_p:.4f}, RMSE={rmse_p:.4f}, δ1={d1_p*100:.2f}%")

# === Print Final Metrics ===
print("\n=== Depth Estimation Evaluation Results ===")
print("Using True Scale Factor:")
print(f"Mean AbsRel: {np.mean(abs_rel_errors_true):.4f}, RMSE: {np.mean(rmse_errors_true):.4f}")
print(f"δ1: {np.mean(delta1_true) * 100:.2f}%, δ2: {np.mean(delta2_true) * 100:.2f}%, δ3: {np.mean(delta3_true) * 100:.2f}%")

print("\nUsing Predicted Scale Factor:")
print(f"Mean AbsRel: {np.mean(abs_rel_errors_pred):.4f}, RMSE: {np.mean(rmse_errors_pred):.4f}")
print(f"δ1: {np.mean(delta1_pred) * 100:.2f}%, δ2: {np.mean(delta2_pred) * 100:.2f}%, δ3: {np.mean(delta3_pred) * 100:.2f}%")

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
