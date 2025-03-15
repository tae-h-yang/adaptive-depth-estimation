import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from metric_depth_estimator import MetricDepthEstimator
from train import MaxDepthPredictor  # Trained scaling factor model

# === Define Dataset Path ===
dataset_path = "datasets/sync/office_0026"

# === Load True Scaling Factors ===
true_scale_factors = dict(np.load("adaptive_depth/training_data_office_0026.npy", allow_pickle=True))

# === Load Depth Anything V2 Model ===
initial_max_depth = 20.0
model = MetricDepthEstimator(encoder='vits', dataset='hypersim', max_depth=initial_max_depth)

# === Load Trained Model for Scaling Factor Prediction ===
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
scale_model = MaxDepthPredictor().to(device)
scale_model.load_state_dict(torch.load("adaptive_depth/max_depth_predictor2.pth", map_location=device)["model_state"])
scale_model.eval()

# === Get Sorted List of RGB and Depth Files ===
rgb_files = sorted([f for f in os.listdir(dataset_path) if f.startswith("rgb_") and f.endswith(".jpg")])
depth_files = sorted([f for f in os.listdir(dataset_path) if f.startswith("sync_depth_") and f.endswith(".png")])

assert len(rgb_files) == len(depth_files), "Mismatch in RGB and Depth files!"

# === Process Only the First Image ===
rgb_path = os.path.join(dataset_path, rgb_files[75])
depth_path = os.path.join(dataset_path, depth_files[75])

# Load RGB and Depth Images
rgb = cv2.imread(rgb_path)
depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

# Convert Depth to Meters
if depth_gt.max() > 10:
    depth_gt = depth_gt.astype(np.float32) / 1000.0  # Convert mm → meters

# === Run Depth Estimation ===
depth_pred = model.infer_image(rgb)

# Ensure depth prediction shape matches
if depth_pred is None or depth_pred.shape != depth_gt.shape:
    raise ValueError("Depth prediction failed or shape mismatch!")

# === Get True Scaling Factor (from .npy file) ===
true_scale_factor = true_scale_factors.get(rgb_path, np.log(1.0))  # Ensure we undo the log
true_scale_factor = float(true_scale_factor)  
true_scale_factor = np.exp(true_scale_factor)  # Undo log-scaling

# === Get Predicted Scaling Factor ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

image_tensor = transform(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
with torch.no_grad():
    predicted_log_scale = scale_model(image_tensor).item()
    predicted_scale_factor = np.exp(predicted_log_scale)  # Undo log-scaling

print(f"True Scale Factor: {true_scale_factor:.4f}")
print(f"Predicted Scale Factor: {predicted_scale_factor:.4f}")

# Rescale Predicted Depth Maps
depth_pred_true = depth_pred * true_scale_factor
depth_pred_pred = depth_pred * predicted_scale_factor

# === Save Depth Maps as NPY Files ===
np.save("adaptive_depth/pointcloud/rescaled_depth_gt.npy", depth_pred_true)
np.save("adaptive_depth/pointcloud/rescaled_depth_predicted.npy", depth_pred_pred)

print("Saved rescaled depth maps:")
print("- rescaled_depth_gt.npy (Using ground truth scale)")
print("- rescaled_depth_predicted.npy (Using predicted scale)")
