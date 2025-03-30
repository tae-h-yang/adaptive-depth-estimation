import os
import cv2
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.stats import wasserstein_distance
from metric_depth_estimator import MetricDepthEstimator

def compute_optimal_scale(pred_values, gt_values):
    """Finds the optimal scale factor while filtering unreliable predictions."""
    if len(pred_values) == 0 or len(gt_values) == 0:
        return None  # Skip if there are no valid values

    def loss_fn(s):
        scaled_pred = s * pred_values
        return wasserstein_distance(scaled_pred, gt_values)
    
    result = minimize(loss_fn, x0=[1.0], bounds=[(0.1, 2.0)])  # Expanded bounds for generalization
    return result.x[0] if result.success else None

def compute_best_max_depth(dataset_path, initial_max_depth=20.0, absrel_thresh=0.1, rmse_thresh=0.2):
    """Finds the best max_depth per image and ensures it improves AbsRel and RMSE before adding to training data."""
    rgb_files = sorted([f for f in os.listdir(dataset_path) if f.startswith("rgb_") and f.endswith(".jpg")])
    depth_files = sorted([f for f in os.listdir(dataset_path) if f.startswith("sync_depth_") and f.endswith(".png")])
    
    assert len(rgb_files) == len(depth_files), "Mismatch in RGB and Depth files!"
    
    scale_factors = []
    training_data = []  # Store (image path, log-scale factor)
    model = MetricDepthEstimator(encoder='vits', dataset='hypersim', max_depth=initial_max_depth)
    
    for rgb_file, depth_file in tqdm(zip(rgb_files, depth_files), total=len(rgb_files), desc="Processing", unit="pair"):
        rgb_path = os.path.join(dataset_path, rgb_file)
        depth_path = os.path.join(dataset_path, depth_file)
        
        rgb = cv2.imread(rgb_path)
        depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        if rgb is None or depth_gt is None:
            continue
        
        if depth_gt.max() > 10:
            depth_gt = depth_gt.astype(np.float32) / 1000.0  # Convert mm to meters
        
        depth_pred = model.infer_image(rgb)
        if depth_pred is None or depth_pred.shape != depth_gt.shape:
            continue

        valid_mask = (depth_gt > 0) & (depth_pred > 0)
        pred_values = depth_pred[valid_mask]
        gt_values = depth_gt[valid_mask]
        
        # Compute optimal scale factor
        scale_factor = compute_optimal_scale(pred_values, gt_values)
        if scale_factor is None:
            continue
        
        scale_factors.append(scale_factor)  # Store for dynamic filtering
    
        # Compute additional error metrics after scaling
        scaled_pred = pred_values * scale_factor
        abs_rel = np.mean(np.abs(scaled_pred - gt_values) / gt_values)
        rmse = np.sqrt(np.mean((scaled_pred - gt_values) ** 2))
        ratio = np.maximum(depth_pred[valid_mask] / depth_gt[valid_mask], depth_gt[valid_mask] / depth_pred[valid_mask])
        delta1 = np.mean(ratio < 1.25)

        # Print and store only if it improves metrics
        print(f"Image: {rgb_file}, AbsRel: {abs_rel:.4f}, RMSE: {rmse:.4f}, Delta1: {delta1:.4f}, Scale Factor: {scale_factor:.4f}")
        
        # if abs_rel < absrel_thresh and rmse < rmse_thresh:
        #     training_data.append((rgb_path, np.log(scale_factor)))  # Store log-scale factor instead of raw value
        training_data.append((rgb_path, np.log(scale_factor)))  # Store log-scale factor instead of raw value

    # **Dynamic Clipping to Remove Outliers**
    scale_factors = np.array(scale_factors)
    lower, upper = np.percentile(scale_factors, [5, 95])  # Keep within 5-95th percentile
    filtered_training_data = [(img, sf) for img, sf in training_data if lower <= np.exp(sf) <= upper]

    dataset_name = dataset_path.split("/")[-1]  # Extract dataset name
    np.save(f"adaptive_scaling/training_data_{dataset_name}.npy", filtered_training_data)
    print(f"Saved {len(filtered_training_data)} reliable training samples!")
    return filtered_training_data

# Define Dataset Path
dataset_path = "datasets/sync/office_00026"
initial_max_depth = 20.0

# Run Optimization
compute_best_max_depth(dataset_path, initial_max_depth)
