import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from math import exp

# Import trained model
from train import MaxDepthPredictor  

# Custom Dataset for Evaluation
class MaxDepthEvalDataset(Dataset):
    def __init__(self, data_file):
        self.data = np.load(data_file, allow_pickle=True)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),  # Ensure consistency with training
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path, true_log_scale_factor = self.data[idx]  # True log-scaled value
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        return image, torch.tensor(float(true_log_scale_factor), dtype=torch.float32), image_path

# Load Model
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
model = MaxDepthPredictor().to(device)
checkpoint_path = "adaptive_depth/adaptive_depth_predictor2.pth"

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print("Loaded trained model for evaluation.")
else:
    raise FileNotFoundError("Checkpoint file not found! Train the model first.")

model.eval()

# Load Dataset
eval_dataset = MaxDepthEvalDataset("adaptive_depth_depth/training_data_office_0003.npy")  # Change path if needed
eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

# Evaluation Metrics
mae, rmse = 0.0, 0.0
total_samples = 0
predictions = []

# Evaluation Loop
with torch.no_grad():
    for images, true_log_scale_factors, image_paths in tqdm(eval_loader, desc="Evaluating Model"):
        images = images.to(device)
        true_log_scale_factors = true_log_scale_factors.to(device)
        
        predicted_log_scale_factors = model(images)  # Predict log scale factor
        predicted_scale_factors = torch.exp(predicted_log_scale_factors)  # Convert back to scale factor
        true_scale_factors = torch.exp(true_log_scale_factors)  # Convert back to scale factor

        # Compute errors
        abs_error = torch.abs(predicted_scale_factors - true_scale_factors)
        squared_error = (predicted_scale_factors - true_scale_factors) ** 2
        
        mae += abs_error.sum().item()
        rmse += squared_error.sum().item()
        total_samples += 1

        # Store predictions for logging
        predictions.append((image_paths[0], predicted_scale_factors.item(), true_scale_factors.item()))
        print(f"{image_paths[0]}, true scale factor: {true_scale_factors}, predicted scale factor: {predicted_scale_factors:.4f}")

# Final Metrics
mae /= total_samples
rmse = (rmse / total_samples) ** 0.5

print("\n=== Model Evaluation Results ===")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Save predictions to a file for further analysis
np.save("adaptive_depth/predicted_vs_true.npy", predictions)
print("Predictions saved for further analysis.")
