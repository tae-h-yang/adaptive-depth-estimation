import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load depth image
depth_path = "/Users/tyang/repos/cs229-project/datasets/TUM_RGBD/rgbd_dataset_freiburg1_desk/depth/1305031453.374112.png"
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

# Convert depth values to meters
depth_meters = depth.astype(np.float32) / 5000.0

# Remove zero values (missing depth) for histogram
depth_values = depth_meters[depth_meters > 0]

# Create a figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Histogram of depth values
axes[0].hist(depth_values.flatten(), bins=100, color='blue', alpha=0.7)
axes[0].set_xlabel("Depth Value (meters)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Histogram of Depth Values")
axes[0].grid(True)

# Grayscale depth map
im1 = axes[1].imshow(depth_meters, cmap='gray', interpolation='nearest', vmin=min(depth_values))
axes[1].set_title("Depth Map in Grayscale")
axes[1].axis("off")
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Depth (meters)")

# Jet colormap depth map
im2 = axes[2].imshow(depth_meters, cmap='jet', interpolation='nearest', vmin=min(depth_values))
axes[2].set_title("Colored Depth Map (Jet Colormap)")
axes[2].axis("off")
fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Depth (meters)")

# Adjust layout
plt.tight_layout()
plt.show()
