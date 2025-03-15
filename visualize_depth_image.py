import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load depth image
depth_path = "datasets/sync/office_0003/sync_depth_00000.png"
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

# Load corresponding RGB image
rgb_path = "datasets/sync/office_0003/rgb_00000.jpg"  # <-- change to actual RGB image path
rgb_image = cv2.imread(rgb_path)
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Convert depth values to meters
depth_meters = depth.astype(np.float32) / 1000.0

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

# Original RGB image
axes[1].imshow(rgb_image)
axes[1].set_title("Original RGB Image")
axes[1].axis("off")

# Jet colormap depth map
im2 = axes[2].imshow(depth_meters, cmap='jet', interpolation='nearest', vmin=min(depth_values))
axes[2].set_title("Colored Depth Map (Jet Colormap)")
axes[2].axis("off")
fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Depth (meters)")

# Adjust layout
plt.tight_layout()
plt.show()
