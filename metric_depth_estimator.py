import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from util import inspect_depth

class MetricDepthEstimator(DepthAnythingV2):
    def __init__(self, encoder='vitl', dataset='hypersim', max_depth=20):
        """
        Initialize the Depth Anything v2 model.

        Parameters:
        - encoder: 'vits', 'vitb', or 'vitl' (determines model size)
        - dataset: 'hypersim' (indoor) or 'vkitti' (outdoor)
        - max_depth: Maximum depth range (20m for indoor, 80m for outdoor)
        """
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

        # Model configurations
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }

        if encoder not in model_configs:
            raise ValueError(f"Invalid encoder '{encoder}'. Choose from: 'vits', 'vitb', 'vitl'.")

        self.encoder = encoder
        self.dataset = dataset
        self.max_depth = max_depth
        self.model_path = f"checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth"

        # Initialize parent class (DepthAnythingV2) with chosen parameters
        super().__init__(**{**model_configs[encoder], 'max_depth': max_depth})

        # Move model to the correct device
        self.to(self.device)
        self._load_model()

    def _load_model(self):
        """Loads the model weights."""
        print(f"Loading model: {self.model_path} on {self.device}")
        try:
            self.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.eval()
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint not found: {self.model_path}")

    def infer(self, image_path):
        """
        Runs inference on a given image.

        Parameters:
        - image_path: Path to the input image.

        Returns:
        - depth_map: Numpy array representing the estimated depth.
        """
        raw_img = cv2.imread(image_path)
        if raw_img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        metric_depth = self.infer_image(raw_img)  # Directly call `infer_image()` from parent class
        return metric_depth

if __name__ == "__main__":
    metric_depth_estimator = MetricDepthEstimator(encoder='vits', dataset='hypersim', max_depth=20)
    start = timer()
    rgb_path = "datasets/sync/office_0019/rgb_00000.jpg"
    # metric_depth = metric_depth_estimator.infer('datasets/TUM_RGBD/rgbd_dataset_freiburg1_desk/rgb/1305031453.359684.png')
    metric_depth = metric_depth_estimator.infer(rgb_path)
    end = timer()
    print(f"Inference time: {end - start:.6f} seconds")

    # inspect_depth(metric_depth)

    # Load corresponding RGB image
      # <-- change to actual RGB image path
    rgb_image = cv2.imread(rgb_path)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Convert depth values to meters
    depth_meters = metric_depth.astype(np.float32)

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


