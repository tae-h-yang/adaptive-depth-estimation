import cv2
import torch
from timeit import default_timer as timer
from util.utils import inspect_depth
from depth_anything_v2.dpt import DepthAnythingV2

class RelativeDepthEstimator(DepthAnythingV2):
    def __init__(self, encoder='vitl'):
        """
        Initialize the Depth Anything v2 model.

        Parameters:
        - encoder: 'vits', 'vitb', or 'vitl' (determines model size)
        """
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

        # Model configurations
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        if encoder not in model_configs:
            raise ValueError(f"Invalid encoder '{encoder}'. Choose from: 'vits', 'vitb', 'vitl'.")

        self.encoder = encoder
        self.model_path = f"checkpoints/depth_anything_v2_{encoder}.pth"

        # Initialize parent class (DepthAnythingV2) with chosen parameters
        super().__init__(**{**model_configs[encoder]})

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
    relative_depth_estimator = RelativeDepthEstimator(encoder='vits')
    start = timer()
    relative_depth = relative_depth_estimator.infer('datasets/TUM_RGBD/rgbd_dataset_freiburg1_desk/rgb/1305031453.359684.png')
    end = timer()
    print(f"Inference time: {end - start:.6f} seconds")

    # Pass non metric and inverse relative depth values (D_metric = 1/(a * D_pred + b))
    inspect_depth(relative_depth)
    # inspect_depth(1/relative_depth)
