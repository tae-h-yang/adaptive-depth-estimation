import cv2
import torch
import numpy as np

from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder = 'vitl' # or 'vits', 'vitb'
dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 20 # 20 for indoor model, 80 for outdoor model

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
model.to(device)  # Move model to MPS
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
model.eval()

raw_img = cv2.imread('datasets/TUM_RGBD/rgbd_dataset_freiburg1_desk/rgb/1305031452.791720.png')
depth = model.infer_image(raw_img) # HxW depth map in meters in numpy

depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)  # Scale depth values to [0,255]
depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)  # Apply color map
# Display depth map
cv2.imshow("Depth Map", depth_colored)
cv2.waitKey(0)  # Wait for a key press to close window
cv2.destroyAllWindows()