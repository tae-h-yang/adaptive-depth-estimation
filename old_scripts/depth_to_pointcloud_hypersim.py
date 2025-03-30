import argparse
import cv2
import glob
import numpy as np
import os
from PIL import Image
import torch

from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

def save_ply(filename, points, colors):
    """
    Save point cloud to a .ply file in ASCII format (compatible with MeshLab).

    Args:
        filename (str): Path to save the .ply file.
        points (numpy.ndarray): Nx3 array of XYZ coordinates.
        colors (numpy.ndarray): Nx3 array of RGB values (0-255).
    """
    assert points.shape[0] == colors.shape[0], "Points and colors must have the same number of rows!"

    num_points = points.shape[0]

    # Create the correct PLY header
    ply_header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

    # Combine XYZ and RGB data into a single array
    ply_data = np.hstack((points, colors)).astype(np.float32)  # Ensure float format for MeshLab compatibility

    # Save the file correctly
    with open(filename, 'w') as f:
        f.write(ply_header)
        np.savetxt(f, ply_data, fmt="%.4f %.4f %.4f %d %d %d")  # Ensure proper float & int formatting

    print(f"Saved {num_points} points to {filename}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate depth maps and point clouds from images.')
    parser.add_argument('--encoder', default='vitl', type=str, choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='Model encoder to use.')
    parser.add_argument('--load-from', default='', type=str, required=True,
                        help='Path to the pre-trained model weights.')
    parser.add_argument('--max-depth', default=20, type=float,
                        help='Maximum depth value for the depth map.')
    parser.add_argument('--img-path', type=str, required=True,
                        help='Path to the input image or directory containing images.')
    parser.add_argument('--outdir', type=str, default='./vis_pointcloud',
                        help='Directory to save the output point clouds.')

    args = parser.parse_args()

    # Force CPU on macOS M1/M2 to avoid compatibility issues
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Model configuration based on the chosen encoder
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    print("Initializing model...")

    # Initialize the DepthAnythingV2 model
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # Get the list of image files to process
    if os.path.isfile(args.img_path):
        filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)

    if not filenames:
        print("Error: No valid images found in the specified path!")
        return

    # Extract image dimensions from the first image
    first_image = Image.open(filenames[0])
    image_width, image_height = first_image.size

    print(f"Detected Image Size: {image_width}x{image_height}")

    # Given M_proj values from dataset (extracted earlier)
    M_proj = np.array([
        [1.7320507492870254, 0.0, 0.0, 0.0],
        [0.0, 2.309400999049367, 0.0, 0.0],
        [0.0, 0.0, -1.002002002002002, -2.002002002002002],
        [0.0, 0.0, -1.0, 0.0]
    ])

    # Compute focal lengths and principal points from M_proj
    f_x = M_proj[0, 0] * (image_width / 2)  # Scale focal length to pixel space
    f_y = M_proj[1, 1] * (image_height / 2)  # Scale focal length to pixel space
    c_x = (image_width / 2) * (1 + M_proj[0, 2])  # Principal point (assumed centered)
    c_y = (image_height / 2) * (1 + M_proj[1, 2])

    print(f"Computed Camera Intrinsics: f_x={f_x}, f_y={f_y}, c_x={c_x}, c_y={c_y}")

    # Create the output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)

    # Process each image file
    for k, filename in enumerate(filenames):
        print(f'Processing {k+1}/{len(filenames)}: {filename}')

        # Load the image
        color_image = Image.open(filename).convert('RGB')
        width, height = color_image.size

        # Read the image using OpenCV
        image = cv2.imread(filename)
        pred = depth_anything.infer_image(image, height)

        # Resize depth prediction to match the original image size
        resized_pred = Image.fromarray(pred).resize((width, height), Image.NEAREST)

        # Generate mesh grid and calculate point cloud coordinates
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = (x - c_x) / f_x
        y = (y - c_y) / f_y
        z = np.array(resized_pred)
        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)

        # Normalize colors to 0-255
        colors = np.array(color_image).reshape(-1, 3)

        # Save point cloud as .ply file
        output_ply = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + ".ply")
        save_ply(output_ply, points, colors)

if __name__ == '__main__':
    main()
