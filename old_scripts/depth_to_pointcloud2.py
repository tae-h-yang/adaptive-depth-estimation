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
    parser.add_argument('--focal-length-x', default=470.4, type=float,
                        help='Focal length along the x-axis.')
    parser.add_argument('--focal-length-y', default=470.4, type=float,
                        help='Focal length along the y-axis.')

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

        # depth_normalized = cv2.normalize(pred, None, 0, 255, cv2.NORM_MINMAX)  # Scale depth values to [0,255]
        # depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)  # Apply color map
        # # Display depth map
        # cv2.imshow("Depth Map", depth_colored)
        # cv2.waitKey(0)  # Wait for a key press to close window
        # cv2.destroyAllWindows()

        # Resize depth prediction to match the original image size
        resized_pred = Image.fromarray(pred).resize((width, height), Image.NEAREST)

        # Generate mesh grid and calculate point cloud coordinates
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = (x - width / 2) / args.focal_length_x
        y = (y - height / 2) / args.focal_length_y
        z = np.array(resized_pred)
        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)

        # Normalize colors to 0-255
        colors = np.array(color_image).reshape(-1, 3)

        # Save point cloud as .ply file
        output_ply = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + ".ply")
        save_ply(output_ply, points, colors)

if __name__ == '__main__':
    main()
