import argparse
import numpy as np
import h5py  # Import HDF5 reader
import open3d as o3d
from PIL import Image

def save_ply(filename, points, colors):
    """
    Save point cloud to a .ply file.

    Args:
        filename (str): Path to save the .ply file.
        points (numpy.ndarray): Nx3 array of XYZ coordinates.
        colors (numpy.ndarray): Nx3 array of RGB values (0-255).
    """
    assert points.shape[0] == colors.shape[0], "Points and colors must have the same number of rows!"

    num_points = points.shape[0]
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

    # Save the point cloud
    ply_data = np.hstack((points, colors)).astype(np.float32)
    with open(filename, 'w') as f:
        f.write(ply_header)
        np.savetxt(f, ply_data, fmt="%.4f %.4f %.4f %d %d %d")

    print(f"Saved {num_points} points to {filename}")

def depth_to_point_cloud(depth_path, color_path, output_path):
    """
    Convert an HDF5 depth image to a 3D point cloud.

    Args:
        depth_path (str): Path to the HDF5 depth image.
        color_path (str): Path to the corresponding RGB image.
        output_path (str): Path to save the output point cloud.
    """
    # Load the depth image from HDF5
    with h5py.File(depth_path, 'r') as f:
        depth = np.array(f['dataset'])  # Adjust this key if needed
        print(f"Loaded depth image shape: {depth.shape}")

    # Load the color image
    color_image = Image.open(color_path).convert('RGB')
    color_image = np.array(color_image)

    # Get image dimensions
    height, width = depth.shape
    print(f"Detected Image Size: {width}x{height}")

    # Given M_proj values from dataset
    M_proj = np.array([
        [1.7320507492870254, 0.0, 0.0, 0.0],
        [0.0, 2.309400999049367, 0.0, 0.0],
        [0.0, 0.0, -1.002002002002002, -2.002002002002002],
        [0.0, 0.0, -1.0, 0.0]
    ])

    # Compute focal lengths and principal points from M_proj
    f_x = M_proj[0, 0] * (width / 2)  # Scale focal length to pixel space
    f_y = M_proj[1, 1] * (height / 2)  # Scale focal length to pixel space
    c_x = (width / 2) * (1 + M_proj[0, 2])  # Principal point (assumed centered)
    c_y = (height / 2) * (1 + M_proj[1, 2])

    print(f"Computed Camera Intrinsics: f_x={f_x}, f_y={f_y}, c_x={c_x}, c_y={c_y}")

    # Generate mesh grid
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Convert pixel coordinates to camera coordinates
    x = (x - c_x) / f_x
    y = (y - c_y) / f_y
    z = depth  # Depth is already in meters

    # Compute 3D coordinates
    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)

    # Reshape colors
    colors = color_image.reshape(-1, 3)

    # Filter out invalid depth points (zero depth is ignored)
    valid_mask = (z > 0).reshape(-1)
    points = points[valid_mask]
    colors = colors[valid_mask]

    # Save point cloud
    save_ply(output_path, points, colors)

def main():
    parser = argparse.ArgumentParser(description="Convert an HDF5 depth image to a 3D point cloud.")
    parser.add_argument('--depth', type=str, required=True, help='Path to the HDF5 depth image.')
    parser.add_argument('--color', type=str, required=True, help='Path to the corresponding RGB image.')
    parser.add_argument('--out', type=str, required=True, help='Output PLY file path.')

    args = parser.parse_args()

    depth_to_point_cloud(args.depth, args.color, args.out)

if __name__ == '__main__':
    main()
