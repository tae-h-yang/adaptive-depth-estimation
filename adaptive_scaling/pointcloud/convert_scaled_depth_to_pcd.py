import cv2
import numpy as np
import open3d as o3d

# Load the first RGB image (for colors)
rgb_path = "datasets/sync/office_0026/rgb_00075.jpg"  # Ensure the correct path
rgb = cv2.imread(rgb_path)
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)  # Convert to RGB format

# Load rescaled depth maps
depth_pred_true = np.load("adaptive_scaling/pointcloud/rescaled_depth_gt.npy")  # Ground truth scale
depth_pred_pred = np.load("adaptive_scaling/pointcloud/rescaled_depth_predicted.npy")  # Predicted scale

# Intrinsic Parameters
fx_d = 582.62448167737955
fy_d = 582.69103270988637
cx_d = 313.04475870804731
cy_d = 238.44389626620386

height, width = depth_pred_true.shape

def generate_pointcloud(depth_rescaled, filename):
    """Generate a point cloud and save it as a PLY file."""
    points = []
    colors = []

    for v in range(height):
        for u in range(width):
            Z = depth_rescaled[v, u]
            if Z == 0 or Z > 10:  # Ignore invalid depth
                continue

            X = (u - cx_d) * Z / fx_d
            Y = (v - cy_d) * Z / fy_d

            # Get RGB color
            color = rgb[v, u] / 255.0  # Normalize to [0,1]

            points.append([X, Y, Z])
            colors.append(color)

    # Convert to Open3D format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    # Save as PLY file
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Point cloud saved as {filename}")

# Generate and save point clouds
generate_pointcloud(depth_pred_true, "adaptive_scaling/pointcloud/rescaled_depth_gt.ply")
generate_pointcloud(depth_pred_pred, "adaptive_scaling/pointcloud/rescaled_depth_predicted.ply")
