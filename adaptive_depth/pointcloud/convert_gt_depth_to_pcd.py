import numpy as np
import cv2
import open3d as o3d

# Load depth image
depth_path = "datasets/sync/office_0026/sync_depth_00075.png"
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

# Convert depth values to meters
depth_meters = depth / 1000.0  # Convert mm to meters

# Depth Intrinsic Parameters (NYU dataset)
fx_d = 582.62448167737955
fy_d = 582.69103270988637
cx_d = 313.04475870804731
cy_d = 238.44389626620386

# Extrinsic parameters: Rotation and Translation
R = np.array([
    [ 9.9997798940829263e-01,  5.0518419386157446e-03,  4.3011152014118693e-03],
    [-5.0359919480810989e-03,  9.9998051861143999e-01, -3.6879781309514218e-03],
    [-4.3196624923060242e-03,  3.6662365748484798e-03,  9.9998394948385538e-01]
])
R = np.linalg.inv(R)

t_x, t_y, t_z = 0.025031875059141302, 0.00066238747008330102, -0.00029342312935846411
T = np.array([t_x, t_y, t_z]).reshape(3, 1)

# Get depth image shape
height, width = depth.shape

# Generate 3D point cloud
points = []
colors = []

# Load corresponding RGB image
rgb_path = "datasets/sync/office_0026/rgb_00075.jpg"  # Ensure correct path
rgb_image = cv2.imread(rgb_path)
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

for v in range(height):
    for u in range(width):
        Z = depth_meters[v, u]
        if Z == 0 or Z > 10:  # Ignore invalid depth
            continue

        X = (u - cx_d) * Z / fx_d
        Y = (v - cy_d) * Z / fy_d

        # Convert to RGB frame
        point3D = np.array([[X], [Y], [Z]])
        transformed_point = R @ point3D + T
        X_rgb, Y_rgb, Z_rgb = transformed_point.flatten()

        # Get RGB color
        color = rgb_image[v, u] / 255.0  # Normalize to [0, 1]
        
        points.append([X_rgb, Y_rgb, Z_rgb])
        colors.append(color)

# Convert to Open3D format
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(points))
pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

# Save as PLY file
o3d.io.write_point_cloud("adaptive_depth/pointcloud/gt_pointcloud.ply", pcd)
print("Point cloud saved as output_pointcloud.ply")
