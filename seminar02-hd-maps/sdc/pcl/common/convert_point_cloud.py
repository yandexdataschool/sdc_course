import numpy as np
import open3d as o3d


def convert_point_cloud_xyz_to_o3d(point_cloud_xyz: np.ndarray) -> o3d.geometry.PointCloud:
    o3d.geometry.PointCloud()
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud_xyz)
    return point_cloud_o3d


def convert_point_cloud_o3d_to_xyz(point_cloud_o3d: o3d.geometry.PointCloud) -> np.ndarray:
    return np.array(point_cloud_o3d.points)
