import os
import open3d as o3d
from .convert_point_cloud import convert_point_cloud_o3d_to_xyz


def read_point_cloud_o3d(point_cloud_path: str, format='ply'):
    assert os.path.isfile(point_cloud_path)
    point_cloud = o3d.io.read_point_cloud(point_cloud_path, format=format, print_progress=True)
    assert isinstance(point_cloud, o3d.geometry.PointCloud)
    return point_cloud


def read_point_cloud_xyz(point_cloud_path: str, format='ply'):
    point_cloud_o3d = read_point_cloud_o3d(point_cloud_path)
    return convert_point_cloud_o3d_to_xyz(point_cloud_o3d)
