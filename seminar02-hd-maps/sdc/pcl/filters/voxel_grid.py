import numpy as np
from ..common.convert_point_cloud import (
    convert_point_cloud_o3d_to_xyz,
    convert_point_cloud_xyz_to_o3d,
)


def apply_voxel_grid(point_cloud_xyz, voxel_size: float) -> np.ndarray:
    point_cloud_o3d = convert_point_cloud_xyz_to_o3d(point_cloud_xyz)
    point_cloud_o3d = point_cloud_o3d.voxel_down_sample(voxel_size=voxel_size)
    return convert_point_cloud_o3d_to_xyz(point_cloud_o3d)
