import numpy as np


def transform_point_cloud_xyz(
        cloud_xyz: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    assert isinstance(cloud_xyz, np.ndarray)
    assert cloud_xyz.ndim == 2
    assert cloud_xyz.shape[1] == 3

    assert isinstance(transform_matrix, np.ndarray)
    assert transform_matrix.shape == (4, 4)
    assert transform_matrix[3, 0] == 0
    assert transform_matrix[3, 1] == 0
    assert transform_matrix[3, 2] == 0
    assert transform_matrix[3, 3] == 1

    assert cloud_xyz.dtype == transform_matrix.dtype

    rotation_matrix = transform_matrix[:3, :3]
    translation_vector = transform_matrix[:3, 3]
    return cloud_xyz @ rotation_matrix.T + translation_vector[None, :]
