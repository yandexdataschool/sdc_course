import numpy as np


def verify_transform_matrix(transform_matrix: np.ndarray):
    assert isinstance(transform_matrix, np.ndarray)
    assert transform_matrix.shape == (4, 4)
    assert transform_matrix.dtype == np.float64
    assert np.all(transform_matrix[3, :] == np.array([0., 0., 0., 1.], dtype=np.float64))


def compose_transforms(*transforms):
    """
    Takes sequence of transforms [T_1, T_2, ..., T_N] and returns transform
    T = T_N * ... * T_2 * T_1
    """
    composed_transform = np.eye(4, dtype=np.float64)
    for transform in transforms:
        composed_transform = np.dot(transform, composed_transform)
    return composed_transform
