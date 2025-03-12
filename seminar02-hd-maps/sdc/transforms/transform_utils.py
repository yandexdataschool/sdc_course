import numpy as np


def compose_transforms(*transforms):
    """
    Takes sequence of transforms [T_1, T_2, ..., T_N] and returns transform
    T = T_N * ... * T_2 * T_1
    """
    composed_transform = np.eye(4, dtype=np.float64)
    for transform in transforms:
        composed_transform = np.dot(transform, composed_transform)
    return composed_transform
