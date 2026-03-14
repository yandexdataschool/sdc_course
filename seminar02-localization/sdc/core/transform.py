import numpy as np


def build_2d_transform_matrix(position_x, position_y, yaw) -> np.ndarray:
    return np.array([
        [np.cos(yaw), -np.sin(yaw), position_x],
        [np.sin(yaw), np.cos(yaw), position_y],
        [0, 0, 1]
    ])
