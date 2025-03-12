import typing as T
import numpy as np
from scipy.spatial.transform import Rotation
from .rotation_rpy import RotationRPY


def convert_rpy_to_rotation_matrix(rotation_rpy: RotationRPY) -> np.ndarray:
    rotation_matrix = Rotation.from_euler(
        'xyz',
        [rotation_rpy.roll,
         rotation_rpy.pitch,
         rotation_rpy.yaw]).as_matrix()
    assert rotation_matrix.shape == (3, 3) and rotation_matrix.dtype == np.float64
    return rotation_matrix


def convert_xyz_and_rpy_to_transform_matrix(
        position_xyz: np.ndarray,
        rotation_rpy: RotationRPY) -> np.ndarray:
    assert isinstance(position_xyz, np.ndarray)
    assert position_xyz.shape == (3,) and position_xyz.dtype == np.float64
    assert isinstance(rotation_rpy, RotationRPY)

    transform_matrix = np.eye(4, dtype=np.float64)
    transform_matrix[:3, :3] = convert_rpy_to_rotation_matrix(rotation_rpy)
    transform_matrix[:3, 3] = position_xyz
    return transform_matrix


def verify_transform_matrix(transform_matrix):
    assert isinstance(transform_matrix, np.ndarray)
    assert transform_matrix.shape == (4, 4)
    assert transform_matrix.dtype == np.float64
    assert np.all(transform_matrix[3, :] == np.array([0., 0., 0., 1.], dtype=np.float64))


def convert_rotation_matrix_to_quaternion(rotation_matrix: np.ndarray):
    assert rotation_matrix.shape == (3, 3)
    assert rotation_matrix.dtype == np.float64
    return Rotation.from_matrix(rotation_matrix).as_quat()


def convert_quaternion_to_rotation_matrix(quaternion: T.Union[T.List[float], np.ndarray]):
    quaternion = np.array(quaternion, copy=False)
    assert quaternion.shape == (4,)
    assert quaternion.dtype == np.float64
    return Rotation.from_quat(quaternion).as_matrix()
