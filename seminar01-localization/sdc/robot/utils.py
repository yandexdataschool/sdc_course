import numpy as np
from sdc.core.transform import build_2d_transform_matrix
from .robot_base import RobotBase


def get_robot_global_pose(robot: RobotBase) -> np.ndarray:
    return build_2d_transform_matrix(
        position_x=robot.state.position_x,
        position_y=robot.state.position_y,
        yaw=robot.state.yaw,
    )
