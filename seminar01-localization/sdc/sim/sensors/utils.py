import numpy as np
from sdc.sim.sensors.base import SensorBase
from sdc.sim.robot.utils import get_robot_global_pose


def get_sensor_global_position(sensor: SensorBase) -> np.ndarray:
    assert sensor.mounted, f'Sensor "{sensor.id}" is not mounted'
    robot = sensor.robot
    T_body_sensor = robot.get_sensor_pose(sensor.id)
    T_world_body = get_robot_global_pose(robot)
    sensor_global_position = (T_world_body @ T_body_sensor)[:2, 2]
    return sensor_global_position
