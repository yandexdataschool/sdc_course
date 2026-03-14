import numpy as np
from sdc.msgs import ImuMessage
from .base import SensorBase


class ImuSensor(SensorBase):
    """IMU sensor. Measures vehicle angular velocity"""
    def __str__(self):
        return 'IMU'

    @property
    def observation_size(self):
        return 1

    def _observe_clear(self):
        return np.array([self._robot.state.angular_velocity])

    def _generate_message(self) -> ImuMessage:
        angular_velocity = self.observe()[0]
        return ImuMessage(self.time, angular_velocity)
