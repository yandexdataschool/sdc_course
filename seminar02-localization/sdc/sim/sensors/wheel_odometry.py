import numpy as np
from sdc.msgs import WheelOdometryMessage
from .base import SensorBase


class WheelOdometrySensor(SensorBase):
    """Wheel odometry sensor. Measures linear velocity"""
    def __str__(self):
        return "WO"

    @property
    def observation_size(self):
        return 1

    def _observe_clear(self):
        return np.array([self._robot.state.linear_velocity])

    def _generate_message(self) -> WheelOdometryMessage:
        linear_velocity = self.observe()[0]
        return WheelOdometryMessage(self.time, linear_velocity)
