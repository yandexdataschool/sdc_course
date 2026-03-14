import numpy as np
from sdc.msgs import GnssPositionMessage
from .base import SensorBase
from .utils import get_sensor_global_position


class GnssSensor(SensorBase):
    """GNSS sensor. Measures global vehicle position"""
    def __str__(self):
        return 'GNSS'

    @property
    def observation_size(self):
        return 2

    def _observe_clear(self) -> np.ndarray:
        return get_sensor_global_position(self)

    def _generate_message(self) -> GnssPositionMessage:
        observation_x, observation_y = self.observe()
        return GnssPositionMessage(self.time, observation_x, observation_y)
