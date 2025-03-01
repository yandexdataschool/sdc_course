import numpy as np
from sdc.sensors.base import SensorBase


class GnssSensor(SensorBase):
    """GNSS sensor. Measures global vehicle position"""
    def __str__(self):
        return 'GNSS'

    @property
    def observation_size(self):
        return 2

    def _observe_clear(self):
        return np.array([self._car._position_x, self._car._position_y])


if __name__ != '__main__':
    sensor = GnssSensor(noise_variances=[15, 15])
    assert sensor.observation_size == 2
    assert np.all(sensor.get_noise_covariance() == np.diag([15, 15]))
