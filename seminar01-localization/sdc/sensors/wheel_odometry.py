import numpy as np
from sdc.sensors.base import SensorBase


class WheelOdometrySensor(SensorBase):
    """Wheel odometry sensor. Measures linear velocity"""
    def __str__(self):
        return 'CAN'

    @property
    def observation_size(self):
        return 1

    def _observe_clear(self):
        return np.array([self._car._linear_velocity])


if __name__ != '__main__':
    sensor = WheelOdometrySensor(noise_variances=[15])
    assert sensor.observation_size == 1
    assert np.all(sensor.get_noise_covariance() == np.diag([15]))
