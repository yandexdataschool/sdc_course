import numpy as np
from sdc.sensors.base import SensorBase


class ImuSensor(SensorBase):
    """IMU sensor. Measures vehicle angular velocity"""
    def __str__(self):
        return 'IMU'

    @property
    def observation_size(self):
        return 1

    def _observe_clear(self):
        return np.array([self._car._angular_velocity])


if __name__ != '__main__':
    sensor = ImuSensor(noise_variances=[1])
    assert sensor.observation_size == 1
    assert np.all(sensor.get_noise_covariance() == np.diag([1]))
