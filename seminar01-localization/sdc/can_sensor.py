# -*- coding: utf-8 -*-
import numpy as np
from .car_sensor_base import CarSensorBase


class CanSensor(CarSensorBase):
    """Sensor that reports linear velocity of the car."""
    def __init__(self, *args, **kwargs):
        super(CanSensor, self).__init__(*args, **kwargs)

    def __str__(self):
        return 'CAN'

    @property
    def observation_size(self):
        return 1

    def _observe_clear(self):
        return np.array([self._car._velocity])


if __name__ != '__main__':
    sensor = CanSensor(noise_variances=[15])
    assert sensor.observation_size == 1
    assert np.all(sensor.get_noise_covariance() == np.diag([15]))
