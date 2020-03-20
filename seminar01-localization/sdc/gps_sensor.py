# -*- coding: utf-8 -*-
import numpy as np
from .car_sensor_base import CarSensorBase


class GpsSensor(CarSensorBase):
    """GPS-датчик. Измеряет глобальное положение автомобиля."""
    def __init__(self, *args, **kwargs):
        super(GpsSensor, self).__init__(*args, **kwargs)

    def __str__(self):
        return 'GPS'

    @property
    def observation_size(self):
        return 2

    def _observe_clear(self):
        return np.array([self._car._position_x, self._car._position_y])


if __name__ != '__main__':
    sensor = GpsSensor(noise_variances=[15, 15])
    assert sensor.observation_size == 2
    assert np.all(sensor.get_noise_covariance() == np.diag([15, 15]))
