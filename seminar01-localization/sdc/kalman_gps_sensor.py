# -*- coding: utf-8 -*-
import numpy as np
from .kalman_sensor_base import KalmanSensorBase


class KalmanGpsSensor(KalmanSensorBase):
    """Калмановский эквивалент GPS-датчика."""
    def __init__(self, *args, **kwargs):
        super(KalmanGpsSensor, self).__init__(*args, **kwargs)

    def __str__(self):
        return 'KalmanGPS'

    @property
    def observation_size(self):
        return 2

    def get_observation_matrix(self):
        observation_matrix = np.zeros((self.observation_size, self.state_size), dtype=np.float64)
        observation_matrix[0, self._car_model.POS_X_INDEX] = 1
        observation_matrix[1, self._car_model.POS_Y_INDEX] = 1
        return observation_matrix


if __name__ != '__main__':
    sensor = KalmanGpsSensor(noise_variances=[5, 5])
    assert sensor.observation_size == 2
    assert np.all(sensor.get_noise_covariance() == np.diag([5, 5]))
