import numpy as np
from sdc.kalman.sensors.base import KalmanSensorBase


class KalmanGnssSensor(KalmanSensorBase):
    """Калмановский эквивалент GNSS-датчика"""
    def __str__(self):
        return 'KalmanGNSS'

    @property
    def observation_size(self):
        return 2

    def get_observation_matrix(self):
        observation_matrix = np.zeros((self.observation_size, self.state_size), dtype=np.float64)
        observation_matrix[0, self._car_model.POSITION_X_IDX] = 1
        observation_matrix[1, self._car_model.POSITION_Y_IDX] = 1
        return observation_matrix


if __name__ != '__main__':
    sensor = KalmanGnssSensor(noise_variances=[5, 5])
    assert sensor.observation_size == 2
    assert np.all(sensor.get_noise_covariance() == np.diag([5, 5]))
