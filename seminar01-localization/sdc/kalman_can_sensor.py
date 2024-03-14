import numpy as np
from .kalman_sensor_base import KalmanSensorBase


class KalmanCanSensor(KalmanSensorBase):
    def __init__(self, *args, **kwargs):
        super(KalmanCanSensor, self).__init__(*args, **kwargs)

    def __str__(self):
        return 'KalmanCAN'

    @property
    def observation_size(self):
        return 1

    def get_observation_matrix(self):
        observation_matrix = np.zeros(
            (self.observation_size, self.state_size), dtype=np.float64)
        observation_matrix[0, self._car_model.VEL_INDEX] = 1
        return observation_matrix


if __name__ != '__main__':
    sensor = KalmanCanSensor(noise_variances=[5])
    assert sensor.observation_size == 1
    assert np.all(sensor.get_noise_covariance() == np.diag([5]))
