import numpy as np
from sdc.kalman.sensors.base import KalmanSensorBase


class KalmanImuSensor(KalmanSensorBase):
    """Калмановский IMU-датчик"""
    def __str__(self):
        return "KalmanIMU"

    @property
    def observation_size(self):
        return 1

    def get_observation_matrix(self):
        """Калмановская матрица наблюдений C"""
        observation_matrix = np.zeros((self.observation_size, self.state_size), dtype=np.float64)
        observation_matrix[0, self._car_model.ANGULAR_VELOCITY_IDX] = 1
        return observation_matrix
