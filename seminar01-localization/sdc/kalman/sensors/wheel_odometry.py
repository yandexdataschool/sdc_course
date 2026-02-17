import numpy as np
from sdc.kalman.sensors.base import KalmanSensorBase


class KalmanWheelOdometrySensor(KalmanSensorBase):
    def __str__(self):
        return "KalmanWO"

    @property
    def observation_size(self):
        return 1

    def get_observation_matrix(self):
        observation_matrix = np.zeros(
            (self.observation_size, self.state_size), dtype=np.float64)
        observation_matrix[0, self._robot_model.LINEAR_VELOCITY_IDX] = 1
        return observation_matrix
