import abc
import numpy as np
from sdc.kalman.filter import kalman_process_observation


class KalmanSensorBase(abc.ABC):
    """
    Модель наблюдений в модели калмановской локализации.
    """
    def __init__(self, sensor_id: str, noise_variances: np.ndarray):
        """
        :param noise_covariance: Ожидаемые значения дисперсии наблюдений (уровень шума).
        """
        self._sensor_id = sensor_id
        self._noise_variances = np.array(noise_variances)
        assert self._noise_variances.shape == (self.observation_size,)

    def _mount(self, robot_model):
        """This method is called when sensor model is added to robot model"""
        self._robot_model = robot_model

    @property
    def id(self) -> str:
        return self._sensor_id

    @property
    def state_size(self):
        return self._robot_model.state_size

    @property
    @abc.abstractmethod
    def observation_size(self) -> int:
        """Возвращает размер наблюдения"""
        ...

    @abc.abstractmethod
    def get_observation_matrix(self):
        """Марица наблюдений С для фильтра Калмана"""
        ...

    def get_noise_covariance(self):
        """Диагональная матрица ковариации шума для фильтра Калмана"""
        return np.diag(self._noise_variances)

    def process_observation(self, observation):
        C = self.get_observation_matrix()
        Q = self.get_noise_covariance()
        mu = self._robot_model.state
        S = self._robot_model.covariance_matrix
        new_mu, new_S = kalman_process_observation(mu, S, observation, C, Q)
        self._robot_model.state = new_mu
        self._robot_model.covariance_matrix = new_S
