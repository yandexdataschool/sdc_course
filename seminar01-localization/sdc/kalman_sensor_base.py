import abc
import numpy as np
from .kalman_filter import kalman_process_observation


class KalmanSensorBase(abc.ABC):
    """
    Модель наблюдений в модели калмановской локализации.
    """
    def __init__(self,  noise_variances=None):
        """
        :param noise_covariance: Ожидаемые значения дисперсии наблюдений (уровень шума).
        """
        if noise_variances is None:
            self._noise_variances = np.zeros(self.observation_size, dtype=np.float64)
        else:
            self._noise_variances = np.array(noise_variances)
            assert self._noise_variances.shape == (self.observation_size,)

    def _initialize(self, car_model):
        """Вызывается в момент добавления сенсора в машину"""
        self._car_model = car_model

    @property
    def state_size(self):
        return self._car_model._state_size

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
        mu = self._car_model.state
        S = self._car_model.covariance_matrix
        new_mu, new_S = kalman_process_observation(mu, S, observation, C, Q)
        self._car_model.state = new_mu
        self._car_model.covariance_matrix = new_S
