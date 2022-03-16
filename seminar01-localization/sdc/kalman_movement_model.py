# -*- coding: utf-8 -*-
import numpy as np
from .timestamp import Timestamp


class KalmanMovementModel(object):
    """Модель эволюции в калмановской локализации.
    Продвигает автомобиль вперед с его текущей скоростью"""
    def __init__(self, noise_covariance_density=None):
        """
        :type noise_covariance_density: np.ndarray или None
        :param noise_covariance_density: плотность матрицы ковариации.
            Элементами этой матрицы являются скорости нарастания ковариации между элементами в секунду.
            Если матрица плотности ковариции не задана, то полагается нулевой.
        """
        self._car_model = None
        self._noise_covariance_density = noise_covariance_density

    @property
    def state_size(self):
        return self._car_model._state_size

    def _initialize(self, car_model):
        """Вызывается при добавлении модели движения к автомобилю.
        Привязывает модель к конкретному автомобилю.
        """
        self._car_model = car_model
        state_size = car_model._state_size
        if self._noise_covariance_density is None:
            self._noise_covariance_density = np.zeros((state_size, state_size), dtype=np.float64)
        else:
            self._noise_covariance_density = np.array(self._noise_covariance_density, dtype=np.float64)
            assert self._noise_covariance_density.shape == (state_size, state_size)

    def get_next_state(self, dt):
        """Возвращает состояние в следующий момент времени."""
        assert isinstance(dt, Timestamp)
        car = self._car_model
        state_size = car.state_size
        state = car.state
        assert state.shape[0] == state_size
        dt_sec = dt.to_seconds()
        x = state[car.POS_X_INDEX]
        y = state[car.POS_Y_INDEX]
        yaw = state[car.YAW_INDEX]
        vel = state[car.VEL_INDEX]
        omega = state[car.OMEGA_INDEX]

        new_state = np.zeros_like(state)
        new_state[car.POS_X_INDEX] = x + vel * np.cos(yaw) * dt_sec
        new_state[car.POS_Y_INDEX] = y + vel * np.sin(yaw) * dt_sec
        new_state[car.YAW_INDEX] = yaw + omega * dt_sec
        new_state[car.VEL_INDEX] = vel
        new_state[car.OMEGA_INDEX] = omega
        return new_state

    def get_state_jacobian_matrix(self, dt):
        """Возвращает матрицу матрицу Якоби car.time. В случае линейной системе матрица Якоби представляет
        собой матрицу перехода A для текущего момента времени car.time."""
        assert isinstance(dt, Timestamp)
        car = self._car_model
        state_size = car._state_size
        state = car.state
        assert state.shape[0] == state_size

        dt_sec = dt.to_seconds()
        vel = state[car.VEL_INDEX]
        yaw = state[car.YAW_INDEX]
        J = np.eye(state_size, dtype=np.float64)
        J[car.POS_X_INDEX, car.VEL_INDEX] = np.cos(yaw) * dt_sec
        J[car.POS_Y_INDEX, car.VEL_INDEX] = np.sin(yaw) * dt_sec
        J[car.POS_X_INDEX, car.YAW_INDEX] = -vel * np.sin(yaw) * dt_sec
        J[car.POS_Y_INDEX, car.YAW_INDEX] = vel * np.cos(yaw) * dt_sec
        J[car.YAW_INDEX, car.OMEGA_INDEX] = dt_sec
        return J

    def get_noise_covariance(self, dt):
        """Возвращает матрицу ковариации шума для текущего момента времени car.time"""
        assert isinstance(dt, Timestamp)
        return self._noise_covariance_density * float(dt.to_seconds())

    def get_noise_covariance_density(self):
        """Возвращает матрицу плотности ковариации шума для текущего момента времени car.time"""
        return self._noise_covariance_density
