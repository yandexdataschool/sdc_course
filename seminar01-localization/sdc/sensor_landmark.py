# -*- coding: utf-8 -*-
import numpy as np
from .car_sensor_base import CarSensorBase


def get_global_to_local_tranform_matrix(x, y, yaw):
    # Найдем матрицу перехода из системы координат машины в глобальную систему координат
    R = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ], dtype=np.float64)
    T_local2global = np.eye(3, dtype=np.float64)
    T_local2global[:2, :2] = R
    T_local2global[0, 2] = x
    T_local2global[1, 2] = y
    # В действительности нам нужна матрица перехода из глобальной системы координат в локальную
    T_global2local = np.linalg.inv(T_local2global)
    return T_global2local


def get_landmark_position_in_local_frame(x, y, yaw, landmark_x, landmark_y):
    """Позиция маяка (landmark_x, landmark_y), как и положение робота (x, y, yaw) заданы
    в глобальной системе координат. Функция возвращает позицию маяка в локальной системе координат
    робота.
    :param x: x-координата робота в глобальной системе координат
    :param y: y-координата робота в глобальной системе координат
    :param yaw: Угол поворота робота относительно оси OX глобальной системы координат
    :param landmark_x: x-координата маяка в глобальной системе координат
    :param landmark_y: y-координата маяка в глобальной системе координат
    """
    T_global2local = get_global_to_local_tranform_matrix(x=x, y=y, yaw=yaw)
    # Положение наблюдаемого объекта в глобальной системе координат
    L_global = np.array([landmark_x, landmark_y, 1], dtype=np.float64)
    # Положение наблюдаемого объекта в локальной системе координат
    L_local = np.dot(T_global2local, L_global)
    return L_local[:2]


def get_landmarks_position_in_local_frame(x, y, yaw, landmarks_xy):
    T_global2local = get_global_to_local_tranform_matrix(x=x, y=y, yaw=yaw)
    return np.dot(landmarks_xy, T_global2local[:2, :2].T) + T_global2local[:2, 0][None, :]


class LandmarkSensor(CarSensorBase):
    def __init__(self, x, y, *args, **kwargs):
        """
        :param x: x-координата наблюдаемого объекта в глобальной системе координат
        :param y: y-координата наблюдаемого объекта в глобальной системе координат
        """
        super(LandmarkSensor, self).__init__(*args, **kwargs)
        self._x = x
        self._y = y

    def __str__(self):
        return 'Landmark'

    @property
    def observation_size(self):
        return 2

    def _observe_clear(self):
        """Возвращает истинное положение объекта в системе координат машины (локальной системе
        координат)"""
        return get_landmark_position_in_local_frame(
            x=self._car._position_x,
            y=self._car._position_y,
            yaw=self._car._yaw,
            landmark_x=self._x,
            landmark_y=self._y)


class LandmarksSensor(CarSensorBase):
    def __init__(self, landmarks_global_positions, *args, **kwargs):
        self._landmarks_global_positions = np.array(landmarks_global_positions)
        assert self._landmarks_global_positions.shape[1] == 2
        self._landmarks_number = self._landmarks_global_positions.shape[0]

    @property
    def observation_size(self):
        return 2 * self._landmarks_number

    def _observe_clear(self):
        return get_landmarks_position_in_local_frame(
            x=self._car._position_x,
            y=self._car._position_y,
            yaw=self._car._yaw,
            landmarks_xy=self._landmarks_global_positions)


if __name__ != '__main__':
    sensor = LandmarkSensor(x=5, y=5, noise_variances=[2, 2])
    assert sensor.observation_size == 2
    assert np.all(sensor.get_noise_covariance() == np.diag([2, 2]))
