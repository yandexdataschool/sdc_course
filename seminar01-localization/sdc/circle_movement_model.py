# -*- coding: utf-8 -*-
import numpy as np
from .timestamp import Timestamp
from .movement_model_base import MovementModelBase


class CircleMovementModel(MovementModelBase):
    """Реализует движение автомобиля по циклоиде"""
    def __init__(self, *args, **kwargs):
        super(CircleMovementModel, self).__init__(*args, **kwargs)
        self._radius = None
        self._omega = None
        self._velocity = None

    def _initialize(self, car):
        super(CircleMovementModel, self)._initialize(car)
        # Определение парметров траектории (x_c, y_c, t_0) из начального сотояния робота
        # x(t) = x_c + r * cos(w(t - t_0))
        # y(t) = y_c + r * sin(w(t - t_0))
        t = car.time
        position_x = car._position_x
        position_y = car._position_y
        linear_velocity = car._linear_velocity
        yaw = car._yaw
        angular_velocity = car._angular_velocity
        assert angular_velocity != 0.

        # Линейная скорость при движении по окружности не зависит от времени
        self._angular_velocity = angular_velocity
        # Угловая скорость при движении по окружности не зависит от времени
        self._linear_velocity = linear_velocity

        self._center_x = position_x - linear_velocity * np.sin(yaw) / angular_velocity
        self._center_y = position_y + linear_velocity * np.cos(yaw) / angular_velocity
        phase = np.arctan2(position_y - self._center_y, position_x - self._center_x)
        self._t_0 = t.to_seconds() - phase / angular_velocity
        self._radius = abs(linear_velocity / angular_velocity)
        # Угол yaw опережает фазу фращения на pi / 2
        assert np.isclose((phase + np.pi / 2) % (2 * np.pi), yaw % (2 * np.pi))

    def _move(self, dt):
        assert isinstance(dt, Timestamp)
        assert self._car._linear_velocity == self._linear_velocity, 'Linear velocity must be constant'
        assert self._car._angular_velocity == self._angular_velocity, 'Angular velocity must be constant'

        car = self._car
        curr_t = car._time
        next_t = curr_t + dt

        phase = self._angular_velocity * (next_t.to_seconds() - self._t_0)
        next_yaw = phase + np.pi / 2.
        next_x = self._center_x + self._radius * np.cos(phase)
        next_y = self._center_y + self._radius * np.sin(phase)

        # Продвигаем время, выставляем новое состояние
        car.time = next_t
        car._position_x = next_x
        car._position_y = next_y
        car._yaw = next_yaw
