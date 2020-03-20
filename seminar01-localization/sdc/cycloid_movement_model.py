# -*- coding: utf-8 -*-
import numpy as np
from .timestamp import Timestamp
from .movement_model_base import MovementModelBase


class CycloidMovementModel(MovementModelBase):
    """Реализует движение автомобиля по циклоиде"""
    def __init__(self, x_vel=0, y_vel=0, omega=0, *args, **kwargs):
        """
        :param x_vel: Скорость движения центра вращения вдоль оси X
        :param y_vel: Скорость двжиения центра вращения вдоль оси Y
        :param omega: Угловая скорость (рад/с) при движении по циклоиде
        """
        super(CycloidMovementModel, self).__init__(*args, **kwargs)
        self.x_vel = x_vel
        self.y_vel = y_vel
        self.omega = omega

    def _move(self, dt):
        assert isinstance(dt, Timestamp)
        car = self._car
        dt_sec = dt.to_seconds()

        x = car._position_x
        y = car._position_y
        vel = car._velocity
        yaw = car._yaw

        vel_x = vel * np.cos(yaw)
        vel_y = vel * np.sin(yaw)

        new_x = x + vel_x * dt_sec
        new_y = y + vel_y * dt_sec
        new_vel_x = vel_x - self.omega * (vel_y - self.y_vel) * dt_sec
        new_vel_y = vel_y + self.omega * (vel_x - self.x_vel) * dt_sec

        # Продвигаем время, выставляем новое состояние
        car.time += dt
        car._position_x = new_x
        car._position_y = new_y
        car._velocity = np.sqrt(new_vel_x**2 + new_vel_y**2)
        car._yaw = np.arctan2(new_vel_y, new_vel_x)
