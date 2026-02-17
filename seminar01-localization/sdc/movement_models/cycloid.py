import numpy as np
from sdc.core.timestamp import Timestamp
from sdc.movement_models.base import MovementModelBase


class CycloidMovementModel(MovementModelBase):
    """Реализует движение автомобиля по циклоиде"""
    def __init__(self, x_vel=0, y_vel=0, angular_velocity=0, *args, **kwargs):
        """
        :param x_vel: Скорость движения центра вращения вдоль оси X
        :param y_vel: Скорость движения центра вращения вдоль оси Y
        :param angular_velocity: Угловая скорость (рад/с) при движении по циклоиде
        """
        super().__init__(*args, **kwargs)
        self.x_vel = x_vel
        self.y_vel = y_vel
        self.angular_velocity = angular_velocity

    def _move_by_impl(self, dt: Timestamp):
        assert isinstance(dt, Timestamp)
        assert self._time == self._robot.time

        dt_sec = dt.to_seconds()

        x = self._robot.state.position_x
        y = self._robot.state.position_y
        vel = self._robot.state.linear_velocity
        yaw = self._robot.state.yaw

        vel_x = vel * np.cos(yaw)
        vel_y = vel * np.sin(yaw)

        new_x = x + vel_x * dt_sec
        new_y = y + vel_y * dt_sec
        new_vel_x = vel_x - self.angular_velocity * (vel_y - self.y_vel) * dt_sec
        new_vel_y = vel_y + self.angular_velocity * (vel_x - self.x_vel) * dt_sec

        # Продвигаем время, выставляем новое состояние
        self._robot.state.position_x = new_x
        self._robot.state.position_y = new_y
        self._robot.state.linear_velocity = np.sqrt(new_vel_x**2 + new_vel_y**2)
        self._robot.state.yaw = np.arctan2(new_vel_y, new_vel_x)

        self._time += dt
