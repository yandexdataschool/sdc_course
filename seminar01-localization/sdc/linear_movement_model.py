import numpy as np
from .timestamp import Timestamp
from .movement_model_base import MovementModelBase


class LinearMovementModel(MovementModelBase):
    """Продвигает автомобиль вперед с его текущей скоростью"""
    def __init__(self, *args, **kwargs):
        super(LinearMovementModel, self).__init__(*args, **kwargs)

    def _move(self,  dt):
        assert isinstance(dt, Timestamp)
        self._car._state = self.move_state(self._car._state, dt)
        self._car._time = self._car._time + dt

    def move_state(self, state, dt):
        assert isinstance(dt, Timestamp)
        car = self._car
        state_size = self._car._state_size
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
