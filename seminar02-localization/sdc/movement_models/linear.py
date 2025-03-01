import numpy as np
from sdc.timestamp import Timestamp
from sdc.movement_models.base import MovementModelBase


class LinearMovementModel(MovementModelBase):
    """Moves vehicle forward  with its current angular and linear velocities"""
    def _move(self, dt: Timestamp):
        assert isinstance(dt, Timestamp)
        self._car._state = self.move_state(self._car._state, dt)
        self._car._time = self._car._time + dt

    def move_state(self, state, dt):
        assert isinstance(dt, Timestamp)
        car = self._car
        state_size = self._car._state_size
        assert state.shape[0] == state_size
        dt_sec = dt.to_seconds()
        x = state[car.POSITION_X_IDX]
        y = state[car.POSITION_Y_IDX]
        yaw = state[car.YAW_IDX]
        linear_velocity = state[car.LINEAR_VELOCITY_IDX]
        angular_velocity = state[car.ANGULAR_VELOCITY_IDX]
        new_state = np.zeros_like(state)
        new_state[car.POSITION_X_IDX] = x + linear_velocity * np.cos(yaw) * dt_sec
        new_state[car.POSITION_Y_IDX] = y + linear_velocity * np.sin(yaw) * dt_sec
        new_state[car.YAW_IDX] = yaw + angular_velocity * dt_sec
        new_state[car.LINEAR_VELOCITY_IDX] = linear_velocity
        new_state[car.ANGULAR_VELOCITY_IDX] = angular_velocity
        return new_state
