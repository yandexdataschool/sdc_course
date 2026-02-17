import numpy as np
from sdc.core.timestamp import Timestamp
from sdc.movement_models.base import MovementModelBase


class LinearMovementModel(MovementModelBase):
    """Moves vehicle forward with its current angular and linear velocities"""
    def _move_by_impl(self, dt: Timestamp):
        assert isinstance(dt, Timestamp)
        assert self._time == self._robot.time

        dt_sec = dt.to_seconds()

        state = self._robot.state
        x = state[state.POSITION_X_IDX]
        y = state[state.POSITION_Y_IDX]
        yaw = state[state.YAW_IDX]
        linear_velocity = state[state.LINEAR_VELOCITY_IDX]
        angular_velocity = state[state.ANGULAR_VELOCITY_IDX]

        new_state_array = np.zeros_like(state.array)
        new_state_array[state.POSITION_X_IDX] = x + linear_velocity * np.cos(yaw) * dt_sec
        new_state_array[state.POSITION_Y_IDX] = y + linear_velocity * np.sin(yaw) * dt_sec
        new_state_array[state.YAW_IDX] = yaw + angular_velocity * dt_sec
        new_state_array[state.LINEAR_VELOCITY_IDX] = linear_velocity
        new_state_array[state.ANGULAR_VELOCITY_IDX] = angular_velocity
        self._robot.state.array = new_state_array

        self._time += dt
