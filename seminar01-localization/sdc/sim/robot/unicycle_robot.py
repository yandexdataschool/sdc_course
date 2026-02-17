import dataclasses
import numpy as np
from .robot_base import RobotStateBase, RobotBase


@dataclasses.dataclass
class UnicycleRobotParameters:
    wheel_length: float
    wheel_width: float


class UnicycleRobotV1State(RobotStateBase):
    POSITION_X_IDX = 0
    POSITION_Y_IDX = 1
    YAW_IDX = 2

    def __init__(self):
        self._state = np.zeros(3, dtype=np.float64)

    @property
    def size(self) -> int:
        return len(self._state)

    @property
    def array(self) -> np.ndarray:
        return self._state

    @property
    def position_x(self):
        return self._state[self.POSITION_X_IDX]

    @property
    def position_y(self):
        return self._state[self.POSITION_Y_IDX]

    @property
    def yaw(self):
        return self._state[self.YAW_IDX]

    @position_x.setter
    def position_x(self, position_x):
        self._state[self.POSITION_X_IDX] = position_x

    @position_y.setter
    def position_y(self, position_y):
        self._state[self.POSITION_Y_IDX] = position_y

    @yaw.setter
    def yaw(self, yaw):
        self._state[self.YAW_IDX] = yaw


class UnicycleRobotV1(RobotBase):
    POSITION_X_IDX = 0
    POSITION_Y_IDX = 1
    YAW_IDX = 2

    def __init__(self, params: UnicycleRobotParameters):
        assert isinstance(params, UnicycleRobotParameters)
        super().__init__()
        self._params = params
        self._state = UnicycleRobotV1State()

    @property
    def params(self) -> UnicycleRobotParameters:
        return self._params
