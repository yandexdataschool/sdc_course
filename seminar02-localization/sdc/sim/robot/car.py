import copy
import dataclasses
import numpy as np
import typing as T
from sdc.core.timestamp import Timestamp
from .robot_base import RobotStateBase, RobotBase


@dataclasses.dataclass
class CarParameters:
    length: float
    width: float


class CarState(RobotStateBase):
    POSITION_X_IDX = 0
    POSITION_Y_IDX = 1
    YAW_IDX = 2
    LINEAR_VELOCITY_IDX = 3
    ANGULAR_VELOCITY_IDX = 4

    def __init__(self):
        self._state = np.zeros(5, dtype=np.float64)

    def __getitem__(self, idx) -> float:
        return self._state[idx]

    def __setitem__(self, idx, value):
        self._state[idx] = value

    def reset(self):
        self._state = np.zeros_like(self._state)

    @property
    def size(self):
        return len(self._state)

    @property
    def array(self) -> np.ndarray:
        return self._state

    @array.setter
    def array(self, array: np.ndarray):
        assert len(array) == self.size
        self._state = np.array(array)

    @property
    def position_x(self):
        return self._state[self.POSITION_X_IDX]

    @position_x.setter
    def position_x(self, position_x):
        self._state[self.POSITION_X_IDX] = position_x

    @property
    def position_y(self):
        return self._state[self.POSITION_Y_IDX]

    @position_y.setter
    def position_y(self, position_y):
        self._state[self.POSITION_Y_IDX] = position_y

    @property
    def yaw(self):
        return self._state[self.YAW_IDX]

    @yaw.setter
    def yaw(self, yaw):
        self._state[self.YAW_IDX] = yaw

    @property
    def linear_velocity(self):
        return self._state[self.LINEAR_VELOCITY_IDX]

    @linear_velocity.setter
    def linear_velocity(self, linear_velocity):
        self._state[self.LINEAR_VELOCITY_IDX] = linear_velocity

    @property
    def linear_velocity_x(self):
        return self.linear_velocity * np.cos(self.yaw)

    @property
    def linear_velocity_y(self):
        return self.linear_velocity * np.sin(self.yaw)

    @property
    def angular_velocity(self):
        return self._state[self.ANGULAR_VELOCITY_IDX]

    @angular_velocity.setter
    def angular_velocity(self, angular_velocity):
        self._state[self.ANGULAR_VELOCITY_IDX] = angular_velocity


class Car(RobotBase):
    """Простая модель автомобиля в двухмерном мире.
    В качестве истинных переменных состояния выступают положение, скорости и ориентация относительно оси oX:
    (pos_x, pos_y, vel, yaw)
    Также автомобиль поддерживает значение текущего глобального времени.

    Истинное состояние автомобиля неизвестно внешнему наблюдателю, однако у автомобиля есть набор сенсоров,
    у которых можно спрашивать текущие значения скоростей и глобальных координат. Сенсоры выдают данные
    с некоторым шумом.
    """

    def __init__(self, params: CarParameters):
        super().__init__()
        self._params = params
        self._state = CarState()

        # States history
        self._positions_x = []
        self._positions_y = []
        self._yaws = []
        self._linear_velocities = []
        self._angular_velocities = []

    def set_initial_state(self, initial_state):
        self._initial_state = copy.deepcopy(initial_state)
        self._state = copy.deepcopy(initial_state)

    def __str__(self):
        return '{}(x={:.2f}[m], y={:.2f}[m], yaw={:.2f}[rad], lin_vel={:.2f}[m/s], '\
            'ang_vel={:.2f}[rad/s], t={})'.format(
                type(self).__name__,
                self.state.position_x,
                self.state.position_y,
                self.state.yaw,
                self.state.linear_velocity,
                self.state.angular_velocity,
                self.time)

    def _move_by_impl(self, dt: Timestamp):
        super()._move_by_impl(dt)

        # Храним историю состояний
        self._positions_x.append(self.state.position_x)
        self._positions_y.append(self.state.position_y)
        self._yaws.append(self.state.yaw)
        self._linear_velocities.append(self.state.linear_velocity)
        self._angular_velocities.append(self.state.angular_velocity)


def initialize_car_model(
    car: Car,
    initial_position: T.Tuple[float, float],
    initial_yaw: float,
    initial_linear_velocity: float,
    initial_angular_velocity: float,
):
    assert car.time is None

    car.state.position_x = initial_position[0]
    car.state.position_y = initial_position[1]
    car.state.yaw = initial_yaw
    car.state.linear_velocity = initial_linear_velocity
    car.state.angular_velocity = initial_angular_velocity

    assert car.state.position_x == initial_position[0]
    assert car.state.position_y == initial_position[1]
    assert car.state.yaw == initial_yaw
    assert car.state.linear_velocity == initial_linear_velocity
    assert car.state.angular_velocity == initial_angular_velocity
