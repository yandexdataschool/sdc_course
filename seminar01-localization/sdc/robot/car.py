import numpy as np
from sdc.core.timestamp import Timestamp
from .robot_base import RobotBase


class Car(RobotBase):
    """Простая модель автомобиля в двухмерном мире.
    В качестве истинных переменных состояния выступают положение, скорости и ориентация относительно оси oX:
    (pos_x, pos_y, vel, yaw)
    Также автомобиль поддерживает значение текущего глобального времени.

    Истинное состояние автомобиля неизвестно внешнему наблюдателю, однако у автомобиля есть набор сенсоров,
    у которых можно спрашивать текущие значения скоростей и глобальных координат. Сенсоры выдают данные
    с некоторым шумом.
    """

    POSITION_X_IDX = 0
    POSITION_Y_IDX = 1
    YAW_IDX = 2
    LINEAR_VELOCITY_IDX = 3
    ANGULAR_VELOCITY_IDX = 4

    def __init__(
            self,
            initial_position=None,
            initial_yaw=None,
            initial_linear_velocity=None,
            initial_angular_velocity=None,
            movement_model=None):
        """
        :param initial_position: list, tuple, np.ndarray with two elements (shape = (2,))
        :param intial_yaw: float
        :param initial_linear_velocity: float
        :param movement_model: MovementModelBase or None. Represents the real movement trajectory
        """
        super().__init__()

        assert isinstance(initial_position, (list, tuple, np.ndarray))
        if initial_position is None:
            self.initial_position = np.zeros(2, dtype=np.float64)
        else:
            self.initial_position = np.array(initial_position, dtype=np.float64)
            assert self.initial_position.shape == (2,)

        if initial_yaw is None:
            self.initial_yaw = 0.
        else:
            self.initial_yaw = float(initial_yaw)

        if initial_linear_velocity is None:
            self.initial_linear_velocity = 0.
        else:
            self.initial_linear_velocity = float(initial_linear_velocity)

        if initial_angular_velocity is None:
            self.initial_angular_velocity = 0.
        else:
            self.initial_angular_velocity = float(initial_angular_velocity)

        # Инициализация состояния автомобиля
        self._state = np.zeros(5)
        self._position_x = self.initial_position[0]
        self._position_y = self.initial_position[1]
        self._yaw = self.initial_yaw
        self._linear_velocity = self.initial_linear_velocity
        self._angular_velocity = self.initial_angular_velocity

        # У автомобиля есть некоторая траектория, задаваемая моделью движения
        self.set_movement_model(movement_model)
        self._sensor_by_id = dict()

        # История состояний
        self._positions_x = []
        self._positions_y = []
        self._yaws = []
        self._linear_velocities = []
        self._linear_velocities_x = []
        self._linear_velocities_y = []
        self._angular_velocities = []

    def __str__(self):
        return '{}(x={:.2f}[m], y={:.2f}[m], yaw={:.2f}[rad], lin_vel={:.2f}[m/s], '\
            'ang_vel={:.2f}[rad/s], t={})'.format(
                type(self).__name__,
                self._position_x, self._position_y, self._yaw, self._linear_velocity,
                self._angular_velocity, self.time)

    def move(self, dt):
        assert isinstance(dt, Timestamp)
        self._movement_model._move(dt)
        # Храним историю состояний
        self._positions_x.append(self._position_x)
        self._positions_y.append(self._position_y)
        self._yaws.append(self._yaw)
        self._linear_velocities.append(self._linear_velocity)
        self._linear_velocities_x.append(self._linear_velocity_x)
        self._linear_velocities_y.append(self._linear_velocity_y)
        self._angular_velocities.append(self._angular_velocity)

    ######################################################################
    # Доступ к переменным состояния модели (на самом деле скрыты от нас) #
    ######################################################################
    @property
    def _state_size(self):
        return len(self._state)

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
