import typing as T
import numpy as np
from sdc.timestamp import Timestamp
from sdc.movement_models.base import MovementModelBase
from sdc.sensors.base import SensorBase
from sdc.sensors.gnss import GnssSensor
from sdc.sensors.imu import ImuSensor
from sdc.sensors.wheel_odometry import WheelOdometrySensor
from sdc.sensors.landmark import LandmarkSensor


class Car:
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
            self.initial_angular_velocity= 0.
        else:
            self.initial_angular_velocity = float(initial_angular_velocity)

        # Инициализация состояния автомобиля
        self._state = np.zeros(5)
        self._position_x = self.initial_position[0]
        self._position_y = self.initial_position[1]
        self._yaw = self.initial_yaw
        self._linear_velocity = self.initial_linear_velocity
        self._angular_velocity = self.initial_angular_velocity

        self._time = Timestamp()
        assert self._time.nsec == 0 and self._time.sec == 0

        # У автомобиля есть некоторая траектория, задаваемая моделью движения
        self.set_movement_model(movement_model)
        # У автомобиля есть некоторый набор сенсоров (датичков)
        self._sensors = []
        self._wo_sensor = None  # Одометрия
        self._gnss_sensor = None  # GPS
        self._imu_sensor = None  # IMU (гироскоп)
        self._landmark_sensors = []  # Сенсоры наблюдения за маяками

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

    def set_movement_model(
            self, movement_model: T.Optional[MovementModelBase]):
        if movement_model is not None:
            assert isinstance(movement_model, MovementModelBase)
            # Привязываем модель движения к автомобилю
            self._movement_model = movement_model
            # Привязываем автомобиль к модели движения
            movement_model._initialize(self)
        else:
            self._movement_model = None

    def add_sensor(self, sensor):
        assert isinstance(sensor, SensorBase)
        if isinstance(sensor, WheelOdometrySensor):
            self._wo_sensor = sensor
        elif isinstance(sensor, GnssSensor):
            self._gnss_sensor = sensor
        elif isinstance(sensor, ImuSensor):
            self._imu_sensor = sensor
        elif isinstance(sensor, LandmarkSensor):
            self._landmark_sensors.append(sensor)
        else:
            assert False, f'Unknown sensor type {type(sensor)}'
        self._sensors.append(sensor)
        sensor._initialize(self)

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
    #  Доступ к компонентам автомобиля - модели движения и сенсорам      #
    ######################################################################
    @property
    def movement_model(self):
        return self._movement_model

    @property
    def sensors(self):
        return self._sensors

    @property
    def wo_sensor(self):
        return self._wo_sensor

    @property
    def gnss_sensor(self):
        return self._gnss_sensor

    @property
    def imu_sensor(self):
        return self._imu_sensor

    @property
    def landmark_sensors(self):
        return self._landmark_sensors

    ######################################################################
    # Доступ к переменным состояния модели (на самом деле скрыты от нас) #
    ######################################################################
    @property
    def _state_size(self):
        return len(self._state)

    @property
    def _position_x(self):
        return self._state[self.POSITION_X_IDX]

    @_position_x.setter
    def _position_x(self, position_x):
        self._state[self.POSITION_X_IDX] = position_x

    @property
    def _position_y(self):
        return self._state[self.POSITION_Y_IDX]

    @_position_y.setter
    def _position_y(self, position_y):
        self._state[self.POSITION_Y_IDX] = position_y

    @property
    def _yaw(self):
        return self._state[self.YAW_IDX]

    @_yaw.setter
    def _yaw(self, yaw):
        self._state[self.YAW_IDX] = yaw

    @property
    def _linear_velocity(self):
        return self._state[self.LINEAR_VELOCITY_IDX]

    @_linear_velocity.setter
    def _linear_velocity(self, velocity):
        self._state[self.LINEAR_VELOCITY_IDX] = velocity

    @property
    def _linear_velocity_x(self):
        return self._linear_velocity * np.cos(self._yaw)

    @property
    def _linear_velocity_y(self):
        return self._linear_velocity * np.sin(self._yaw)

    @property
    def _angular_velocity(self):
        return self._state[self.ANGULAR_VELOCITY_IDX]

    @_angular_velocity.setter
    def _angular_velocity(self, angular_velocity):
        self._state[self.ANGULAR_VELOCITY_IDX] = angular_velocity

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, time):
        assert isinstance(time, Timestamp)
        assert self._time <= time
        self._time = time
