import numpy as np
import typing as T
from sdc.core.timestamp import Timestamp
from sdc.kalman.movement_model import KalmanMovementModel
from sdc.kalman.sensors.base import KalmanSensorBase
from sdc.kalman.filter import kalman_transit_covariance


class KalmanCar:
    POSITION_X_IDX = 0
    POSITION_Y_IDX = 1
    YAW_IDX = 2
    LINEAR_VELOCITY_IDX = 3
    ANGULAR_VELOCITY_IDX = 4

    def __init__(self, initial_covariance_matrix=None, *args, **kwargs):
        self._sensor_by_id: T.Dict[str, KalmanSensorBase] = dict()

        if initial_covariance_matrix is None:
            initial_covariance_matrix = 100 * np.eye(self.state_size)

        self._covariance_matrix = initial_covariance_matrix
        self._state = np.zeros(self.state_size, dtype=np.float64)

        # States history
        self._positions_x = []
        self._positions_y = []
        self._yaws = []
        self._linear_velocities = []
        self._angular_velocities = []

    @property
    def state_size(self) -> int:
        return 5

    @property
    def state(self):
        return np.array(self._state)

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

    @state.setter
    def state(self, state):
        state = np.array(state, copy=False)
        assert state.shape == (self.state_size,)
        self._state = state
        # Храним историю состояний
        self._positions_x.append(self.position_x)
        self._positions_y.append(self.position_y)
        self._yaws.append(self.yaw)
        self._linear_velocities.append(self.linear_velocity)
        self._angular_velocities.append(self.angular_velocity)

    @property
    def covariance_matrix(self):
        return self._covariance_matrix

    @covariance_matrix.setter
    def covariance_matrix(self, covariance_matrix):
        covariance_matrix = np.array(covariance_matrix, copy=False)
        assert covariance_matrix.shape == (self.state_size, self.state_size)
        self._covariance_matrix = covariance_matrix

    def add_sensor(self, sensor: KalmanSensorBase):
        assert sensor.id not in self._sensor_by_id
        self._sensor_by_id[sensor.id] = sensor
        sensor._mount(self)

    def get_sensor(self, sensor_id: str) -> KalmanSensorBase:
        return self._sensor_by_id[sensor_id]

    @property
    def movement_model(self) -> T.Optional[KalmanMovementModel]:
        return self._movement_model

    def set_movement_model(self, movement_model=None):
        if movement_model is None:
            movement_model = KalmanMovementModel()
        assert isinstance(movement_model, KalmanMovementModel)
        # Привязываем модель движения к автомобилю
        self._movement_model = movement_model
        # Привязываем автомобиль к модели движения
        movement_model._attach(self)

    def move(self, dt):
        assert isinstance(dt, Timestamp)
        # Делаем предсказание на момент времени t + dt
        new_mu = self.movement_model.get_next_state(dt)

        J = self.movement_model.get_state_jacobian_matrix(dt)
        R = self.movement_model.get_noise_covariance(dt)
        S = self.covariance_matrix
        new_S = kalman_transit_covariance(S, J, R)

        self.state = new_mu
        self.covariance_matrix = new_S

        # Храним историю состояний
        self._positions_x.append(self.position_x)
        self._positions_y.append(self.position_y)
        self._yaws.append(self.yaw)
        self._linear_velocities.append(self.linear_velocity)
        self._angular_velocities.append(self.angular_velocity)
