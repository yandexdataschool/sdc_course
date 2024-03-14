import numpy as np
from .car import Car
from .timestamp import Timestamp
from .kalman_movement_model import KalmanMovementModel
from .kalman_can_sensor import KalmanCanSensor
from .kalman_gps_sensor import KalmanGpsSensor
from .kalman_imu_sensor import KalmanImuSensor
from .kalman_filter import kalman_transit_covariance


class KalmanCar(Car):
    def __init__(self, initial_covariance_matrix=None, *args, **kwargs):
        super(KalmanCar, self).__init__(*args, **kwargs)
        if initial_covariance_matrix is None:
            initial_covariance_matrix = 100 * np.eye(self.state_size)
        self._covariance_matrix = initial_covariance_matrix

    @property
    def state_size(self):
        return self._state_size

    @property
    def state(self):
        return np.array(self._state)

    @state.setter
    def state(self, state):
        state = np.array(state, copy=False)
        assert state.shape == (self.state_size,)
        self._state = state
        # Храним историю состояний
        self._positions_x.append(self._position_x)
        self._positions_y.append(self._position_y)
        self._yaws.append(self._yaw)
        self._velocities.append(self._velocity)
        self._velocities_x.append(self._velocity_x)
        self._velocities_y.append(self._velocity_y)
        self._omegas.append(self._omega)

    @property
    def covariance_matrix(self):
        return self._covariance_matrix

    @covariance_matrix.setter
    def covariance_matrix(self, covariance_matrix):
        covariance_matrix = np.array(covariance_matrix, copy=False)
        assert covariance_matrix.shape == (self.state_size, self.state_size)
        self._covariance_matrix = covariance_matrix

    def add_sensor(self, sensor):
        if isinstance(sensor, KalmanCanSensor):
            self._can_sensor = sensor
        elif isinstance(sensor, KalmanGpsSensor):
            self._gps_sensor = sensor
        elif isinstance(sensor, KalmanImuSensor):
            self._imu_sensor = sensor
        else:
            assert False, f'Unknown sensor type {type(sensor)}'
        self._sensors.append(sensor)
        sensor._initialize(self)

    def set_movement_model(self, movement_model=None):
        if movement_model is None:
            movement_model = KalmanMovementModel()
        assert isinstance(movement_model, KalmanMovementModel)
        # Привязываем модель движения к автомобилю
        self._movement_model = movement_model
        # Привязываем автомобиль к модели движения
        movement_model._initialize(self)

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
        self._positions_x.append(self._position_x)
        self._positions_y.append(self._position_y)
        self._yaws.append(self._yaw)
        self._velocities.append(self._velocity)
        self._velocities_x.append(self._velocity_x)
        self._velocities_y.append(self._velocity_y)
        self._omegas.append(self._omega)
