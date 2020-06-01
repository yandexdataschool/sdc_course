# -*- coding: utf-8 -*-
import numpy as np
from .timestamp import Timestamp
from .movement_model_base import MovementModelBase
from .car_sensor_base import CarSensorBase
from .can_sensor import CanSensor
from .gps_sensor import GpsSensor
from .imu_sensor import ImuSensor


class Car(object):
    """
    Simple model of car in 2d world.

    Hidden variables of the model are position, velocity and orientation around oX axis.
    (pos_x, pos_y, vel, yaw).
    It also contains the current timestamp for which the hidden state is valid.

    Ground truth state of the car is unknown to external user. Car has a set of sensors
    which provide data with some noise.
    """

    POS_X_INDEX = 0
    POS_Y_INDEX = 1
    YAW_INDEX = 2
    VEL_INDEX = 3
    OMEGA_INDEX = 4

    def __init__(self,
                 initial_position=None,
                 initial_velocity=None,
                 initial_yaw=None,
                 initial_omega=None,
                 movement_model=None):
        """
        :param initial_position: list, tuple, np.ndarray with two elements (shape = (2,))
        :param initial_velocity: float
        :param intial_yaw: float
        :param movement_model: MovementModelBase or None. Represents the real movement trajectory
        """
        assert isinstance(initial_position, (list, tuple, np.ndarray))
        if initial_position is None:
            self.initial_position = np.zeros(2, dtype=np.float64)
        else:
            self.initial_position = np.array(initial_position, dtype=np.float64)
            assert self.initial_position.shape == (2,)

        if initial_velocity is None:
            self.initial_velocity = 0.
        else:
            self.initial_velocity = float(initial_velocity)

        if initial_yaw is None:
            self.initial_yaw = 0.
        else:
            self.initial_yaw = float(initial_yaw)

        if initial_omega is None:
            self.initial_omega = 0.
        else:
            self.initial_omega = float(initial_omega)

        # initialization of the car state
        self._state = np.zeros(5)
        self._position_x = self.initial_position[0]
        self._position_y = self.initial_position[1]
        self._yaw = self.initial_yaw
        self._velocity = self.initial_velocity
        self._omega = self.initial_omega

        self._time = Timestamp()
        assert self._time.nsec == 0 and self._time.sec == 0

        # car trajectory is defined by some movement model
        self.set_movement_model(movement_model)
        # car sensors initialization
        self._sensors = []
        self._can_sensor = None  # odometry
        self._gps_sensor = None
        self._imu_sensor = None

        # car states history
        self._positions_x = []
        self._positions_y = []
        self._velocities_x = []
        self._velocities_y = []
        self._velocities = []
        self._yaws = []
        self._omegas = []

    def __str__(self):
        return '{}(x={:.2f}[m], y={:.2f}[m], yaw={:.2f}[rad], v={:.2f}[m/s], '\
            'omega={:.2f}[rad/s], t={})'.format(
                type(self).__name__,
                self._position_x, self._position_y, self._yaw, self._velocity,
                self._omega, self.time)

    def set_movement_model(self, movement_model=None):
        if movement_model is None:
            movement_model = MovementModelBase()
        assert isinstance(movement_model, MovementModelBase)
        self._movement_model = movement_model  # assign movement model to car
        movement_model._initialize(self)  # assign car to movement model

    def add_sensor(self, sensor):
        assert isinstance(sensor, CarSensorBase)
        if isinstance(sensor, CanSensor):
            self._can_sensor = sensor
        elif isinstance(sensor, GpsSensor):
            self._gps_sensor = sensor
        elif isinstance(sensor, ImuSensor):
            self._imu_sensor = sensor
        else:
            assert False, 'Unknown sensor type'
        self._sensors.append(sensor)
        sensor._initialize(self)

    def move(self, dt):
        assert isinstance(dt, Timestamp)
        self._movement_model._move(dt)
        # update car state history
        self._positions_x.append(self._position_x)
        self._positions_y.append(self._position_y)
        self._yaws.append(self._yaw)
        self._velocities.append(self._velocity)
        self._velocities_x.append(self._velocity_x)
        self._velocities_y.append(self._velocity_y)
        self._omegas.append(self._omega)

    ######################################################################
    #  Access to car components - movement model and sensors             #
    ######################################################################
    @property
    def movement_model(self):
        return self._movement_model

    @property
    def sensors(self):
        return self._sensors

    @property
    def can_sensor(self):
        return self._can_sensor

    @property
    def gps_sensor(self):
        return self._gps_sensor

    @property
    def imu_sensor(self):
        return self._imu_sensor

    ######################################################################
    # Access to hidden state                                             #
    ######################################################################
    @property
    def _state_size(self):
        return len(self._state)

    @property
    def _position_x(self):
        return self._state[self.POS_X_INDEX]

    @_position_x.setter
    def _position_x(self, position_x):
        self._state[self.POS_X_INDEX] = position_x

    @property
    def _position_y(self):
        return self._state[self.POS_Y_INDEX]

    @_position_y.setter
    def _position_y(self, position_y):
        self._state[self.POS_Y_INDEX] = position_y

    @property
    def _yaw(self):
        return self._state[self.YAW_INDEX]

    @_yaw.setter
    def _yaw(self, yaw):
        self._state[self.YAW_INDEX] = yaw

    @property
    def _velocity(self):
        return self._state[self.VEL_INDEX]

    @_velocity.setter
    def _velocity(self, velocity):
        self._state[self.VEL_INDEX] = velocity

    @property
    def _velocity_x(self):
        return self._velocity * np.cos(self._yaw)

    @property
    def _velocity_y(self):
        return self._velocity * np.sin(self._yaw)

    @property
    def _omega(self):
        return self._state[self.OMEGA_INDEX]

    @_omega.setter
    def _omega(self, omega):
        self._state[self.OMEGA_INDEX] = omega

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, time):
        assert isinstance(time, Timestamp)
        assert self._time <= time
        self._time = time
