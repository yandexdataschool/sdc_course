# -*- coding: utf-8 -*-
import numpy as np
from .timestamp import Timestamp


class CarSensorBase(object):
    """
    Base class for car sensor.

    Each sensor is parametrised by a level of noise which it introduces in observation.

    Sensor also knows about global time (via self._car.time field) and should return
    the same observation value if one asks the value multiple times without changing the
    global time.

    A new sensor can be created using the following template (see also CarSensor, ImuSensor,
    GpsSensor implementations):

        class NewSensor(CarSensorBase):
            def __init__(self, ...):
                super(NewSensor, self).__init__()
                ...

        @property
        def observation_size(self):
            return N

        def _observe_clear(self):
            return np.array(...)

    """
    def __init__(self, noise_variances=None, random_state=None):
        # each sensor has his own random values generator
        self._gen = np.random.RandomState(random_state)

        if noise_variances is None:
            self._noise_variances = np.zeros(self.observation_size, dtype=np.float64)
        else:
            self._noise_variances = np.array(noise_variances)
            assert self._noise_variances.shape == (self.observation_size,)

        self._car = None
        self._last_time = None
        self._last_observation = None
        self._history = []  # observation history

    def _initialize(self, car):
        """Method should be called when the sensor is added to a car"""
        self._car = car

    @property
    def state_size(self):
        return self._car._state_size

    def get_noise_covariance(self):
        """Diagonal metrix, ground truth value of noise covariance"""
        return np.diag(self._noise_variances)

    def observe(self):
        """
        Return observation value with noise. Two calls of this function for the same
        timestamp should return the same values.
        """
        if  self._last_time is not None and self._last_time == self._car.time:
            return self._last_observation

        observation = self._observe_clear()
        assert observation.shape == (self.observation_size,)

        for i, variance in enumerate(self._noise_variances):
            if variance > 0:
                observation[i] += self._gen.normal(scale=np.sqrt(variance))
        self._last_observation = observation
        self._last_time = Timestamp.nanoseconds(self._car.time.to_nanoseconds())
        observation = np.array(self._last_observation)
        self._history.append(observation)
        return observation

    @property
    def history(self):
        return np.array(self._history)

    #####################################################
    #      Methods to be overloaded  in subclasses      #
    #####################################################
    @property
    def observation_size(self):
        """Returns the size of observation"""
        assert False, 'Not implemented'

    def _observe_clear(self):
        """Returns clear observation value (without noise)"""
        assert False, 'Not implemented'
