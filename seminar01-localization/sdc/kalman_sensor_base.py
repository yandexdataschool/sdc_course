# -*- coding: utf-8 -*-
import numpy as np
from .kalman_filter import kalman_process_observation


class KalmanSensorBase(object):
    """
    Kalman model for sensor observations
    """
    def __init__(self,  noise_variances=None):
        """
        :param noise_covariance: expected value of observation noise variance.
        """
        if noise_variances is None:
            self._noise_variances = np.zeros(self.observation_size, dtype=np.float64)
        else:
            self._noise_variances = np.array(noise_variances)
            assert self._noise_variances.shape == (self.observation_size,)

    def _initialize(self, car_model):
        """Method should be called when the sensor is added to a car"""
        self._car_model = car_model

    @property
    def state_size(self):
        return self._car_model._state_size

    @property
    def observation_size(self):
        """Returns observation size"""
        assert False, 'Not implemented'

    def get_observation_matrix(self):
        """Returns observation C for kalman filter"""
        assert False, 'Not implemented'

    def get_noise_covariance(self):
        """Returns diagonal matrix with noise covariance"""
        return np.diag(self._noise_variances)

    def process_observation(self, observation):
        C = self.get_observation_matrix()
        Q = self.get_noise_covariance()
        mu = self._car_model.state
        S = self._car_model.covariance_matrix
        new_mu, new_S = kalman_process_observation(mu, S, observation, C, Q)
        self._car_model.state = new_mu
        self._car_model.covariance_matrix = new_S
