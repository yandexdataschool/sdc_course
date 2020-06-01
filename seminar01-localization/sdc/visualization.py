# -*- coding: utf-8 -*-
import numpy as np
from .car import Car
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, Rectangle


class CarPlotter(object):
    def __init__(self, car_width=1, car_length=0.5,
                 real_color='g', obs_color='b', pred_color='r',
                 head_width=1):
        """
        :param car_width: car width
        :param car_length: car length
        :param real_color: color for drawing real position
        :param obs_color:  color for drawing observed position
        :param pred_color: color for drawing kalman prediction
        :param head_width: width of arrow for drawing car heading
        """
        self.car_width = car_width
        self.car_length = car_length

        self.real_color = real_color
        self.obs_color = obs_color
        self.pred_color = pred_color

        self.real_vel_ = None
        self.obs_vel_ = None
        self.head_width = head_width

    def plot_car(self, ax, car, marker_size=6):
        """
        Draws car real position + gps + odometry
        :param marker_size: linear size of marker for position and gps drawing
        """
        assert isinstance(car, Car)
        real_position_x = car._position_x
        real_position_y = car._position_y
        real_velocity_x = car._velocity_x
        real_velocity_y = car._velocity_y

        # drawing the real position of car center
        real_position = np.array([real_position_x, real_position_y])
        self._plot_point(ax, real_position, marker='o', marker_color=self.real_color,
                marker_size=marker_size)
        # drawing the real heading
        self.real_vel_ = plt.arrow(real_position_x, real_position_y,
                                   real_velocity_x, real_velocity_y,
                                   color=self.real_color,
                                   head_width=self.head_width)
        # drawing the rectangle to represent a car
        angle = np.arctan2(real_velocity_y, real_velocity_x)
        y_rec = real_position_y - 0.5 * (self.car_length * np.cos(angle) +
                self.car_width * np.sin(angle))
        x_rec = real_position_x - 0.5 * (self.car_width * np.cos(angle) -
                self.car_length * np.sin(angle))
        rec = Rectangle(xy=(x_rec, y_rec), width=self.car_width, height=self.car_length,
                        angle=np.rad2deg(angle))
        rec.set_facecolor('none')
        rec.set_edgecolor('k')
        ax.add_artist(rec)

        # if car has gps sensor, draw gps observations
        if car.gps_sensor is not None:
            gps_noise_covariance = car.gps_sensor.get_noise_covariance()
            self._plot_ellipse(ax, car.gps_sensor.observe(), gps_noise_covariance,
                    color=self.obs_color)
            self._plot_point(ax, car.gps_sensor.observe(), marker='*',
                             marker_color=self.obs_color, marker_size=marker_size)

    def plot_kalman_car(self, ax, kalman_car):
        # car state retrieval
        position_x = kalman_car._position_x
        position_y = kalman_car._position_y
        velocity_x = kalman_car._velocity_x
        velocity_y = kalman_car._velocity_y
        covariance = kalman_car.covariance_matrix
        # position drawing
        mu = np.array([position_x, position_y])
        sigma = covariance[:2, :2]
        self._plot_ellipse(ax, mu, sigma, color=self.pred_color)
        self._plot_point(ax, mu, marker='o', marker_size=6, marker_color=self.pred_color)
        # drawing the velocity
        mu = np.array([position_x + velocity_x, position_y + velocity_y])
        plt.arrow(position_x, position_y, velocity_x, velocity_y, color=self.pred_color,
                  head_width=self.head_width)


    def plot_trajectory(self, ax, car, traj_color='g'):
        """Draws car trajectory"""
        ax.plot(car._positions_x, car._positions_y, linestyle='-', color=traj_color)

    def plot_observations(self, ax, x, y, color='b'):
        ax.plot(x, y, linestyle='-', color=color)

    def get_limits(self, car):
        """
        Returns x-axis and y-axis limits to fit the whole trajectory
        """
        min_pos_x = np.min(car._positions_x)
        max_pos_x = np.max(car._positions_x)
        min_pos_y = np.min(car._positions_y)
        max_pos_y = np.max(car._positions_y)
        max_vel_x = np.max(np.abs(car._velocities_x))
        max_vel_y = np.max(np.abs(car._velocities_y))
        # adds a half of the car dimensions to borders
        max_length = max(self.car_width, self.car_length)
        x_limits = (min_pos_x - max_vel_x - 0.5 * max_length,
                    max_pos_x + max_vel_x + 0.5 * max_length)
        y_limits = (min_pos_y - max_vel_y - 0.5 * max_length,
                    max_pos_y + max_vel_y + 0.5 * max_length)
        return x_limits, y_limits

    def _plot_point(self, ax, mu, marker='o', marker_size=6, marker_color='b'):
        """Draws one point"""
        ax.scatter(mu[0], mu[1], marker=marker, color=marker_color, s=marker_size**2,
                   edgecolors='k')

    def _plot_ellipse(self, ax, mu, sigma, color='b'):
        """
        Draws covariance ellipse
        :param ax: canvas
        :param mu: distribution mean value
        :param sigma: distribution covariance
        :param marker: marker type for drawing the mean value
        :parma marker_size: linear size of marker for drawing the mean value
        :param color: color of the marker and the ellipse
        """
        assert mu.shape == (2,)
        assert sigma.shape == (2, 2)
        lambda_, v = np.linalg.eig(sigma)
        lambda_ = np.sqrt(lambda_)
        ell = Ellipse(xy=mu, width=lambda_[0] * 2, height=lambda_[1] * 2,
                      angle=np.rad2deg(np.arccos(v[0, 0])), alpha=0.3, zorder=5)
        ell.set_edgecolor('k')
        ell.set_facecolor(color)
        ax.add_artist(ell)
        return ell
