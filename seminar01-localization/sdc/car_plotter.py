import numpy as np
from matplotlib.patches import Ellipse, Rectangle

from .car import Car


class CarPlotter:
    def __init__(self, car_width=1, car_height=0.5,
                 real_color='g', obs_color='b', pred_color='r',
                 head_width=1):
        """
        :param car_width: Ширина автомобиля
        :param car_height: Длина автомобиля
        :param real_color: Цвет для отрисовки реального положения
        :param obs_color: Цвет для отрисовки наблюдений
        :param pred_color: Цвет для отрисовки калмановского предсказания
        :param head_width: Ширина стрелки при отрисовке скорости автомобиля
        """
        self.car_width = car_width
        self.car_height = car_height

        self.real_color = real_color
        self.obs_color = obs_color
        self.pred_color = pred_color

        self.real_vel_ = None
        self.obs_vel_ = None
        self.head_width = head_width

    def plot_car(self, ax, car, marker_size=6):
        """Отрисовывает положение автомобиля и покзания GPS и одометрии.
        :param marker_size: Линейный размер точки положения и GPS-показания
        :param color: Цвет
        """
        assert isinstance(car, Car)
        real_position_x = car._position_x
        real_position_y = car._position_y
        real_velocity_x = car._velocity_x
        real_velocity_y = car._velocity_y

        # Отрисовка реального положения центра автомобиля
        real_position = np.array([real_position_x, real_position_y])
        self._plot_point(ax, real_position, marker='o', marker_color=self.real_color, marker_size=marker_size)
        # Отрисовка реального направления движения
        self.real_vel_ = ax.arrow(real_position_x, real_position_y,
                                  real_velocity_x, real_velocity_y,
                                  color=self.real_color,
                                  head_width=self.head_width)
        # Отрисовка "прямоугольника" автомобиля
        angle = np.arctan2(real_velocity_y, real_velocity_x)
        y_rec = real_position_y - 0.5 * (self.car_height * np.cos(angle) + self.car_width * np.sin(angle))
        x_rec = real_position_x - 0.5 * (self.car_width * np.cos(angle) - self.car_height * np.sin(angle))
        rec = Rectangle(xy=(x_rec, y_rec), width=self.car_width, height=self.car_height,
                        angle=np.rad2deg(angle))
        rec.set_facecolor('none')
        rec.set_edgecolor('k')
        ax.add_artist(rec)

        # Если установлен GPS-датчик, то отрисовать показания GPS
        if car.gps_sensor is not None:
            gps_noise_covariance = car.gps_sensor.get_noise_covariance()
            self._plot_ellipse(ax, car.gps_sensor.observe(), gps_noise_covariance, color=self.obs_color)
            self._plot_point(ax, car.gps_sensor.observe(), marker='*',
                             marker_color=self.obs_color, marker_size=marker_size)

    def plot_kalman_car(self, ax, kalman_car):
        # Извлекаем состояние
        position_x = kalman_car._position_x
        position_y = kalman_car._position_y
        velocity_x = kalman_car._velocity_x
        velocity_y = kalman_car._velocity_y
        covariance = kalman_car.covariance_matrix
        # Отрисовка положения
        mu = np.array([position_x, position_y])
        sigma = covariance[:2, :2]
        self._plot_ellipse(ax, mu, sigma, color=self.pred_color)
        self._plot_point(ax, mu, marker='o', marker_size=6, marker_color=self.pred_color)
        # Отрисовка скорости
        mu = np.array([position_x + velocity_x, position_y + velocity_y])
        ax.arrow(position_x, position_y, velocity_x, velocity_y, color=self.pred_color,
                 head_width=self.head_width)

    def plot_trajectory(self, ax, car, traj_color='g'):
        """Отрисовывает весь уже проделанный автомобилем путь"""
        ax.plot(car._positions_x, car._positions_y, linestyle='-', color=traj_color)

    def plot_observations(self, ax, x, y, color='b'):
        ax.plot(x, y, linestyle='-', color=color)

    def get_limits(self, car):
        """Иногда требуется подогнать размер полотна, чтобы оно вмещало в себя всю траектория.
        Данный метод возвращает диапазоны значений вдоль каждой из осей.
        """
        min_pos_x = np.min(car._positions_x)
        max_pos_x = np.max(car._positions_x)
        min_pos_y = np.min(car._positions_y)
        max_pos_y = np.max(car._positions_y)
        max_vel_x = np.max(np.abs(car._velocities_x))
        max_vel_y = np.max(np.abs(car._velocities_y))
        # Дополнительная граница в полкорпуса
        max_length = max(self.car_width, self.car_height)
        x_limits = (min_pos_x - max_vel_x - 0.5 * max_length, max_pos_x + max_vel_x + 0.5 * max_length)
        y_limits = (min_pos_y - max_vel_y - 0.5 * max_length, max_pos_y + max_vel_y + 0.5 * max_length)
        return x_limits, y_limits

    def _plot_point(self, ax, mu, marker='o', marker_size=6, marker_color='b'):
        """Отрисовывает точку"""
        ax.scatter(mu[0], mu[1], marker=marker, color=marker_color, s=marker_size**2, edgecolors='k')

    def _plot_ellipse(self, ax, mu, sigma, color='b'):
        """Отрисовывает эллипс ковариации
        :param ax: полотно
        :param mu: цент нормального распределения
        :param sigma: ковариация нормального распределения
        :param marker: тип маркера для отображения центра
        :parma marker_size: линейный размер маркера для отображения центра
        :param color: цвет маркера и эллипса
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
