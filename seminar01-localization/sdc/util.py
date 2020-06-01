import numpy as np

from sdc.car import Car
from sdc.linear_movement_model import LinearMovementModel
from sdc.can_sensor import CanSensor
from sdc.gps_sensor import GpsSensor
from sdc.imu_sensor import ImuSensor
from sdc.kalman_car import KalmanCar
from sdc.kalman_can_sensor import KalmanCanSensor
from sdc.kalman_gps_sensor import KalmanGpsSensor
from sdc.kalman_imu_sensor import KalmanImuSensor
from sdc.kalman_movement_model import KalmanMovementModel


def create_car(initial_position=[5, 5],
               initial_velocity=5,
               initial_omega=0.0,
               initial_yaw=np.pi / 4,
               can_noise_variances=[0.25],   # std deviation == 0.5 m
               gps_noise_variances=[1, 1],   # std deviation = 1 m
               imu_noise_variances=None,     # don't use IMU by default
               random_state=0):
    # Car initial state
    car = Car(
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        initial_yaw=initial_yaw,
        initial_omega=initial_omega)
    # adding sensors
    if can_noise_variances is not None:
        car.add_sensor(CanSensor(noise_variances=can_noise_variances, random_state=random_state))
        random_state += 1
    if gps_noise_variances is not None:
        car.add_sensor(GpsSensor(noise_variances=gps_noise_variances, random_state=random_state))
        random_state += 1
    if imu_noise_variances is not None:
        car.add_sensor(ImuSensor(noise_variances=imu_noise_variances, random_state=random_state))
        random_state += 1
    # movement model
    movement_model = LinearMovementModel()
    car.set_movement_model(movement_model)
    return car


def create_kalman_car(car, gps_variances=None, can_variances=None, imu_variances=None):
    """Creates kalman model of a car"""
    noise_covariance_density = np.diag([
        0.1,
        0.1,
        0.1,   # yaw variance
        0.1,   # velocity variance
        0.1    # angular velocity variance
    ])
    # initial state of kalman model
    kalman_car = KalmanCar(
        initial_position=car.initial_position,
        initial_velocity=car.initial_velocity,
        initial_yaw=car.initial_yaw,
        initial_omega=car.initial_omega)
    # initial covariance matrix
    kalman_car.covariance_matrix = noise_covariance_density

    # movement model
    kalman_movement_model = KalmanMovementModel(noise_covariance_density=noise_covariance_density)
    kalman_car.set_movement_model(kalman_movement_model)

    for sensor in car.sensors:
        noise_variances = sensor._noise_variances
        if isinstance(sensor, GpsSensor):
            noise_variances = noise_variances if gps_variances is None else gps_variances
            kalman_sensor = KalmanGpsSensor(noise_variances=noise_variances)
        elif isinstance(sensor, CanSensor):
            noise_variances = noise_variances if can_variances is None else can_variances
            kalman_sensor = KalmanCanSensor(noise_variances=noise_variances)
        elif isinstance(sensor, ImuSensor):
            noise_variances = noise_variances if imu_variances is None else imu_variances
            kalman_sensor = KalmanImuSensor(noise_variances=noise_variances)
        else:
            assert False
        kalman_car.add_sensor(kalman_sensor)
    return kalman_car

