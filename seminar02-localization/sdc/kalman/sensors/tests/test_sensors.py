import numpy as np
from sdc.kalman.sensors.gnss import KalmanGnssSensor
from sdc.kalman.sensors.imu import KalmanImuSensor
from sdc.kalman.sensors.wheel_odometry import KalmanWheelOdometrySensor


def test_kalman_gnss_sensor():
    sensor = KalmanGnssSensor("KalmanGNSS", noise_variances=[5, 5])
    assert sensor.observation_size == 2
    assert np.all(sensor.get_noise_covariance() == np.diag([5, 5]))


def test_kalman_imu_sensor():
    sensor = KalmanImuSensor("KalmanIMU", noise_variances=[5])
    assert sensor.observation_size == 1
    assert np.all(sensor.get_noise_covariance() == np.diag([5]))


def test_kalman_wheel_odometry_sensor():
    sensor = KalmanWheelOdometrySensor("KlamanWO", noise_variances=[5])
    assert sensor.observation_size == 1
    assert np.all(sensor.get_noise_covariance() == np.diag([5]))
