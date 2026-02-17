import numpy as np
from sdc.sim.sensors.gnss import GnssSensor
from sdc.sim.sensors.imu import ImuSensor
from sdc.sim.sensors.wheel_odometry import WheelOdometrySensor


def test_gnss_sensor():
    sensor = GnssSensor(
        sensor_id="GNSS",
        frequency=5,
        topic="/gnss",
        noise_variances=[15, 15]
    )
    assert sensor.observation_size == 2
    assert np.all(sensor.get_noise_covariance() == np.diag([15, 15]))


def test_wheel_odometry_sensor():
    sensor = WheelOdometrySensor(
        sensor_id="WO",
        frequency=50,
        topic="/odometry",
        noise_variances=[15]
    )
    assert sensor.observation_size == 1
    assert np.all(sensor.get_noise_covariance() == np.diag([15]))


def test_imu_sensor():
    sensor = ImuSensor(
        sensor_id="IMU",
        frequency=100,
        topic="/imu",
        noise_variances=[1])
    assert sensor.observation_size == 1
    assert np.all(sensor.get_noise_covariance() == np.diag([1]))
