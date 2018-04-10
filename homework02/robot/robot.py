import numpy as np


class Robot(object):
    def __init__(self, pos, vel, yaw=0.0, odometer_noise_x=0.1, odometer_noise_y=0.1):
        assert type(pos) == np.ndarray
        assert type(vel) == np.ndarray
        assert pos.shape == (2,)
        assert vel.shape == (2,)

        self.state_ = np.hstack((pos, vel, np.array([yaw])))
        self.odometer_noise = (odometer_noise_x, odometer_noise_y)

    def move(self, dt):
        self.state_[:4] = np.array((
            (1, 0, dt,  0),
            (0, 1,  0, dt),
            (0, 0,  1,  0),
            (0, 0,  0,  1)
        )).dot(self.state_[:4])
        
    def velocity(self):
        return self.state_[2:4] + np.random.normal(scale=self.odometer_noise, size=2)
