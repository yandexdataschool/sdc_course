import numpy as np

class Transform2D(object):
    '''Represents transformations and provides convenient operations'''
    
    def __init__(self, R, t):
        self._R = R
        self._t = t
        
    @classmethod
    def from_pose(cls, pose):
        sin_yaw = np.sin(pose[2])
        cos_yaw = np.cos(pose[2])
        R = np.array([cos_yaw, -sin_yaw, sin_yaw, cos_yaw]).reshape((2, 2))
        t = np.array(pose[:2])
        return cls(R=R, t=t)
    
    def transform(self, point):
        return np.dot(self._R, point) + self._t
    
    def inverse(self):
        return Transform2D(R=self._R.T, t=-np.dot(self._R.T, self._t))
    
    def to_pose(self):
        yaw = np.arctan2(self._R[1, 0], self._R[0, 0])
        return np.array([self._t[0], self._t[1], yaw])
    
    def __mul__(self, rhs):
        return Transform2D(R=np.dot(self._R, rhs._R), t=np.dot(self._R, rhs._t) + self._t)
