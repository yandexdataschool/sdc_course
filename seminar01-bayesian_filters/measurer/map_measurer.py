import numpy as np
import cv2

class MapMeasurer(object):
    def __init__(self, keypoints_map):
        self.map_ = keypoints_map.position

    def measure(self, yaw, pos):
        
        hom_map = np.hstack((self.map_, np.ones((self.map_.shape[0], 1))))
        robot_to_map = np.array((
            (np.cos(yaw), -np.sin(yaw), pos[0]),
            (np.sin(yaw),  np.cos(yaw), pos[1]),
            (          0,            0,      1)
        ))

        map_in_robot = (np.linalg.inv(robot_to_map).dot(hom_map.T)).T
        map_in_robot = map_in_robot[:, :2]

        return map_in_robot.ravel()
