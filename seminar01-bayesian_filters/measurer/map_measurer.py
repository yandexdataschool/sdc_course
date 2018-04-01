import numpy as np
import cv2

class MapMeasurer(object):
    def __init__(self, keypoints_map):
        self.map_ = keypoints_map.position
        self.kp_id_ = keypoints_map.kp_id

    def measure(self, robot):
        yaw = robot.state_[-1]
        pos = robot.state_[:2]

        hom_map = np.vstack(self.map_, np.ones(len(self.map_)))

        robot_to_map = np.array((
            (np.cos(yaw), -np.sin(yaw), pos[0]),
            (np.sin(yaw),  np.cos(yaw), pos[1]),
            (          0,            0,      1)
        ))

        map_in_robot = (np.linalg.inv(robot_to_map) * self.map_.T).T
        map_in_robot = map_in_robot[:, :2]

        return map_in_robot, self.kp_id_