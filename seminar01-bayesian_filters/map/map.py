import numpy as np

class KeypointsMap(object):
    def __init__(self, keypoints_position=None, keypoints_id=None):
        if keypoints_position is None:
            kyepoints_position = []
            keypoints_id = []
            for x in range(-10, 10, 5):
                for y in range(-10, 10, 5):
                    keypoints_position.append((x, y))
                    keypoints_id.append(len(keypoints_id))

            keypoints_position = np.array(keypoints_position)
            keypoints_id = np.array(keypoints_id)

        self.position = keypoints_position
        self.kp_id = keypoints_id