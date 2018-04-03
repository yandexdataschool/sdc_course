import numpy as np

class KeypointsMap(object):
    def __init__(self, keypoints_position=None, keypoints_id=None):
        if keypoints_position is None:
            keypoints_position = []
            for x in [-10, 10]:
                for y in [-10, 10]:
                    keypoints_position.append((x, y))
                    
            keypoints_position = np.array(keypoints_position)

        self.position = keypoints_position