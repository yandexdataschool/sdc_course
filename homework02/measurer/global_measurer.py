import numpy as np

class GlobalMeasurer(object):
    def __init__(self, noise=1):
        self.noise_ = noise
    
    def measure(self, robot):
        true_pos = robot.state_[:2]
        return true_pos +\
            np.random.normal(scale=self.noise_, size=2)
