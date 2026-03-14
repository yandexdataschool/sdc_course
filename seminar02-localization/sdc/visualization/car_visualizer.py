import numpy as np
import matplotlib.pyplot as plt
from .robot_visualizer_base import RobotVisualizerBase
from matplotlib.patches import Rectangle


class CarVisualizer(RobotVisualizerBase):
    def __init__(self, robot):
        self._robot = robot

    def draw(self, ax):
        rectangle_xy = gt_body_position + body_to_world_rotation_matrix @ np.array([0, -0.5 * vehicle_width])
        rec = Rectangle(
            xy=rectangle_xy,
            width=vehicle_length,
            height=vehicle_width,
            angle=np.rad2deg(yaw),
            linewidth=3)
        rec.set_facecolor('none')
        rec.set_edgecolor('k')
        ax.add_artist(rec)
