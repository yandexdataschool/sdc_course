import numpy as np
import matplotlib.pyplot as plt
from .robot_visualizer_base import RobotVisualizerBase
from matplotlib.patches import Rectangle


class UnicycleRobotVisualizer(RobotVisualizerBase):
    def __init__(self, robot):
        self._robot = robot

    def draw(self, ax):
        # Getting current robot state
        position_x = self._robot.state.position_x
        position_y = self._robot.state.position_y
        yaw = self._robot.state.yaw

        # Wheel rectangle
        wheel_length = self._robot.params.wheel_length
        wheel_width = self._robot.params.wheel_width
        body_to_world_rotation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)],
        ])
        wheel_rectangle_xy = \
            np.array([position_x, position_y]) + \
            body_to_world_rotation_matrix @ np.array([-0.5 * wheel_length, -0.5 * wheel_width])
        wheel_rectangle = Rectangle(
            xy=wheel_rectangle_xy,
            width=wheel_length,
            height=wheel_width,
            angle=np.rad2deg(yaw),
            linewidth=2)
        wheel_rectangle.set_facecolor('gray')
        wheel_rectangle.set_edgecolor('k')
        ax.add_artist(wheel_rectangle)

        # Center
        ax.scatter(position_x, position_y, s=16, color='r')

        # Visualizing body frame OX and OY axes
        direction = body_to_world_rotation_matrix @ np.array([1.25 * wheel_length, 0.])
        plt.arrow(
            position_x, position_y,
            direction[0], direction[1],
            color='r',
            head_width=0.05)
        direction = body_to_world_rotation_matrix @ np.array([0, 1.25 * wheel_width])
        plt.arrow(
            position_x, position_y,
            direction[0], direction[1],
            color='r',
            head_width=0.05)
