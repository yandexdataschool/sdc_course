from sdc.sim.sensors.base import SensorBase
from sdc.sim.sensors.utils import get_sensor_global_position


class SensorVisualizer:
    def __init__(self, sensor_base: SensorBase, marker_size=20, marker_color='b'):
        self._sensor = sensor_base
        self._marker_size = marker_size
        self._marker_color = marker_color

    def draw(self, ax):
        sensor_global_position = get_sensor_global_position(self._sensor)
        ax.scatter(
            sensor_global_position[0],
            sensor_global_position[1],
            s=self._marker_size,
            color=self._marker_color,
            zorder=1, label=self._sensor.id)
