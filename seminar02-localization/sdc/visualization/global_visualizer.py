from .base_visualizer import VisualizerBase


class GlobalVisualizer(VisualizerBase):
    def __init__(self):
        self._visualizers = []

    def add_visualizer(self, visualizer: VisualizerBase):
        self._visualizers.append(visualizer)

    def draw(self, ax):
        for visualizer in self._visualizers:
            visualizer.draw(ax)
