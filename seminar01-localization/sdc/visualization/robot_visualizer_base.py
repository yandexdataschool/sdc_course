import abc


class RobotVisualizerBase(abc.ABC):
    @abc.abstractmethod
    def draw(self, ax):
        ...
