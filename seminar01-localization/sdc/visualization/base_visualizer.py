import abc


class VisualizerBase(abc.ABC):
    @abc.abstractmethod
    def draw(self, ax):
        pass
