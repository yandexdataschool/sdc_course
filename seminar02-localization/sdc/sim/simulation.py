import dataclasses
from sdc.core.timestamp import Timestamp


@dataclasses.dataclass
class SimulationParameters:
    start_time: Timestamp
    finish_time: Timestamp
    time_step: Timestamp


class Simulation:
    def __init__(self, params: SimulationParameters):
        self._params = params
        self._now = None

    def add_robot(self, robot):
        self._robot = robot

    def _initialize(self):
        self._robot.set_time(self._params.start_time)
        self._now = self._params.start_time

    def run(self):
        assert self._robot is not None
        assert self._now is None
        self._initialize()
        while self._now < self._params.finish_time:
            self._robot.move_by(self._params.time_step)
            self._now += self._params.time_step
