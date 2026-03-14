import abc
from sdc.core.timestamp import Timestamp


class SimulationComponent(abc.ABC):
    def __init__(self):
        # Time must be initialized before starting simulation process
        self._time = None

    @property
    def time(self):
        return self._time

    def set_time(self, time):
        assert isinstance(time, Timestamp)
        assert self._time is None, "Time is already initialized"
        self._set_time_impl(time)
        assert self._time == time, (
            f'Unexpected final time for component "{type(self)}": {self._time} != {time}'
        )

    def move_by(self, dt: Timestamp):
        time = self._time + dt
        assert isinstance(dt, Timestamp)
        result = self._move_by_impl(dt)
        assert self._time == time, (
            f'Unexpected final time for component "{type(self)}": {self._time} != {time}'
        )
        return result

    def move_to(self, time: Timestamp):
        assert isinstance(time, Timestamp)
        assert self._time < time
        result = self.move_by(time - self.time)
        assert self._time == time, (
            f'Unexpected final time for component "{type(self)}": {self._time} != {time}'
        )
        return result

    @abc.abstractmethod
    def _set_time_impl(self):
        ...

    @abc.abstractmethod
    def _move_by_impl(self):
        ...
