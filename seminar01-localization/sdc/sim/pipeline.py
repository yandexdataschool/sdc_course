import copy
from sdc.msgs import PipelineMessage
from sdc.core.timestamp import Timestamp
from sdc.sim.component import SimulationComponent


class Pipeline(SimulationComponent):
    def __init__(self):
        super().__init__()
        self.messages = []

    def inject_message(self, pipeline_message: PipelineMessage):
        self.messages.append(pipeline_message)

    def _set_time_impl(self, time: Timestamp):
        self._time = copy.deepcopy(time)

    def _move_by_impl(self, dt: Timestamp):
        self._time += dt
