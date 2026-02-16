import abc
import copy
from sdc.core.timestamp import Timestamp


class MessageBase(abc.ABC):
    def __init__(self, stamp: Timestamp):
        self.stamp = copy.deepcopy(stamp)


class PipelineMessage:
    def __init__(self, topic: str, stamp: Timestamp, message: MessageBase):
        self.topic = topic
        self.stamp = copy.deepcopy(stamp)
        self.message = message


class GnssPositionMessage(MessageBase):
    def __init__(self, stamp: Timestamp, x: float, y: float):
        super().__init__(stamp)
        self.x = x
        self.y = y
