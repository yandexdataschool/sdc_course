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


class WheelOdometryMessage(MessageBase):
    def __init__(self, stamp: Timestamp, linear_velocity: float):
        super().__init__(stamp)
        self.linear_velocity = linear_velocity


class ImuMessage(MessageBase):
    def __init__(self, stamp: Timestamp, angular_velocity: float):
        super().__init__(stamp)
        self.angular_velocity = angular_velocity
