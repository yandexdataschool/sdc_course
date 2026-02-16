import abc
import copy
import numpy as np
import typing as T
from sdc.core.timestamp import Timestamp
from sdc.sim.component import SimulationComponent
from sdc.sim.pipeline import Pipeline
from sdc.movement_models.base import MovementModelBase
from sdc.sensors.base import SensorBase


class RobotStateBase(abc.ABC):
    def __init__(self):
        pass

    @property
    @abc.abstractmethod
    def size(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def array(self) -> np.ndarray:
        ...


class RobotBase(SimulationComponent):
    def __init__(self):
        super().__init__()

        self._state = None
        self._movement_model = None
        self._pipeline = None

        # Robot has sensors uniquely identified by their string IDs
        self._sensor_by_id: T.Dict[str, SensorBase] = dict()
        self._sensor_pose_by_id: T.Dict[str, np.ndarray] = dict()

    @property
    @abc.abstractmethod
    def state(self) -> RobotStateBase:
        return ...

    @property
    def movement_model(self) -> T.Optional[MovementModelBase]:
        return self._movement_model

    @movement_model.setter
    def movement_model(self, movement_model: T.Optional[MovementModelBase]):
        # Attaching movement plan/trajectory to simulated robot
        if movement_model is not None:
            assert isinstance(movement_model, MovementModelBase)
            self._movement_model = movement_model
            movement_model._initialize(self)
        else:
            self._movement_model = None

    @property
    def pipeline(self) -> T.Optional[Pipeline]:
        return self._pipeline

    @pipeline.setter
    def pipeline(self, pipeline: T.Optional[Pipeline]):
        self._pipeline = pipeline

    def add_sensor(self, sensor: SensorBase, sensor_pose: np.ndarray):
        assert isinstance(sensor, SensorBase)
        assert sensor.id not in self._sensor_by_id
        self._sensor_by_id[sensor.id] = sensor
        self._sensor_pose_by_id[sensor.id] = sensor_pose
        sensor._mount(self)

    def get_sensor(self, sensor_id: str) -> SensorBase:
        return self._sensor_by_id[sensor_id]

    def get_sensor_pose(self, sensor_id: str) -> np.ndarray:
        return self._sensor_pose_by_id[sensor_id]

    def get_sensors_by_type(self, sensor_type: type) -> T.List[SensorBase]:
        sensors = []
        for sensor_id, sensor in self._sensors_by_id.items():
            if isinstance(sensor, sensor_type):
                sensors.append(sensor)
        return sensors

    @property
    def time(self):
        return self._time

    def _set_time_impl(self, time: Timestamp):
        if self._movement_model is not None:
            self._movement_model.set_time(time)

        for sensor in self._sensor_by_id.values():
            sensor.set_time(time)

        if self._pipeline is not None:
            self._pipeline.set_time(time)

        self._time = copy.deepcopy(time)

    def _move_by_impl(self, dt: Timestamp):
        # Moving robot in simulated scene
        if self._movement_model is not None:
            self._movement_model._move_by(dt)

        # Generating sensors measurements
        for sensor in self._sensor_by_id.values():
            for message in sensor.move_by(dt):
                if self._pipeline is not None:
                    self._pipeline.inject_message(message)

        # Running internal pipeline
        if self._pipeline is not None:
            self._pipeline.move_by(dt)

        self._time += dt
