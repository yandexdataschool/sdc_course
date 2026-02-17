import abc
import copy
import numpy as np
import typing as T
from sdc.msgs import PipelineMessage, MessageBase
from sdc.sim.component import SimulationComponent
from sdc.core.timestamp import Timestamp


class SensorBase(SimulationComponent):
    """У сенсора есть реальный уровень шума, который он добавляет в наблюдения. Кроме того,
    есть уровень шума который предполагается нами и используется в фильтре Калмана.
    Сенсор запоминает последний момент времени выдачи показания. И если вдруг показание запрошено в тот же
    момент модельного времени еще раз, то возвращается то же самое показания (Логично? Логично!)

    Для создания сенсора на основе класса SensorBase достаточно создать новый класс
    по следующему шаблону (примеры: WheelOdometrySensor, ImuSensor, GnssSensor):

        class NewSensor(SensorBase):
            def __init__(self, ...):
                super(NewSensor, self).__init__()
                ...

        @property
        def observation_size(self):
            return N

        def _observe_clear(self):
            return np.array(...)
    """
    def __init__(self, sensor_id: str, frequency: float, topic: str, start_offset:
                 T.Optional[Timestamp] = None, noise_variances=None, random_state=None):
        super().__init__()

        self._sensor_id = sensor_id
        self._period = Timestamp.from_seconds(1.0 / frequency)
        if start_offset is None:
            start_offset = Timestamp(0, 0)
        self._start_offset = copy.deepcopy(start_offset)
        self._topic = topic
        self._next_message_time = None

        # Даешь каждому сенсору свой генератор!
        self._gen = np.random.RandomState(random_state)
        # Устанавливем реальный уровень шума
        if noise_variances is None:
            self._noise_variances = np.zeros(self.observation_size, dtype=np.float64)
        else:
            self._noise_variances = np.array(noise_variances)
            assert self._noise_variances.shape == (self.observation_size,)
        self._robot = None
        self._last_time = None
        self._last_observation = None
        # Сенсоры хранят историю своих показаний
        self._history = []

    @property
    def id(self) -> str:
        return self._sensor_id

    def _mount(self, robot):
        """This method is called automatically when sensor is added to robot"""
        assert self._robot is None
        self._robot = robot

    @property
    def mounted(self) -> bool:
        return self._robot is not None

    @property
    def robot(self):
        return self._robot

    @property
    def state_size(self):
        return self._robot._state_size

    def get_noise_covariance(self):
        """Диагональная матрица ковариации с истинными значениями шума"""
        return np.diag(self._noise_variances)

    def observe(self):
        """Возвращает значение наблюдения для рассматриваемого автомобиля.
        Если наблюдение формально запрошено несколько раз в один и тот же момент времени,
        то возвращает один и тот же результат."""
        if self._last_time is None:
            # Не было ни одного наблюдения
            pass
        elif self._last_time == self.time:
            # Запрошено наблюдение в тот же момент времени
            return self._last_observation
        observation = self._observe_clear()
        assert observation.shape == (self.observation_size,)

        for i, variance in enumerate(self._noise_variances):
            if variance > 0:
                observation[i] += self._gen.normal(scale=np.sqrt(variance))
        self._last_observation = observation
        self._last_time = Timestamp.from_nanoseconds(self.time.to_nanoseconds())
        observation = np.array(self._last_observation)
        self._history.append(observation)
        return observation

    @property
    def history(self):
        return np.array(self._history)

    #########################################
    #      Методы для переопределения       #
    #########################################
    @property
    @abc.abstractmethod
    def observation_size(self) -> int:
        """Возвращает размер наблюдения"""
        ...

    @abc.abstractmethod
    def _observe_clear(self):
        """Возвращает незашумленное значение наблюдения."""
        ...

    @abc.abstractmethod
    def _generate_message(self) -> MessageBase:
        ...

    def _generate_pipeline_message(self) -> PipelineMessage:
        return PipelineMessage(self._topic, self._time, self._generate_message())

    def _set_time_impl(self, time: Timestamp):
        self._time = copy.deepcopy(time)
        self._next_message_time = self._time + self._start_offset

    def _move_by_impl(self, dt: Timestamp) -> T.List[MessageBase]:
        finish_time = self._time + dt
        messages = []

        if self._next_message_time > finish_time:
            # No messages generated during interval [self._time, self._time + dt]
            self._time = finish_time
            return messages

        while self._next_message_time <= finish_time:
            self._time = copy.deepcopy(self._next_message_time)
            messages.append(self._generate_pipeline_message())
            self._next_message_time += self._period
        self._time = finish_time
        return messages
