import abc
import numpy as np
from .timestamp import Timestamp


class CarSensorBase(abc.ABC):
    """У сенсора есть реальный уровень шума, который он добавляет в наблюдения. Кроме того,
    есть уровень шума который предполагается нами и используется в фильтре Калмана.
    Сенсор запоминает последний момент времени выдачи показания. И если вдруг показание запрошено в тот же
    момент модельного времени еще раз, то возвращается то же самое показания (Логично? Логично!)

    Для создания сенсора на основе класса CarSensorBase достаточно создать новый класс
    по следующему шаблону (примеры: CarSensor, ImuSensor, GpsSensor):

        class NewSensor(CarSensorBase):
            def __init__(self, ...):
                super(NewSensor, self).__init__()
                ...

        @property
        def observation_size(self):
            return N

        def _observe_clear(self):
            return np.array(...)
    """
    def __init__(self, noise_variances=None, random_state=None):
        # Даешь каждому сенсору свой генератор!
        self._gen = np.random.RandomState(random_state)
        # Устанавливем реальный уровень шума
        if noise_variances is None:
            self._noise_variances = np.zeros(self.observation_size, dtype=np.float64)
        else:
            self._noise_variances = np.array(noise_variances)
            assert self._noise_variances.shape == (self.observation_size,)
        self._car = None
        self._last_time = None
        self._last_observation = None
        # Сенсоры хранят историю своих показаний
        self._history = []

    def _initialize(self, car):
        """Вызывается в момент добавления сенсора в машину"""
        self._car = car

    @property
    def state_size(self):
        return self._car._state_size

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
        elif self._last_time == self._car.time:
            # Запрошено наблюдение в тот же момент времени
            return self._last_observation
        observation = self._observe_clear()
        assert observation.shape == (self.observation_size,)

        for i, variance in enumerate(self._noise_variances):
            if variance > 0:
                observation[i] += self._gen.normal(scale=np.sqrt(variance))
        self._last_observation = observation
        self._last_time = Timestamp.nanoseconds(self._car.time.to_nanoseconds())
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
