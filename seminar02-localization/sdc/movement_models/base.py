import copy
from sdc.core.timestamp import Timestamp
from sdc.sim.component import SimulationComponent


class MovementModelBase(SimulationComponent):
    """Отвечает за движение автомобиля. Представляет метод _move, который вызывается для продвижения
    автомобиля далее вдоль траектории на запрошенный шаг времени dt. Модель движения имеет прямой
    доступ к скрытому состоянию автомобиля для его корректного изменения

    Калмановская локализация:
        Реализует модель эволюции.
        Предоставляет интерфейс для получения матрицы перехода и шума в текущий момент времени.
    """

    def __init__(self):
        super().__init__()
        self._robot = None

    def _attach(self, robot):
        """This method is called automatically when trajecotry/movement model is attached to robot"""
        self._robot = robot

    def _set_time_impl(self, time: Timestamp):
        self._time = copy.deepcopy(time)
