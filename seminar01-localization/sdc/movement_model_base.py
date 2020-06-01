# -*- coding: utf-8 -*-


class MovementModelBase(object):
    """
    Represents car movement.

    One should call method _move to move car along a trajectory during time 'dt'. Class has a
    direct access to car state and updates it after each move.
    """

    def __init__(self):
        self._car = None

    def _initialize(self, car):
        """
        Method is called when the movement model is added to a car.
        Ties model instance and car instance.
        """
        self._car = car

    @property
    def state_size(self):
        return self._car._state_size

    def _move(self,  dt):
        """
        Moves the car along its trajectory during time 'dt'. Changes car timestamp.
        Trajectory can be represented as motion equations.
        """
        assert False
