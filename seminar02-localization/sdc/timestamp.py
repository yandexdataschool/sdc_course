from __future__ import annotations


class Timestamp:
    """
    Special class designed to represent time in the model.
    Has too private members "_sec" and "_nsec" accessed via properties.
    """
    NANO_SEC_COEFF = 1000000000  # Number of nanoseconds in one second
    MICRO_SEC_COEFF = 1000000    # Number of microseconds in one second
    MILLI_SEC_COEFF = 1000       # Number of milliseconds in one second

    def __init__(self, sec: int = 0, nsec: int = 0):
        self.sec = sec
        self.nsec = nsec

    #########################################
    #   Static methods for construction     #
    #########################################
    @staticmethod
    def nanoseconds(nsec: int) -> Timestamp:
        assert isinstance(nsec, int)
        assert nsec >= 0
        sec = nsec // Timestamp.NANO_SEC_COEFF
        nsec -= sec * Timestamp.NANO_SEC_COEFF
        return Timestamp(sec, nsec)

    @staticmethod
    def microseconds(mcs: int) -> Timestamp:
        nsec = int(mcs * (Timestamp.NANO_SEC_COEFF / Timestamp.MICRO_SEC_COEFF))
        return Timestamp.nanoseconds(nsec)

    @staticmethod
    def milliseconds(ms: int) -> Timestamp:
        nsec = int(ms * (Timestamp.NANO_SEC_COEFF / Timestamp.MILLI_SEC_COEFF))
        return Timestamp.nanoseconds(nsec)

    @staticmethod
    def seconds(sec: int) -> Timestamp:
        nsec = int(sec * Timestamp.NANO_SEC_COEFF)
        return Timestamp.nanoseconds(nsec)

    #########################################
    #      Assignments operators            #
    #########################################
    def __le__(self, rhs):
        if self.sec <= rhs.sec:
            return True
        if self.sec == rhs.sec:
            return self.nsec <= rhs.nsec
        return False

    def __lt__(self, rhs):
        if self.sec < rhs.sec:
            return True
        if self.sec == rhs.sec:
            return self.nsec < rhs.nsec
        return False

    def __ge__(self, rhs):
        return not self < rhs

    def __gt__(self, rhs):
        return not self <= rhs

    def __eq__(self, rhs):
        return self.sec == rhs.sec and self.nsec == rhs.nsec

    #########################################
    #      Арифметрические операторы        #
    #########################################
    def __add__(self, rhs) -> Timestamp:
        """a + b"""
        return Timestamp.nanoseconds(self.to_nanoseconds() + rhs.to_nanoseconds())

    def __sub__(self, rhs) -> Timestamp:
        """a - b"""
        return Timestamp.nanoseconds(self.to_nanoseconds() - rhs.to_nanoseconds())

    def __iadd__(self, rhs) -> Timestamp:
        """a += b"""
        nsec = self.to_nanoseconds() + rhs.to_nanoseconds()
        self.sec = nsec // self.NANO_SEC_COEFF
        self.nsec = nsec - self.sec * self.NANO_SEC_COEFF
        return self

    #########################################
    #      Свойства                         #
    #########################################
    @property
    def sec(self) -> int:
        return self._sec

    @property
    def nsec(self) -> int:
        return self._nsec

    @sec.setter
    def sec(self, sec: int):
        assert isinstance(sec, int), f'sec must be an integer value but got {type(sec)}'
        assert sec >= 0
        self._sec = sec

    @nsec.setter
    def nsec(self, nsec: int):
        assert isinstance(nsec, int), f'nsec must be an integer value but got {type(nsec)}'
        assert nsec >= 0
        assert nsec < self.NANO_SEC_COEFF
        self._nsec = nsec

    def to_seconds(self) -> float:
        """
        :rtype float:
        :returns: The number of seconds passed from the zero moment
        """
        return float(self.sec) + self.nsec / float(10**9)

    def to_milliseconds(self) -> float:
        """
        :rtype float:
        :returns: The number of milliseconds passed from the zero moment
        """
        return float(self.sec * 10**3) + self.nsec / float(10**6)

    def to_microseconds(self) -> float:
        """
        :rtype float:
        :returns: The number of microseconds passed from the zero moment
        """
        return float(self.sec * 10**6) + self.nsec / float(10**3)

    def to_nanoseconds(self) -> int:
        """
        :rtype float:
        :returns: The number of nanoseconds passed from the zero moment.
        """
        return self.sec * 10**9 + self.nsec

    def __str__(self):
        return f'Time(sec={self.sec},nsec={self.nsec})'


if __name__ != '__main__':
    # Test
    t = Timestamp(1, 19)
    assert t.sec == 1
    assert t.nsec == 19
    t.sec = 2
    t.nsec = 10000
    assert t.sec == 2
    assert t.nsec == 10000
    assert abs(t.to_seconds() - 2.00001) < 1e-9

    t1 = Timestamp(1, 1000000)
    t2 = Timestamp(2, 100000000)
    assert abs(t1.to_seconds() - 1.001) < 1e-9
    assert abs(t2.to_seconds() - 2.1) < 1e-9
    assert abs((t2 - t1).to_seconds() - 1.099) < 1e-9
    assert abs((t2 + t1).to_seconds() - 3.101) < 1e-9
    t2 += t1
    assert abs(t2.to_seconds() - 3.101) < 1e-9
