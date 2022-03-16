# -*- coding: utf-8 -*-


class Timestamp(object):
    """
    Special class designed to represent time in the model.
    Has too private members "_sec" and "_nsec" accessed via properties.
    """
    NANO_SEC_COEFF = 1000000000  # Number of nanoseconds in one second
    MICRO_SEC_COEFF = 1000000    # Number of microseconds in one second
    MILLI_SEC_COEFF = 1000        # Number of milliseconds in one second

    def __init__(self, sec=0, nsec=0):
        assert int(sec) == sec, 'sec must be an integer value'
        assert int(nsec) == nsec, 'nsec must be an integer value'
        self.sec = sec
        self.nsec = nsec

    #########################################
    #   Static methods for construction     #
    #########################################
    @staticmethod
    def nanoseconds(nsec):
        nsec = int(nsec)
        sec = nsec // Timestamp.NANO_SEC_COEFF
        nsec -= sec * Timestamp.NANO_SEC_COEFF
        return Timestamp(sec, nsec)

    @staticmethod
    def microseconds(mcs):
        nsec = int(mcs * (Timestamp.NANO_SEC_COEFF / Timestamp.MICRO_SEC_COEFF))
        return Timestamp.nanoseconds(nsec)

    @staticmethod
    def milliseconds(ms):
        nsec = int(ms * (Timestamp.NANO_SEC_COEFF / Timestamp.MILLI_SEC_COEFF))
        return Timestamp.nanoseconds(nsec)

    @staticmethod
    def seconds(sec):
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
    def __add__(self, rhs):
        """a + b"""
        return Timestamp.nanoseconds(self.to_nanoseconds() + rhs.to_nanoseconds())

    def __sub__(self, rhs):
        """a - b"""
        return Timestamp.nanoseconds(self.to_nanoseconds() - rhs.to_nanoseconds())

    def __iadd__(self, rhs):
        """a += b"""
        nsec = self.to_nanoseconds() + rhs.to_nanoseconds()
        self.sec = nsec // self.NANO_SEC_COEFF
        self.nsec = nsec - self.sec * self.NANO_SEC_COEFF
        return self

    #########################################
    #      Свойства                         #
    #########################################
    @property
    def sec(self):
        return self._sec

    @property
    def nsec(self):
        return self._nsec

    @sec.setter
    def sec(self, sec):
        assert sec == int(sec)
        assert sec >= 0
        self._sec = int(sec)

    @nsec.setter
    def nsec(self, nsec):
        assert nsec == int(nsec)
        assert nsec >= 0
        assert nsec < self.NANO_SEC_COEFF
        self._nsec = int(nsec)

    def to_seconds(self):
        """
        :rtype float:
        :returns: The number of seconds passed from the zero moment
        """
        return self.sec + self.nsec / float(self.NANO_SEC_COEFF)

    def to_milliseconds(self):
        """
        :rtype float:
        :returns: The number of milliseconds passed from the zero moment
        """
        return self.to_seconds() * self.MILLI_SEC_COEFF

    def to_microseconds(self):
        """
        :rtype float:
        :returns: The number of microseconds passed from the zero moment
        """
        return self.to_seconds() * self.MICRO_SEC_COEFF

    def to_nanoseconds(self):
        """
        :rtype int:
        :returns: The number of nanoseconds passed from the zero moment.
        """
        return self.to_seconds() * self.NANO_SEC_COEFF

    def __str__(self):
        return 'Time(sec={},nsec={})'.format(self.sec, self.nsec)


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
