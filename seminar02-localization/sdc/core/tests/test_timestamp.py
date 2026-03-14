from sdc.core.timestamp import Timestamp


def test_timestamp():
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
