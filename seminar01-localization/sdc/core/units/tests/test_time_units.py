from sdc.core.units.time import (
    milliseconds_to_seconds,
    nanoseconds_to_seconds,
    seconds_to_milliseconds,
    seconds_to_nanoseconds,
)


def test_conversions_between_seconds_and_milliseconds():
    # Seconds to milliseconds
    milliseconds = seconds_to_milliseconds(123)
    assert isinstance(milliseconds, int)
    assert milliseconds == 123000
    del milliseconds

    milliseconds = seconds_to_milliseconds(123.0)
    assert isinstance(milliseconds, float)
    assert milliseconds == 123000.0
    del milliseconds

    milliseconds = seconds_to_milliseconds(123.25)
    assert isinstance(milliseconds, float)
    assert milliseconds == 123250.0
    del milliseconds

    # Milliseconds to seconds
    seconds = milliseconds_to_seconds(123000)
    assert isinstance(seconds, float)
    assert seconds == 123.0
    del seconds

    seconds = milliseconds_to_seconds(123000.0)
    assert isinstance(seconds, float)
    assert seconds == 123.0
    del seconds

    seconds = milliseconds_to_seconds(123250.0)
    assert isinstance(seconds, float)
    assert seconds == 123.25
    del seconds


def test_conversions_between_seconds_and_nanoseconds():
    # Seconds to nanoseconds
    nanoseconds = seconds_to_nanoseconds(321)
    assert isinstance(nanoseconds, int)
    assert nanoseconds == 321000000000
    del nanoseconds

    nanoseconds = seconds_to_nanoseconds(321.0)
    assert isinstance(nanoseconds, int)
    assert nanoseconds == 321000000000
    del nanoseconds

    nanoseconds = seconds_to_nanoseconds(321.25)
    assert isinstance(nanoseconds, int)
    assert nanoseconds == 321250000000
    del nanoseconds

    # Nanoseconds to seconds
    seconds = nanoseconds_to_seconds(321000000000)
    assert isinstance(seconds, float)
    assert seconds == 321.0
    del seconds

    seconds = nanoseconds_to_seconds(321250000000)
    assert isinstance(seconds, float)
    assert seconds == 321.25
    del seconds
