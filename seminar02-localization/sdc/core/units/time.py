import typing as T

MILLISECONDS_IN_SECOND = 1000

MICROSECONDS_IN_SECOND = 1000**2
MICROSECONDS_IN_MILLISECONDS = 1000

NANOSECONDS_IN_SECOND = 1000**3
NANOSECONDS_IN_MILLISECONDS = 1000**2
NANOSECONDS_IN_MICROSECONDS = 1000


def milliseconds_to_seconds(milliseconds: T.Union[int, float]) -> T.Union[int, float]:
    return milliseconds / MILLISECONDS_IN_SECOND


def seconds_to_milliseconds(seconds: T.Union[int, float]) -> T.Union[int, float]:
    return seconds * MILLISECONDS_IN_SECOND


def microseconds_to_seconds(microseconds: T.Union[int, float]) -> T.Union[int, float]:
    return microseconds / MICROSECONDS_IN_SECOND


def seconds_to_microseconds(seconds: T.Union[int, float]) -> T.Union[int, float]:
    return seconds * MICROSECONDS_IN_SECOND


def nanoseconds_to_seconds(nanoseconds: T.Union[int, float]) -> float:
    return float(nanoseconds) / NANOSECONDS_IN_SECOND


def nanoseconds_to_seconds_maybe(
    nanoseconds: T.Optional[T.Union[int, float]],
) -> T.Optional[float]:
    if nanoseconds is not None:
        return nanoseconds_to_seconds(nanoseconds)


def seconds_to_nanoseconds(seconds: T.Union[int, float]) -> int:
    return int(round(seconds * NANOSECONDS_IN_SECOND))


def microseconds_to_nanoseconds(
    microseconds: T.Union[int, float],
) -> T.Union[int, float]:
    return microseconds * NANOSECONDS_IN_MICROSECONDS


def days_to_milliseconds(days: T.Union[int, float]) -> T.Union[int, float]:
    return days * 24 * 3600 * 1000
