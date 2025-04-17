'''
A collection of tools for manipulating time strings and converting
to/from NuSTAR's reference time (January 01, 2010 00:00:00 UTC)
'''

import datetime

from astropy.time import Time, TimeDelta


def get_reference_time() -> Time:
    '''Returns the relative time (in ISO format) since January 1, 2010
    since that is the time NuSTAR references.
    '''
    return Time('2010-01-01 00:00:00', format='iso', scale='utc')


def nustar_to_datetime(time: float) -> datetime.datetime:
    '''Converts NuSTAR timestamps (in seconds since 2010-01-01)
    to datetime objects.

    Parameters
    ----------
    time : float
        Number of seconds since 2010-01-01 00:00:00 UTC.

    Returns
    -------
    Datetime object corresponding to the input time.
    '''
    ref = (get_reference_time().datetime).replace(
        tzinfo=datetime.timezone.utc)

    return ref + datetime.timedelta(seconds=time)


def nustar_to_astropy(time: float) -> Time:
    '''Convert a NuSTAR time (seconds since 2010-01-01 00:00:00 UTC) to
    an Astropy Time object.'''
    return Time(nustar_to_datetime(time), format='datetime', scale='utc')


def string_to_nustar(time: str) -> float:
    '''Convert a time string to NuSTAR time,
    i.e. seconds since 2010-01-01 00:00:00 UTC.
    **MUST** be done in datetime instead of Astropy Time...
    Otherwise, some error is confounded with the result.
    '''
    rel_t = get_reference_time().datetime.replace(tzinfo=datetime.timezone.utc)
    time = datetime.datetime.strptime(
        time, '%Y-%m-%d %H:%M:%S').replace(tzinfo=datetime.timezone.utc)

    return (time - rel_t).total_seconds()


def add_timedelta_to_string(time_str: str, time_delta: float):
    '''Add (or subtract) the provided number of seconds to the provided string.

    Parameters
    ----------
    time_str : str
        Time string accepted by Astropy.time.Time.
    td : int or float
        Timedelta to be added to s, in seconds.

    Returns
    -------
    Formatted datetime string with the timedelta added.
    '''
    time = Time(time_str, scale='utc')
    td = TimeDelta(time_delta, format='sec')

    return (time + td).strftime('%Y-%m-%d %H:%M:%S')
