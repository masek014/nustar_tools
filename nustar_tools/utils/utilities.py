import re
import os
import pathlib

import astropy.units as u
import numpy as np
import nustar_pysolar as nustar
import parse

from astropy.io import fits
from astropy.time import Time


CONF_FILE = os.path.dirname(os.path.realpath(__file__)) + '/default_conf.ini'
DATA_DIR_FORMAT = '{working_dir}data/'
ID_DIR_PATH_FORMAT = DATA_DIR_FORMAT + '{date:8}/{id_num:11}/'
PARAMETERS_DIR_PATH_FORMAT = '{id_dir}products/{parameters_str}/'
OUTPUTS_DIR_PATH_FORMAT = PARAMETERS_DIR_PATH_FORMAT + 'FPM{fpm}/'
FIGURES_DIR_PATH_FORMAT = '{id_dir}figures/'
EVT_FILE_PATH_FORMAT = '{id_dir}event_cl/nu{id_num:11}{fpm}06_cl_sunpos.evt'
HK_FILE_PATH_FORMAT = '{id_dir}hk/nu{id_num:11}{fpm}_fpm.hk'
PARAMETERS_DIR_STR_FORMAT = '{bin_size}b_{frame_length}f_{bcwidth}bc_{emin}-{emax}kev_{dthresh}d{cthresh}c'


class BoxcarTooLargeException(Exception):
    pass


def execute_command(cmd: str, quiet: bool = True):
    '''Executes the provided OS command.

    Parameters
    ----------
    cmd : str
        The command to be executed.
    quiet : bool
        Specify whether the output from the command should be silenced.
        True specifies that no output from the executed command
        will be seen.

    Returns
    -------
    None
    '''
    if quiet:
        cmd += ' >/dev/null 2>&1'
    os.system(cmd)


def find_nth(haystack: str, needle: str, nth: int) -> int:
    '''Finds the index of the nth occurrence of a substring.
    Method found here: https://stackoverflow.com/a/1884277

    Parameters
    ----------
    haystack : str
        The input string.
    needle : str
        The desired substring.
    nth : int
        The nth occurrence of the substring.

    Returns
    -------
    start : int
        The index of the nth occurrence.
    '''
    start = haystack.find(needle)
    while start >= 0 and nth > 1:
        start = haystack.find(needle, start+len(needle))
        nth -= 1

    return start


def natural_sort(l: list) -> list:
    '''Sort the provided by natural, human standards as opposed to machine standards.
    Method found here: https://stackoverflow.com/a/4836734
    '''
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [
        convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(l, key=alphanum_key)


def get_evtfile_from_hkfile(hk_file: str) -> str:
    '''Get the path to the event file corresponidng to the given hk file.'''
    p = parse.parse(HK_FILE_PATH_FORMAT, hk_file)
    evt_file = EVT_FILE_PATH_FORMAT.format(**p.named)

    return evt_file


def get_hkfile_from_evtfile(evt_file: str) -> str:
    '''Get the path to the hk file corresponding to the given event file.'''
    p = parse.parse(EVT_FILE_PATH_FORMAT, evt_file)
    hk_file = HK_FILE_PATH_FORMAT.format(**p.named)

    return hk_file


def characterize_frames(evt_data: fits.FITS_rec, frame_length: float):
    '''Determines the number of frames such that each bin is exactly
    the width that is specified by frame_length. The method will
    choose a new end time that is **before** the original end
    time such that an exact integer number of frames can fit
    within the start and (new) end time.

    Parameters
    ----------
    evt_data : FITS record array
        A FITS record array containing the data from evt_file.
    frame_length : float
        Length of a single frame, in seconds

    Returns
    -------
    start_time : float
        The start time of the observation in NuSTAR time.
    end_time : float
        The new end time of the observation in NuSTAR time.
    num_frames : int
        The number of frames in the observation.
    '''
    start_time = evt_data[0]['TIME']
    end_time = evt_data[-1]['TIME']
    end_time = end_time - ((end_time-start_time) % frame_length)
    num_frames = int((end_time - start_time)/frame_length)

    return start_time, end_time, num_frames


def get_nominal_coordinate(
    hdr: fits.Header,
    time: Time | None = None
) -> tuple[u.Quantity, u.Quantity]:
    '''Retrieves the nominal coordinate from the given header
    and corrects it if a time is given. The coordinate is
    converted from RA and Dec into solar coordinates in
    arcseconds.

    Parameters
    ----------
    hdr : FITS header
        The header containing the coordinate.
    time : Astropy time
        The time at which the correction is performed. Defaults
        to the MJD-OBS card in the header if None.

    Returns
    -------
    x : Astropy Quantity
        The x-coordinate of the nominal position in units
        of arcseconds.
    y : Astropy Quantity
        The y-coordinate of the nominal position in units
        of arcseconds.
    '''
    if time is None:
        time = [Time(hdr['MJD-OBS'], format='mjd')]
    elif not isinstance(time, list):
        time = [time]

    ra, dec = [hdr['RA_NOM']] * u.deg, [hdr['DEC_NOM']] * u.deg
    x, y = nustar.convert._delta_solar_skyfield(
        ra, dec, time)  # Units of degrees
    x, y = x.to(u.arcsecond), y.to(u.arcsecond)

    return x, y


def filter_far_data(evt_data: fits.FITS_rec, hdr: fits.Header) -> fits.FITS_rec:
    '''Removes the data that is unphysically far from the FOV.
    It removes all data that is outside of a 2000x2000 arcsecond
    square defined with the nominal coordinate as the center.
    The nominal coordinate is relatively close to the FOV center.

    Parameters
    ----------
    evt_data : FITS record
        The event data to be filtered.
    hdr : FITS header
        The header corresponding to evt_data.

    Returns
    -------
    filtered_data : FITS record
        The filtered data.
    '''
    # print(evt_data['X'])
    # x, y = get_nominal_coordinate(hdr)
    x, y = np.mean(evt_data['X']), np.mean(evt_data['Y'])

    for field in list(hdr.keys()):
        if field.find('TYPE') != -1:
            if hdr[field] == 'X':
                xval = field[5:8]
            if hdr[field] == 'Y':
                yval = field[5:8]
    npixx = hdr[f'TLMAX{xval}']
    npixy = hdr[f'TLMAX{yval}']
    pixsizex = np.abs(hdr[f'TCDLT{xval}'])
    pixsizey = np.abs(hdr[f'TCDLT{yval}'])

    x = (x-npixx*0.5) * pixsizex * u.arcsec
    y = (y-npixy*0.5) * pixsizey * u.arcsec
    delta = 1000 * u.arcsec
    xmin, xmax = x - delta, x + delta
    ymin, ymax = y - delta, y + delta

    # Be sure to use the native conversion unit in the header.
    unit = hdr['TCUNI14']
    min_max = [v.to(unit).value for v in [xmin, xmax, ymin, ymax]]
    filtered_data = evt_data[nustar.filter.by_xy(evt_data, hdr, min_max)]

    return filtered_data


def get_event_data(
    evt_file: str,
    time_interval: list[str, str] | None = None,
    perform_filter: bool = True
) -> tuple[fits.FITS_rec, fits.Header]:
    '''Obtains the data and header in a NuSTAR event file.

    Parameters
    ----------
    evt_file : str
        Name of the input file.
    time_interval : list or tuple of strings
        Specifies the start and end time desired.
        Has the form [start_time, end_time] with the
        time strings in iso format: 'YYYY-MM-DD HH:MM:SS'
    perform_filter : bool
        Specifies whether the unphysically far data should
        be filtered from the event list.

    Returns
    -------
    evt_data : FITS record array
        A FITS record array containing the data from evt_file.
    hdr : FITS header
        A FITS header corresponding to evt_data.
    '''
    with fits.open(evt_file) as hdu:
        evt_data = hdu[1].data
        hdr = hdu[1].header
    if time_interval is not None:
        time = Time(time_interval, format='iso', scale='utc')
        evt_data = evt_data[nustar.filter.by_time(evt_data, hdr, time)]
    if perform_filter:
        evt_data = filter_far_data(evt_data, hdr)

    return evt_data, hdr


def moving_average(a: np.ndarray, n: int) -> np.ndarray:
    '''Computes the moving average with width n across the array a.
    Method found here: https://stackoverflow.com/a/14314054

    Parameters
    ----------
    a : np.ndarray
        The array containing the values.
    n : int
        The boxcar width.

    Returns
    -------
    ret : np.ndarray
        Array containing the boxcar averaged values.
        It has length len(a)-ceil(n/2)
    '''
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]

    return ret[n - 1:] / n


def clean_directory_string(dir_path: str) -> str:
    '''Modifies the input path string to personal preferences.

    Parameters
    ----------
    dir_path : str
        The directory path to be cleaned.

    Returns
    -------
    dir_path : str
        The cleaned directory path.
    '''
    if dir_path[-1] != '/':
        dir_path = dir_path + '/'

    return dir_path


def create_directories(dirs: str | list[str]):
    '''Checks if the provided directory exists.
    Creates the directory if it does not exist.
    A list can be provided to create several directories.

    Parameters
    ----------
    dirs : str or list
        The absolute directory path to be checked,
        or a list of directories to create.
    '''
    if not isinstance(dirs, list):
        dirs = [dirs]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def verify_path(
    in_path: str,
    is_dir: bool = True,
    make_abs_dir: bool = True
) -> str:
    '''Verifies whether the provided path or directory is valid or not.
    Adds a '/' to the end of the string if one is not already there.
    Checks if dir is an absolute directory or not. If it is not,
    it is made into one.

    Parameters
    ----------
    in_path : str
        Input path to be checked.
    is_dir : bool
        Specifies whether in_path is a directory.
    make_abs_dir : bool
        Specifies whether or not the make in_path
        an absolute directory.

    Returns
    -------
    in_path : str
        Outputs the input string with some slight modifications.
        If the directory is not good, an exception is raised.
    '''

    if is_dir:
        in_path = clean_directory_string(in_path)  # Clean up the string
    # Check if the directory exists
    if not os.path.exists(in_path):
        raise FileNotFoundError(f'\'{in_path}\' is not a valid path.')
    elif make_abs_dir:
        # Check if dir is an absolute directory. It is isn't, make it absolute.
        in_path = os.path.abspath(in_path) + '/'

    return in_path


def get_gti_file_from_evt_file(evt_file: str) -> str:
    return evt_file.replace('_cl_sunpos.evt', '_gti.fits')


def get_id_dir_from_evt_file(in_file: str) -> str:
    '''Returns the ID directory from the event file.

    Ex: get_figure_path('/Users/rbmasek/nustar/data/20170321/20210015001/event_cl/nu20210015001A06_cl_sunpos.evt')
    Returns '/Users/rbmasek/nustar/data/20170321/20210015001/'
    '''
    return in_file.split('event_cl')[0]


def get_id_dir_from_hk_file(in_file: str) -> str:
    '''Returns the ID directory from the event file.

    Ex: get_figure_path('/Users/rbmasek/nustar/data/20170321/20210015001/event_cl/nu20210015001A06_cl_sunpos.evt')
    Returns '/Users/rbmasek/nustar/data/20170321/20210015001/'
    '''

    return in_file.split('hk')[0]


def get_id_from_id_dir(id_dir: str) -> str:
    '''Returns a string containing the data ID
    from the provided ID directory.
    '''
    # return parse.parse(ID_DIR_PATH_FORMAT, id_dir)['id_num']
    path = os.path.normpath(id_dir)
    return path.split(os.sep)[-1]


def get_date_from_id_dir(id_dir: str) -> str:
    '''Returns a string containing the observation
    date from the provided ID directory.
    '''
    path = os.path.normpath(id_dir)
    return path.split(os.sep)[-2]
    # return parse.parse(ID_DIR_PATH_FORMAT, id_dir)['date']


def get_fpm_from_filename(file_name: str) -> str:
    '''Returns the FPM of the provided file name.'''
    fpm = 'A'
    if 'B06' in file_name or 'B_fpm' in file_name:
        fpm = 'B'

    return fpm


def convert_pi_to_energy(pi: float | np.ndarray) -> float | np.ndarray:
    '''Converts pulse invariant from channel
    space to energy in units of keV.
    '''
    return (0.040 * pi) + 1.6


def gunzip_file(gz_path: str) -> str:
    '''Gunzips the provided .gz file and keeps the original .gz file.
    The unzipped file is in the same directory as the original .gz file.
    The full path must be provided (not just the file name).
    It will only unzip the file if it has not yet been unzipped;
    this method does nothing if the unzipped file already exists.
    Ex: gunzip_file('/Users/rbmasek/nustar/data/20170321/20210015001/hk/nu20210015001A_fpm.hk.gz')

    Parameters
    ----------
    gz_path : str
        The absolute file path to the .gz file.

    Returns
    -------
    unzipped_file_path : str
        The absolute file path to the unzipped file (with no '/' at the end).
    '''

    unzipped_file_path = gz_path[:gz_path.find('.gz')]
    if not os.path.isfile(unzipped_file_path):
        if gz_path[-1] == '/':
            gz_path = gz_path[:-1]
        os.system('gunzip -c ' + gz_path + ' > ' + unzipped_file_path)

    return unzipped_file_path
