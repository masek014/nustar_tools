import re
import os
import os.path
import math
import parse
import pickle
import distutils
import configparser
import numpy as np

from pathlib import Path
from datetime import datetime, timezone, timedelta

import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table, vstack
import nustar_pysolar as nustar


CONF_FILE = os.path.dirname(os.path.realpath(__file__)) + '/default_conf.ini'
DATA_DIR_FORMAT = '{working_dir}data/'
ID_DIR_PATH_FORMAT = DATA_DIR_FORMAT + '{date:8}/{id_num:11}/'
PARAMETERS_DIR_PATH_FORMAT = '{id_dir}products/{parameters_str}/'
OUTPUTS_DIR_PATH_FORMAT = PARAMETERS_DIR_PATH_FORMAT + 'FPM{fpm}/'
FIGURES_DIR_PATH_FORMAT = '{id_dir}figures/'
EVT_FILE_PATH_FORMAT = '{id_dir}event_cl/nu{id_num:11}{fpm}06_cl_sunpos.evt'
HK_FILE_PATH_FORMAT = '{id_dir}hk/nu{id_num:11}{fpm}_fpm.hk'
PARAMETERS_DIR_STR_FORMAT = '{bin_size}b_{frame_length}f_{bcwidth}bc_{emin}-{emax}kev_{dthresh}d{cthresh}c'
DATE_STR_FORMAT = '%Y-%m-%d %H:%M:%S'


class BoxcarTooLargeException(Exception):
    pass


def apply_config_settings(config_dict, section, config_file=CONF_FILE):
    """
    This method reads in the settings from the configuration
    file and applies them to the specified settings dictionary.

    By default, each setting is saved as a string, so this method
    will automatically detect and convert the input values into
    the proper data types.

    Configuration management based on: https://wiki.python.org/moin/ConfigParserExamples

    Parameters
    ----------
    config_dict : dict
        A dictionary with keys and values
        specifying each setting.
    section : str
        The section in the configuration file
        from which the settings will be read.
    config_file : str
        The name of the configuration file
        from which the settings will be read.
    
    Returns
    -------
    There are no returns. The dictionary input, config_dict, is a
    reference rather than a copy, so the values of config_dict are
    directly updated within this method.
    """

    # Read the configuration file.
    config = configparser.ConfigParser()
    conf = config.read(config_file)
    options = config.options(section)

    # Iterate through each option from the configuration file and update the dictionary.
    for option in options:
        opt = option.upper() # Convert the string to uppercase to match the dictionary key
        val = config.get(section, opt) # Get the correspodning value from the configuration file
        b_found_type = False # Tracks whether the type has been found or not
        
        # Variable val is read in as a string by default.
        # Identify the proper data type of val.
        # There are only three allowed data types: float, int, bool.
        if '.' in val:
            try:
                val = float(val)
                b_found_type = True
            except Exception:
                # NOTE If the string has a '.' but cannot be converted to a float, then it is invalid.
                raise TypeError('Invalid data type for option \'' + str(opt) + \
                    '\' in section \'' + str(section) + '\' of file \'' + str(config_file) + '\'')
        else:

            if '\'' in val and val.count('\'') == 2:
                val = val.replace('\'', '')
                b_found_type = True
                
            if not b_found_type:
                try:
                    val = int(val)
                    b_found_type = True
                except ValueError:
                    pass

            if not b_found_type:
                try:
                    # Method for converting to bool: https://stackoverflow.com/a/18472142
                    val = bool(distutils.util.strtobool(val))
                    b_found_type = True
                except ValueError:
                    pass

            if not b_found_type:
                raise TypeError('Invalid data type for option \'' + str(opt) + \
                    '\' in section \'' + str(section) + '\' of file \'' + str(config_file) + '\'')

        config_dict[opt] = val


def get_config_option(section, option, config_file=CONF_FILE):
    """
    Obtain the value for a particular configuration option.

    Parameters
    ----------
    section : str
        The section containing the desired option.
    option : str
        The desired option.
    config_file : str
        The path to the configuration file.

    Returns
    -------
    The value corresponding to the specified option.
    """
    
    config = configparser.ConfigParser()
    conf = config.read(config_file)

    return config.get(section, option)


def verify_config_types(config_dict, checks):
    """
    Checks that the key values in the dictionary pass the provided checks.

    Parameters
    ----------
    config_dict : dict
        Configuration dictionary.
    checks : list of tuples
        A list where each element is a doublet of the form (key, <data_type>).
        The values with the provided key must be of type <data_type>.
        Ex: A checks element of ('BOXCAR_WIDTH', int) will check if
        config_dict['BOXCAR_WIDTH'] is of type int.
    """
    
    for pair in checks:
        if not isinstance(config_dict[pair[0]], pair[1]):
            raise ValueError(f'{pair[0]} is not a {pair[1]}. Only {pair[1]} values are allowed.')


def execute_command(cmd, b_quiet=True):
    """
    Executes the provided OS command.

    Parameters
    ----------
    cmd : str
        The command to be executed.
    b_quiet : bool
        Specify whether the output from the command should be quieted.
        True specifies that no output from the executed command
        will be seen.

    Returns
    -------
    None
    """

    if b_quiet:
        cmd += ' >/dev/null 2>&1'

    os.system(cmd)


def find_nth(haystack, needle, n):
    """
    Finds the index of the nth occurrence of a substring.
    Method found here: https://stackoverflow.com/a/1884277

    Parameters
    ----------
    haystack : str
        The input string.
    needle : str
        The desired substring.
    n : int
        The nth occurrence of the substring.
    
    Returns
    -------
    start : int
        The index of the nth occurrence.
    """
    
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1

    return start


def natural_sort(l):
    """
    Sort the provided by natural, human standards as opposed to machine standards.
    Method found here: https://stackoverflow.com/a/4836734
    """

    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    
    return sorted(l, key=alphanum_key)


def get_evtfile_from_hkfile(hk_file):
    """
    Get the path to the event file corresponidng to the given hk file.
    """

    p = parse.parse(HK_FILE_PATH_FORMAT, hk_file)
    evt_file = EVT_FILE_PATH_FORMAT.format(**p.named)

    return evt_file


def get_hkfile_from_evtfile(evt_file):
    """
    Get the path to the hk file corresponding to the given event file.
    """

    p = parse.parse(EVT_FILE_PATH_FORMAT, evt_file)
    hk_file = HK_FILE_PATH_FORMAT.format(**p.named)

    return hk_file


def characterize_frames(evt_data, frame_length):
    """
    Determines the number of frames such that each bin is exactly
    the width that is specified by frame_length. The method will
    choose a new end time that is **before** the original end
    time such that an exact integer number of frames can fit
    within the start and (new) end time.

    Parameters
    ----------
    evt_data : FITS record array
        A FITS record array containing the data from evt_file.
    hdr : FITS header
        A FITS header corresponding to evt_data.
    
    Returns
    -------
    start_time : float
        The start time of the observation in NuSTAR time.
    end_time : float
        The new end time of the observation in NuSTAR time.
    num_frames : int
        The number of frames in the observation.
    """

    start_time = evt_data[0]['TIME']
    end_time = evt_data[-1]['TIME']
    end_time = end_time - ( (end_time-start_time) % frame_length )
    num_frames = int((end_time - start_time)/frame_length)

    return start_time, end_time, num_frames


def get_nominal_coordinate(hdr, time=None):
    """
    Retrieves the nominal coordinate from the given header
    and corrects it if a time is given. The coordinate is
    converted from RA and Dec into solar coordinates in
    arcseconds.

    Parameters
    ----------
    hdr : FITS header
        The header containing the coordinate.
    time : Astropy time
        The time at which the correction is performed.

    Returns
    -------
    x : Astropy Quantity
        The x-coordinate of the nominal position in units
        of arcseconds.
    y : Astropy Quantity
        The y-coordinate of the nominal position in units
        of arcseconds.
    """

    if time is None:
        time = [Time(hdr['MJD-OBS'], format='mjd')]
    elif not isinstance(time, list):
        time = [time]

    ra, dec = [hdr['RA_NOM']]*u.deg, [hdr['DEC_NOM']]*u.deg
    x, y = nustar.convert._delta_solar_skyfield(ra, dec, time) # Units of degrees
    x, y = x.to(u.arcsecond), y.to(u.arcsecond)

    return x, y


def filter_far_data(evt_data, hdr):
    """
    Removes the data that is unphysically far from the FOV.
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
    """

    # print(evt_data['X'])
    # x, y = get_nominal_coordinate(hdr)
    x, y = np.mean(evt_data['X']), np.mean(evt_data['Y'])

    for field in list(hdr.keys()):
        if field.find('TYPE') != -1:
            if hdr[field] == 'X':
                xval = field[5:8]
            if hdr[field] == 'Y':
                yval = field[5:8]
    npixx=hdr['TLMAX'+xval]
    npixy=hdr['TLMAX'+yval]
    pixsizex=np.abs(hdr['TCDLT'+xval])
    pixsizey=np.abs(hdr['TCDLT'+yval])

    x = (x-npixx*0.5) * pixsizex * u.arcsec
    y = (y-npixy*0.5) * pixsizey * u.arcsec

    delta = u.Quantity(1000, 'arcsec')
    xmin, xmax = x-delta, x+delta
    ymin, ymax = y-delta, y+delta

    # Be sure to use the native conversion unit in the header.
    unit = hdr['TCUNI14']
    min_max = [v.to(unit).value for v in [xmin, xmax, ymin, ymax]]
    filtered_data = evt_data[nustar.filter.by_xy(evt_data, hdr, min_max)]

    return filtered_data


def get_event_data(evt_file, time_interval=None, b_filter_far_data=True):
    """
    Obtains the data and header in a NuSTAR event file.

    Parameters
    ----------
    evt_file : str
        Name of the input file.
    time_interval : list or tuple of strings
        Specifies the start and end time desired.
        Has the form [start_time, end_time] with the
        time strings in iso format: 'YYYY-MM-DD HH:MM:SS'
    b_filter_far_data : bool
        Specifies whether the unphysically far data should
        be filtered from the event list.

    Returns
    -------
    evt_data : FITS record array
        A FITS record array containing the data from evt_file.
    hdr : FITS header
        A FITS header corresponding to evt_data.
    """

    with fits.open(evt_file) as hdu:
        evt_data = hdu[1].data
        hdr = hdu[1].header

    if time_interval is not None:
        evt_data = evt_data[nustar.filter.by_time(evt_data, hdr,
            Time(time_interval, format='iso', scale='utc'))]

    if b_filter_far_data:
        evt_data = filter_far_data(evt_data, hdr)

    return evt_data, hdr


def moving_average(a, n):
    """
    Computes the moving average with width n across the array a.
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
    """
    
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    
    return ret[n - 1:] / n


def clean_directory_string(dir_path):
    """
    Modifies the input path string to personal preferences.

    Parameters
    ----------
    dir_path : str
        The directory path to be cleaned.

    Returns
    -------
    dir_path : str
        The cleaned directory path.
    """

    if dir_path[-1] != '/':
        dir_path = dir_path + '/'
        
    return dir_path


def copy_file(file_path, to_dir):
    """
    Copies file_path to to_dir.

    Parameters
    ----------
    file_path : str
        Path to the file of interest.
    to_dir : str
        The directory to which the configuration file will be copied.

    Returns
    -------
    None    
    """
    
    os.system(f'cp {file_path} {to_dir}')


def get_cwd():
    
    return os.getcwd() + '/'


def get_home():

    return str(Path.home())


def create_directory(dirs):
    """
    Checks if the provided directory exists.
    Creates the directory if it does not exist.
    A list can be provided to create several directories.

    Parameters
    ----------
    dirs : str or list
        The absolute directory path to be checked,
        or a list of directories to create.
    """

    if not isinstance(dirs, list):
        dirs = [dirs]

    for d in dirs:
        os.makedirs(d, exist_ok=True)


def verify_path(in_path, b_is_dir=True, b_make_absdir=True):
    """
    Verifies whether the provided path or directory is valid or not.
    Adds a '/' to the end of the string if one is not already there.
    Checks if dir is an absolute directory or not. If it is not,
    it is made into one.

    Parameters
    ----------
    in_path : str
        Input path to be checked.
    b_is_dir : bool
        Specifies whether in_path is a directory.
    b_make_absdir : bool
        Specifies whether or not the make in_path
        an absolute directory.

    Returns
    -------
    in_path : str
        Outputs the input string with some slight modifications.
        If the directory is not good, an exception is raised.
    """

    if b_is_dir:
        in_path = clean_directory_string(in_path) # Clean up the string

    # Check if the directory exists
    if not os.path.exists(in_path):
        raise FileNotFoundError(f'\'{in_path}\' is not a valid path.')
    elif b_make_absdir:
        # Check if dir is an absolute directory. It is isn't, make it absolute.
        in_path = os.path.abspath(in_path) + '/'
    
    return in_path


def get_file_name(in_file, b_include_ext=False):
    """
    Returns the file name of the input string.
    Can specify whether the file extension is desired or not.

    Ex: get_figure_path('/Users/rbmasek/nustar/data/20170321/20210015001/event_cl/nu20210015001A06_cl_sunpos.evt', False)
    Returns 'nu20210015001A06_cl_sunpos'

    Parameters
    ----------
    in_file : str
        The path containing the file name.
    b_include_ext : bool
        Specifies whether to include the file extension in the return.
    
    Returns
    -------
    file_name : str
        The file name extracted from in_file.
    """

    file_name = ''
    if b_include_ext:
        file_name = in_file[find_nth(in_file, '/', in_file.count('/'))+1:]
    else:
        file_name = in_file[find_nth(in_file, '/', in_file.count('/'))+1:in_file.find('.')]
    
    return file_name


def get_gti_file_from_evt_file(evt_file):

    return evt_file.replace('_cl_sunpos.evt', '_gti.fits')


def get_id_dir_from_evt_file(in_file):
    """
    Returns the ID directory from the event file.

    Ex: get_figure_path('/Users/rbmasek/nustar/data/20170321/20210015001/event_cl/nu20210015001A06_cl_sunpos.evt')
    Returns '/Users/rbmasek/nustar/data/20170321/20210015001/'
    """

    return in_file.split('event_cl')[0]


def get_id_dir_from_hk_file(in_file):
    """
    Returns the ID directory from the event file.

    Ex: get_figure_path('/Users/rbmasek/nustar/data/20170321/20210015001/event_cl/nu20210015001A06_cl_sunpos.evt')
    Returns '/Users/rbmasek/nustar/data/20170321/20210015001/'
    """

    return in_file.split('hk')[0]


def get_id_from_id_dir(id_dir):
    """
    Returns a string containing the data ID from the provided ID directory.
    """

    return parse.parse(ID_DIR_PATH_FORMAT, id_dir)['id_num']


def get_date_from_id_dir(id_dir):
    """
    Returns a string containing the observation date from the provided ID directory.
    """
    
    path = os.path.normpath(id_dir)
    # return parse.parse(ID_DIR_PATH_FORMAT, id_dir)['date']
    return path.split(os.sep)[-2]


def get_fpm_from_filename(file_name):
    """
    Returns the FPM of the provided file name.
    """

    fpm = 'A'
    if 'B06' in file_name or 'B_fpm' in file_name:
        fpm = 'B'

    return fpm


def get_relative_time():
    """
    Returns the relative time since January 1, 2010
    since that is the time NuSTAR measures against.
    """

    rel_t = datetime.strptime('2010-01-01 00:00:00', DATE_STR_FORMAT).replace(tzinfo=timezone.utc)
    return rel_t


def convert_nustar_time_to_datetime(t):
    """
    Converts NuSTAR timestamps (in seconds since 2010-01-01)
    to datetime objects.

    Parameters
    ----------
    t : float
        Number of seconds since 2010-01-01.
    
    Returns
    -------
    dt_times : datetime.datetime
        Datetime object corresponding to the input time.
    """

    rel_t = get_relative_time()
    dt_time = rel_t + timedelta(seconds=t)

    return dt_time


def convert_nustar_time_to_string(t):

    dt_time = convert_nustar_time_to_datetime(t)
    
    return dt_time.strftime(DATE_STR_FORMAT)


def convert_nustar_time_to_astropy(t):

    s = convert_nustar_time_to_string(t)

    return Time(s, format='iso', scale='utc')


def convert_string_to_datetime(s):

    return datetime.strptime(s, DATE_STR_FORMAT).replace(tzinfo=timezone.utc)


def convert_string_to_astropy(s):

    return Time(s, format='iso', scale='utc')


def convert_string_to_nustar_time(t):
    """
    Converts a timestamp t in format 'YYYY-MM-DD HH:MM:SS' to number of seconds since Jan 1, 2010.
    """

    rel_t = get_relative_time()
    time = datetime.strptime(t, DATE_STR_FORMAT).replace(tzinfo=timezone.utc)

    return (time - rel_t).total_seconds()


def convert_to_iso(t):

    return t.replace('/', '-').replace('T', ' ')


def add_timedelta_to_string(s, td):
    """
    Add (or subtract) the provided number of seconds to the provided string.

    Parameters
    ----------
    s : str
        Formatted datetime string.
    td : int or float
        Timedelta to be added to s, in seconds.
    
    Returns
    -------
    Formatted datetime string with the timedelta added.
    """

    return datetime.strftime((datetime.strptime(s, DATE_STR_FORMAT) + timedelta(seconds=int(td))), DATE_STR_FORMAT)


def convert_pi_to_energy(pi):
    """
    Converts pulse invariant from channel space to energy in units of keV.
    """

    return (0.040*pi) + 1.6


def gunzip_file(gz_path):
    """
    Gunzips the provided .gz file and keeps the original .gz file.
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
    """

    unzipped_file_path = gz_path[:gz_path.find('.gz')]
    if not os.path.isfile(unzipped_file_path):
        if gz_path[-1] == '/':
            gz_path = gz_path[:-1]        
        os.system('gunzip -c ' + gz_path + ' > ' + unzipped_file_path)

    return unzipped_file_path