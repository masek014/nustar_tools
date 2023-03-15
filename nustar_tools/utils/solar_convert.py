import importlib
from os.path import *

from nustar_pysolar import convert

from . import utilities

EVENT_DATA_EXTENSIONS = ['A06_cl.evt', 'B06_cl.evt'] # File extensions of the event data


def convert_to_solar_coords(in_file):
    """
    Converts the given .evt file into a .evt file in solar coordinates.
    The name of the output file is the original name with '_sunpos' appended to the end.

    Parameters
    ----------
    in_file : str
        Directory of the data ID.
        Ex: .../nustar_data/20210108/20612001001/event_cl/nu20612001001A06_cl.evt
    """

    # Make the new filename.
    (sfile, ext) = splitext(in_file)
    out_file = sfile + '_sunpos.evt'
    
    # Only do the conversion if the output file does not exist.
    if not isfile(out_file):
        print('Converting ' + in_file + ' to solar coordinates')
        evtdata, hdr = utilities.get_event_data(in_file, b_filter_far_data=False)
        importlib.reload(convert)
        (newdata, newhdr) = convert.to_solar(evtdata, hdr)
        utilities.fits.writeto(out_file, newdata, newhdr)


def generate_solar_data(id_dir):
    """
    The driver function for the conversion to solar data.
    This function accepts the directory of an event ID and converts the associated data files to solar coordinates.
    The converted files are placed in the same directory as the original files.

    Parameters
    ----------
    id_dir : str
        Directory of the data ID. Ex: /Users/rbmasek/nustar/data/20210108/20612001001/
    """

    dat_id = id_dir[utilities.find_nth(id_dir, '/', id_dir.count('/') - 1) + 1:-1]
    for ext in EVENT_DATA_EXTENSIONS:
        dat_file = id_dir + 'event_cl/' + 'nu' + dat_id + ext
        convert_to_solar_coords(dat_file)