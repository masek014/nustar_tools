import datetime
import math

import astropy.units as u
import numpy as np

from astropy.time import Time

from . import tools as ptools
from ..utils import utilities


# File extensions of the zipped livetime data
DATA_EXTENSIONS = ['A_fpm.hk.gz', 'B_fpm.hk.gz']


def get_livetime_data(hk_file, time_edges):

    hk_data, hk_hdr = utilities.get_event_data(
        hk_file, perform_filter=False)
    hk_times = hk_data['time']
    hk_livetimes = hk_data['livetime']

    # Get the average livetime correction for each of the bins from the housekeeping file livetime times.
    livetimes = np.zeros(len(time_edges)-1)
    for t in range(len(time_edges)-1):
        within_time = ((hk_times >= time_edges[t]) & (
            hk_times < time_edges[t+1]))
        livetimes_in_range = hk_livetimes[within_time]
        livetimes[t] = np.average(livetimes_in_range)

    return livetimes


def compute_average_livetime(hk_file, time_range):
    """
    Read the livetime data and compute the average livetime.

    Parameters
    ----------
    hk_file : str
        Path to the housekeeping file.
    time_range : tuple
        Defines the time range (start, end) over which the average
        livetime will be found. In the format YYYY-MM-DD HH:MM:SS.

    Returns
    -------
    avg_livetime : float
        The average livetime as a fraction of the total time,
        i.e., multiply by 100 to get livetime percent.
    """

    # x_min = utilities.convert_string_to_datetime(time_range[0])
    # x_max = utilities.convert_string_to_datetime(time_range[1])
    x_min = Time(time_range[0], scale='utc').datetime
    x_max = Time(time_range[1], scale='utc').datetime

    hk_data, hk_hdr = utilities.get_event_data(
        hk_file, perform_filter=False)

    mjdref = Time(hk_hdr['mjdrefi'], format='mjd', scale='utc')
    ltims_dt = Time(
        mjdref+hk_data['time']*u.s, format='mjd', scale='utc').datetime

    livetime = hk_data['livetime']
    inds = ((ltims_dt > x_min) & (ltims_dt < x_max))

    avg_livetime = round(np.average(livetime[inds]), 4)

    return avg_livetime


def find_deadtimes(times, counts):
    """
    Find the dead times in the data.
    Dead times are defined as when the
    total counts measured by the detector is zero.

    Parameters
    ----------
    times : np array
        The time edges of the frames.
    counts : np array
        The counts for each bin.

    Returns
    -------
    deadtimes : list of lists
        A list containing the time intervals for each dead period.
        Each element in the list is a list defining the interval [start, end].
    """

    deadtimes = []  # A list of lists
    begin_deadtime = ''
    end_deadtime = ''
    deadtime_found = False

    # Check for deadtime.
    for i, count in enumerate(counts):

        if count == 0 and not deadtime_found:
            begin_deadtime = times[i]
            deadtime_found = True
        elif count != 0 and deadtime_found:
            end_deadtime = times[i]
            deadtime_found = False
            deadtimes.append([begin_deadtime, end_deadtime])

    return deadtimes


def find_saa_and_eclipses(times, vals, avg_width=15):
    """
    Finds the periods of dead time during the SAA passings
    and periods of eclipses for given data.
    For SAA passings, the method simply searches for periods
    when the live time is zero.
    For eclipses, a running average is performed on the live time
    with the specified width. An eclipse is identified when this
    average is greater than or equal to 90 (signifying a 90% live
    time of the detectors).

    Parameters
    ----------
    times : np array
        The array of times.
    vals : np array
        The array of livetime data.
    avg_width : int
        Size to use for the running average when identifying eclipses.

    Returns
    -------
    deadtimes : list
        List of lists containing the start time and end time pairs.
    eclipses : list
        List of lists containing the start time and end time pairs.
    """

    deadtimes = []  # A list of lists
    begin_deadtime = None
    end_deadtime = None
    deadtime_found = False

    eclipses = []  # A list of lists
    begin_eclipse = None
    end_eclipse = None
    eclipse_found = False

    for i, lt in enumerate(vals, start=avg_width-1):

        # Check for SAA deadtime.
        if lt == 0.0 and not deadtime_found:
            begin_deadtime = times[i-avg_width+1]
            deadtime_found = True
        elif lt != 0.0 and deadtime_found:
            end_deadtime = times[i-avg_width+1]
            deadtime_found = False
            deadtimes.append([begin_deadtime, end_deadtime])

        # Check for eclipse.
        if i >= vals.size:
            break
        sum = 0
        for j in range(i-avg_width+1, i+1):
            sum = sum + vals[j]
        avg = sum / avg_width
        if avg >= 0.9 and not eclipse_found:
            begin_eclipse = times[i-avg_width+math.ceil(avg_width/2)]
            eclipse_found = True
        elif avg < 0.9 and eclipse_found:
            end_eclipse = times[i-avg_width+math.ceil(avg_width/2)]
            eclipse_found = False
            eclipses.append([begin_eclipse, end_eclipse])

    # Handle the case if the data set ends during a dead period.
    if deadtime_found:
        end_deadtime = times[-1]
        deadtimes.append([begin_deadtime, end_deadtime])

    # Handle the case if the data set ends during an eclipse.
    if eclipse_found:
        end_eclipse = times[-1]
        eclipses.append([begin_eclipse, end_eclipse])

    return deadtimes, eclipses


def make_livetime_plot(hk_file, fig_dir='./', file_name='livetime',
                       xlim=None, axes_position=[], conf_file=utilities.CONF_FILE):
    """
    Produces a CHU plot from the provided input file.

    Parameters
    ----------
    hk_file : str
        Input file containing the CHU data.
    fig_dir : str
        The output directory of the saved image.
    file_name : str
        The name of the output file (exlcuding extension).
    xlim : tuple of datetime
        Defines the bounds of the time axis.
        Formatted as datetime objects.
    axes_position : list
        List of coordinates used for positioning the
        livetime plot on a subplot (see combined_plots.py).
    conf_file : str
        Path to the configuration file containing some formatting parameters.
    """

    LIVE_CONFIG = {}
    ptools.apply_style()

    id_dir = utilities.get_id_dir_from_hk_file(hk_file)
    in_fpm = utilities.get_fpm_from_filename(hk_file)

    # Load in the livetime file.
    with utilities.fits.open(hk_file) as llist:
        ldata = llist[1].data
        lhdr = llist[1].header

    mjdref = Time(lhdr['mjdrefi'], format='mjd')
    ltims = Time(mjdref+ldata['time']*u.s, format='mjd')

    livetimes = ldata['livetime']
    obs_date = (ltims.datetime)[0].strftime('%Y/%m/%d')

    # Convert to format matplotlib can handle, going astropytime -> datetime -> matplotlibdates.
    dates = ptools.matplotlib.dates.date2num(ltims.datetime)

    if axes_position:
        ax = ptools.plt.axes(axes_position)
    else:
        fig, ax = ptools.plt.subplots()
        ax.set_title(f'NuSTAR FPM {in_fpm} Livetime - {obs_date}')
    ax.legend_ = None  # Turn the legend off

    # Plot the data as a solid line with markers off.
    ax.plot_date(dates, livetimes, color='purple', linestyle='solid',
                 marker='', drawstyle='steps')

    avg_width = 20
    saa, eclipses = find_saa_and_eclipses(dates, livetimes, avg_width)

    # Add the saa lines.
    for pair in saa:
        ax.axvline(x=pair[0], linestyle='dashed', color='red')
        ax.axvline(x=pair[1], linestyle='dashed', color='red')

    # Add the eclipse lines.
    for pair in eclipses:
        ax.axvline(x=pair[0], linestyle='dashed', color='blue')
        ax.axvline(x=pair[1], linestyle='dashed', color='blue')

    # Here, x_min and x_max will include the entire livetime data,
    # including the SAA passings and eclipses.
    # (Used for the standalone livetime plots).
    x_min = ptools.matplotlib.dates.num2date(dates[0])
    x_max = ptools.matplotlib.dates.num2date(dates[-1])

    ax.set_xlim([x_min, x_max])
    ptools.set_x_ticks(ax, x_min, x_max)
    ptools.set_y_ticks(ax)
    ax.set(ylabel='NuSTAR Livetime', yscale='log')

    if not axes_position:
        # Save the full livetime plot.
        print(f'Saving livetime plot to {fig_dir}{file_name}_full')
        ptools.save_plot(fig, fig_dir, file_name+'_full')

    # Create the livetime plot with the frame bounds.
    if xlim is not None:
        ax.set_xlim([*xlim])
        ptools.set_x_ticks(ax, *xlim)
        ptools.set_y_ticks(ax)

        # Save the livetime plot with the frames.
        if not axes_position:
            print(f'Saving livetime plot to {fig_dir}{file_name}')
            ptools.save_plot(fig, fig_dir, file_name)

    return ax


def generate_livetime_plots(id_dir):
    """
    Generate livetime plots using the data in the provided data ID directory.
    """

    # Get the data ID for the set provided directory.
    obs_id = utilities.get_id_from_id_dir(id_dir)
    fig_dir = utilities.FIGURES_DIR_PATH_FORMAT.format(id_dir=id_dir)

    # Create a livetime plot for both the A and B sets.
    for ext in DATA_EXTENSIONS:
        gz_dat_file = f'{id_dir}hk/nu{obs_id}{ext}'
        unzipped_data_file = utilities.gunzip_file(
            gz_dat_file)  # Ensure the data is unzipped
        make_livetime_plot(unzipped_data_file, fig_dir=fig_dir)
