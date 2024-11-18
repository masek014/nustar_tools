import gc
import os
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg') # Removes warning when running script: https://stackoverflow.com/q/34770694

from . import livetime
from . import tools as ptools
from ..utils import utilities

CONF_FILE = utilities.CONF_FILE
LTCV_CONFIG = {
    'PLOT_WIDTH': 12,
    'PLOT_HEIGHT': 6
}


def find_min_max_macropixel_indices(macropixel_arr):
    """
    Determine the bounding indeces for the macropixel array,
    i.e. the indeces which define the four corners of a
    rectangle which contains all nonzero data. 
    """

    num_rows = macropixel_arr.height
    num_cols = macropixel_arr.width

    left_index = num_cols # xmin
    right_index = 0 # xmax
    bottom_index = num_rows # ymin
    top_index = 0 # ymax

    found_bottom = False
    count_nonzero = 0

    for row_num in range(0,num_rows):
        for col_num in range(0,num_cols):
            m_pixel = macropixel_arr.get_macropixel(row_num, col_num)
            if m_pixel.photon_evts:
                count_nonzero += 1
                if not found_bottom:
                    found_bottom = True
                    bottom_index = row_num
                if top_index < row_num:
                    top_index = row_num
                if left_index > col_num:
                    left_index = col_num
                if right_index < col_num:
                    right_index = col_num

    return left_index, right_index, bottom_index, top_index


def calculate_lightcurve(evt_data, number_frames, time_range=None,
        energy_band=None, density=False):
    """
    Bins the evt_data into bins.
    An energy filter is applied which passes energies
    within the given minimum and maximum energy in keV.

    Parameters
    ----------
    evt_file : str
        Path to the event data.
    frame_length : int
        The length of an individual frame, in seconds.
    energy_band : tuple of 2
        Contains the lower and upper bounds
        for the energy, in keV.

    Returns
    -------
    counts : np array
        The counts for each frame (bin).
    time_edges : np array
        The time edges for each frame.
    """

    if energy_band is not None:
        inds = utilities.nustar.filter.by_energy(evt_data, *energy_band)
        evt_data = evt_data[inds]

    time_list = evt_data['TIME']
    counts, time_edges = ptools.np.histogram(time_list, number_frames,
        range=time_range, density=density)
    counts_err = ptools.np.sqrt(counts)

    return time_edges, counts, counts_err


def calculate_lightcurve_rates(evt_data, hk_file, number_frames,
        time_range=None, energy_band=None, density=False):
    """
    Obtains the light curve data from the event files
    then performs the livetime correction on the counts.

    Parameters
    ----------
    evt_file : str
        The absolute directory to the event data file.
    hk_file : str
        The absolute directory to the housekeeping data file.
        frame_length : int
        The length of an individual frame, in seconds.
    energy_band : tuple of 2
        Contains the lower and upper bounds
        for the energy, in keV.

    Returns
    -------
    counts : np array
        The counts for each frame (bin).
    time_edges : np array
        The time edges for each frame.
    count_rates : np array
        The count rates for each frame.
    livetimes : np array
        The the livetimes, as a unitless fraction, for each frame.
    """

    time_edges, counts, counts_err = calculate_lightcurve(evt_data, number_frames, time_range, energy_band, density)

    livetimes = livetime.get_livetime_data(hk_file, time_edges)

    # Now divide the counts by the effective exposure of each of the bins.
    rates = ptools.np.array(counts) / (livetimes * ptools.np.diff(time_edges))
    rates_err = counts_err / (livetimes * ptools.np.diff(time_edges))

    return time_edges, rates, rates_err


def make_lightcurve_plot(
        time_edges, values, values_err=None, fig=None, ax=None,
        xlabel='', ylabel='', title='',
        b_title_add_date=True, b_title_add_time=False, **kwargs
    ):
    """
    This is the general, skeleton structure for making a lightcurve plot.
    If ax is None, then new fig and ax objects will be created and returned.

    Also allows adding several lines to an already existing plot through
    the used of the fig and ax arguments. The axis labels and title will
    remain the same unless new labels are provided.

    time_edges : ptools.np.ndarray
        An array containing the time edges of the bins.
        The times are in NuSTAR time, i.e. the number
        of seconds passed since Jan 1. 2010.
    values : ptools.np.ndarray
        The values that are plotted on the y-axis.
    fig : matplotlib figure
        Can be provided to reuse existing figures.
    ax : matplotlib axes
        Can be provided to reuse existing axes.
    xlabel : str
        The label for the x-axis.
    ylabel : str
        The label for the y-axis.
    title : str
        The plot title.
    b_title_add_date : bool
        Specify to add the observation date to the title.
    b_title_add_time : bool
        Specify to add the observation time to the title.
    kwargs : dict
        Keyword arguments for the cosmetic features of the plotted line.

    Returns
    -------
    fig : matplotlib figure
        The figure on which the plot was made.
    ax : matplotlib axes
        The axes on which the plot was made.
    line : matplotlib line
        The line corresponding to the plotted data.
    """

    if ax is None:
        plt.style.use(os.path.dirname(os.path.realpath(__file__)) + '/styles/lightcurve.mplstyle')
        fig, ax = plt.subplots()
        x_min = utilities.convert_nustar_time_to_datetime(time_edges[0])
        x_max = utilities.convert_nustar_time_to_datetime(time_edges[-1])
        ax.set(
            xlim=[x_min, x_max],
            ylim=[0, ptools.np.max(values[ptools.np.isfinite(values)])*1.05],
            xlabel=xlabel,
            ylabel=ylabel,
            title=title
        )
        ptools.set_x_ticks(ax, x_min, x_max)
        ptools.set_y_ticks(ax)

    else:
        fig = plt.gcf()
        # If no new labels are provided, change the new labels to 
        # match the previous labels.
        if xlabel == '': xlabel = ax.get_xlabel()
        if ylabel == '': ylabel = ax.get_ylabel()
        if title == '': title = ax.get_title()
        b_title_add_date = False
        b_title_add_time = False

    dt_times = [utilities.convert_nustar_time_to_datetime(t) for t in time_edges]
    line = ax.stairs(values, dt_times, **kwargs)
    if values_err is not None:
        if 'color' in kwargs:
            color = kwargs['color']
        else:
            color = None
        dt_times = [utilities.convert_nustar_time_to_datetime(t) for t in time_edges]
        ax.fill_between(dt_times[:-1], values-values_err, values+values_err, step='post', color=color, alpha=0.15)

    # Configure the title.
    start_yyyymmdd, start_hhmmss = dt_times[0].strftime(utilities.DATE_STR_FORMAT).split(' ')
    if b_title_add_date: title += ' - ' + start_yyyymmdd
    if b_title_add_time: title += ', ' + start_hhmmss
    ax.set_title(title)

    return fig, ax, line


def make_event_macropixel_lightcurves(event, fig_dir='', file_name=''):
    """
    Make a grid of figures containing the lightcurve of
    each Macropixel in the provided Event.
    The count rates are plotted over time.

    Parameters
    ----------
    event : Event object
        The Event containing the data to be plotted.
    fig_dir : str
        The directory where the figure will be saved.
    file_name : str
        The name of the image file name.

    Returns
    -------
    fig : matplotlib figure
        The figure on which the plot was made.
    ax : matplotlib axes
        The axes on which the plot was made.
    """

    ptools.apply_style()

    macropixels, row_coords, col_coords = event.get_macropixel_counts(True)
    num_rows, num_cols = len(row_coords), len(col_coords)
    coord_pairs = macropixels.keys()
    min_x = ptools.np.max(col_coords)
    max_y = ptools.np.max(row_coords)

    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12*num_rows, 12*num_cols))
    if not isinstance(ax, ptools.np.ndarray):
        ax = ptools.np.array([[ax]])
    ax = ax.reshape(len(row_coords), len(col_coords)) # Guarantee that it's 2D

    for pair in coord_pairs:
        pixel = macropixels[pair]
        x, y = [int(n) for n in pair.split(',')]
        x = x - min_x - 1
        y = max_y - y
        _, _, counts_line = make_lightcurve_plot(pixel.time_edges, pixel.counts, fig, ax[y,x],
            xlabel='', ylabel=r'Counts', title='('+pair+')',
            color='pink', label='Counts')
        ax2 = ax[y,x].twinx()
        _, _, rates_line = make_lightcurve_plot(pixel.time_edges, pixel.count_rates, fig, ax2,
            xlabel='', ylabel=r'Counts $s^{-1}$', title='('+pair+')',
            color='navy', label='Count rates')
        _, _, avg_rates_line = make_lightcurve_plot(pixel.time_edges, pixel.average_rates, fig, ax2,
            xlabel='', ylabel='', title='',
            color='darkred', label='Average rates')

        lns = [counts_line, rates_line, avg_rates_line]
        labs = [l.get_label() for l in lns]
        ax[y,x].legend(lns, labs)

    if file_name == '':
        file_name = f'event{event.event_id}_macropixel_lightcurves_full'

    ptools.save_plot(fig, fig_dir, file_name)

    start_dt = utilities.datetime.strptime(event.start, utilities.DATE_STR_FORMAT).replace(tzinfo=utilities.timezone.utc)
    end_dt = utilities.datetime.strptime(event.end, utilities.DATE_STR_FORMAT).replace(tzinfo=utilities.timezone.utc)

    for pair in coord_pairs:
        pixel = macropixels[pair]
        x, y = [int(n) for n in pair.split(',')]
        x = x - min_x - 1
        y = max_y - y

        ptools.add_event_lines(ax[y,x], start_dt, end_dt,
            b_vertical=True, b_shaded=True, linestyle='dotted', color='steelblue',)

        # Zoom into the the event time range.
        time_padding = 120
        x_min = start_dt - utilities.timedelta(time_padding)
        x_max = end_dt + utilities.timedelta(time_padding)
        ax[y,x].set_xlim([x_min, x_max])
        ptools.set_x_ticks(ax[y,x], x_min, x_max)
        ax[y,x].legend(lns, labs)

    file_name = f'event{event.event_id}_macropixel_lightcurves'

    ptools.save_plot(fig, fig_dir, file_name)

    return fig, ax


def make_observation_lightcurve(evt_file, hk_file, frame_length, energy_range,
    axes_position=[], fig_dir='', file_name='lightcurve'):
    """
    Make the light curve for the given event file.

    Parameters
    ----------
    evt_file : str
        The absolute directory to the event data file.
    hk_file : str
        The absolute directory to the housekeeping data file.
    frame_length : int
        The length of the time bins, in seconds.
    energy_range : tuple
        The energy range (min, max) to be plotted, in keV.
    axes_position : list
        A list containing position of the axes. This is used when
        including the light curve plot as a subplot.
    fig_dir : str
        The directory where the figure will be saved.
    file_name : str
        The name of the image file name.

    Returns
    -------
    fig : matplotlib figure
        The figure on which the plot was made.
    ax : matplotlib axes
        The axes on which the plot was made.
    """

    ptools.apply_style()
    # utilities.apply_config_settings(LTCV_CONFIG, 'LightcurveSettings', CONF_FILE)

    evt_data, hdr = utilities.get_event_data(evt_file)
    start, end, num_frames = utilities.characterize_frames(evt_data, frame_length)
    time_edges, counts, counts_err = calculate_lightcurve(
        evt_data, num_frames, time_range=(start, end),
        energy_band=energy_range)
    time_edges, count_rates, count_rates_err = calculate_lightcurve_rates(
        evt_data, hk_file, num_frames, time_range=(start, end),
        energy_band=energy_range)
    deadtimes = livetime.find_deadtimes(time_edges, counts)

    # Create the plot.
    if axes_position:
        ptools.apply_style()
        ax = plt.axes(axes_position)
        fig, ax, rates_line = make_lightcurve_plot(time_edges, count_rates, count_rates_err, None, ax, color='gray')
        x_min = utilities.convert_nustar_time_to_datetime(time_edges[0])
        x_max = utilities.convert_nustar_time_to_datetime(time_edges[-1])
        ax.set(xlim=[x_min, x_max], ylabel='Counts s${}^{-1}$')
        ptools.set_x_ticks(ax, x_min, x_max)
        ptools.set_y_ticks(ax)
    else:
        fig, ax, rates_line = make_lightcurve_plot(time_edges, count_rates,
            ylabel='Counts s${}^{-1}$', color='gray', label='Count rates')
    line_list = [rates_line] # Used for creating the legend

    # Add the deadtime lines.
    for pair in deadtimes:
        start_dt = utilities.convert_nustar_time_to_datetime(pair[0])
        end_dt = utilities.convert_nustar_time_to_datetime(pair[1])
        ptools.add_event_lines(ax, start_dt, end_dt, linestyle='dashed', color='red')
    
    add_smoothed = False
    add_derivative = False
    add_smoothed_derivative = False
    add_derivative_zeros = False
    if add_smoothed:
        smoothed_line = ptools.add_smoothed_curve(ax, rates_line,
            color='black', label='Smoothed count rates')
        line_list.append(smoothed_line)
    
    b_derivatives = add_derivative or \
                    add_smoothed_derivative or \
                    add_derivative_zeros

    if b_derivatives:
        derivative_lines = ptools.plot_derivative(ax, rates_line,
            add_derivative,
            add_smoothed_derivative,
            add_derivative_zeros)
        line_list += derivative_lines

    # Add the legend.
    if len(line_list) > 1:
        labs = [l.get_label() for l in line_list]
        ax.legend(line_list, labs, loc=0)

    in_fpm = utilities.get_fpm_from_filename(evt_file)
    time_range = (utilities.convert_nustar_time_to_string(time_edges[0]),
        utilities.convert_nustar_time_to_string(time_edges[-1]))
    avg_livetime = round(livetime.compute_average_livetime(hk_file, time_range)*100,2)
    yyyymmdd = utilities.convert_nustar_time_to_string(time_edges[0]).split(' ')[0]
    title_str = f'NuSTAR FPM {in_fpm} {energy_range[0]}-{energy_range[1]} keV lightcurve, '+\
        f'\n{yyyymmdd} (avg. livetime {avg_livetime}%)'
    ax.set_title(title_str)

    if not axes_position:
        print(fig_dir + file_name)
        ptools.save_plot(fig, fig_dir, file_name)

    return fig, ax


def generate_lightcurves(id_dir, frame_length=10, energy_range=(2.5,10), axes_position=[]):

    id_dir = utilities.clean_directory_string(id_dir)
    obs_id = utilities.get_id_from_id_dir(id_dir)

    for fpm in ['A', 'B']:
        evt_file = utilities.EVT_FILE_PATH_FORMAT.format(id_dir=id_dir, id_num=obs_id, fpm=fpm)
        hk_file = utilities.HK_FILE_PATH_FORMAT.format(id_dir=id_dir, id_num=obs_id, fpm=fpm)
        utilities.gunzip_file(evt_file + '.gz')
        utilities.gunzip_file(hk_file + '.gz')
        make_observation_lightcurve(evt_file, hk_file, frame_length,
            energy_range, axes_position, fig_dir=id_dir+'/figures/', file_name=f'lightcurves{fpm}')