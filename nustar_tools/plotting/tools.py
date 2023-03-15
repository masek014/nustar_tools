import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.special
from scipy.stats import skewnorm

import warnings
warnings.simplefilter('ignore')

from ..utils import utilities


STYLES_DIR = f'{utilities.os.path.dirname(utilities.os.path.realpath(__file__))}/styles/'


def apply_style():
    plt.style.use(f'{STYLES_DIR}plot.mplstyle')


def apply_colorbar(fig, ax, width=0.005, **kwargs):
    """
    Use the apply_colorbar method from mapping.tools.
    """
    
    from ..mapping.tools import apply_colorbar
    apply_colorbar(fig, ax, width, **kwargs)


def save_plot(fig, fig_dir, fig_name, type='png'):
    """
    Parameters
    ----------
    fig : matplotlib figure
        The figure object of the map.
    fig_dir : str
        The directory where the image file will be saved.
    fig_name : str
        The name of the figure.
    type : str
        The image type.
    """

    fig_dir = utilities.verify_path(fig_dir)
    utilities.create_directory(fig_dir)
    ext = '.' + type
    if ext not in fig_name:
        fig_name = fig_name + ext
    # fig.patch.set_alpha(0) # Make the border transparent
    plt.savefig(fig_dir+fig_name, bbox_inches='tight')


def get_frame_limits(evt_data, frame_length):
    """
    Determine the frame limits characterized on the frame length.

    Parameters
    ----------
    evt_data : FITS record
        The data determining the limits.
    frame_length : int
        The length of each time bin, in seconds.

    Returns
    -------
    x_min : datetime.datetime
        The lower time bound of the frame data.
    x_max : datetime.datetime
        The upper time bound of the frame data.
    """

    x_min, x_max, _ = utilities.characterize_frames(evt_data, frame_length)
    x_min = utilities.convert_nustar_time_to_datetime(x_min)
    x_max = utilities.convert_nustar_time_to_datetime(x_max)

    return x_min, x_max


def get_stair_datetimes(stair):
    """
    Converts the timestamps from a matplotlib stair (patch) object into
    an array of datetime objects.

    Parameters
    ----------
    stair : matplotlib patch
        The stair for which the times will be converted.

    Returns
    -------
    dt_times : np.array of datetime
        The converted array containing the datetime objects
        for the provided stair.
    """

    stair_times = stair.get_data()[1]
    unix_times = stair_times*24*60*60
    dt_times = np.array([utilities.datetime.fromtimestamp(t, tz=utilities.timezone.utc) for t in unix_times])

    return dt_times


def choose_tick_interval(x_min, x_max, b_datetime=True):
    """
    Chooses the best interval (in minutes)
    between the major ticks on the time axis.

    Parameters
    ----------
    x_min : datetime.datetime
        The minimum time value on the x-axis.
    x_max : datetime.datetime
        The maximum time value on the x-axis.
    b_datetime : bool
        Specifies whether the provided x_min and x_max are datetime objects.

    Returns
    -------
    time_interval : int
        The time interval between ticks in units of minutes.
    """

    time_diff = 0
    if b_datetime:
        time_diff = (x_max - x_min).total_seconds() # Total time in seconds
    else:
        time_diff = x_max - x_min

    minutes = divmod(time_diff, 60)[0] + 1 # Add one to round up
    time_list = [0.2, 1, 2, 5, 10, 30]
    time_interval = 0
    for t in time_list:
        if minutes / t >= 3: # Ensures at least 3 major tick marks
            time_interval = t
        else:
            break

    return time_interval


def set_x_ticks(ax, x_min=None, x_max=None, b_minor_ticks=True):
    """
    This method applies several cosmetic changes to the x-axis with regards to the ticks.
    It determines the spacing between the major ticks and configures the minor ticks.
    If minor ticks are added, the number of minor ticks between two major ticks is automatically determined.
    This method also sets the format of the tick labels.

    Parameters
    ----------
    ax : matplotlib axes
        The axes to which the changes will be applied.
    x_min : datetime.datetime
        The lower limit on the x-axis. 
    x_max : datetime.datetime
        The upper limit on the x-axis.
    config_dict : dict
        Dictionary containing the settings for the tick lengths and widths.
    b_minor_ticks : bool
        Specify whether minor ticks should be included on the x-axis.
    """

    if x_min is None and x_max is None:
        x_min, x_max = ax.get_xlim()
        x_min = utilities.datetime.fromtimestamp(x_min*24*60*60, tz=utilities.timezone.utc)
        x_max = utilities.datetime.fromtimestamp(x_max*24*60*60, tz=utilities.timezone.utc)

    chosen_time = choose_tick_interval(x_min, x_max)

    # Set up the major ticks
    xlocator = matplotlib.dates.MinuteLocator(byminute=np.arange(0, 60, chosen_time))
    ax.xaxis.set_major_locator(xlocator)
    formatter = matplotlib.dates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis='x', which='both', direction='in')

    # Set up the x-axis minor ticks based on the time interval
    if b_minor_ticks:
        x_locator_number = 2
        if chosen_time > 1:
            for i in range(2, 11):
                if chosen_time % i == 0:
                    x_locator_number = i
        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(x_locator_number))


def set_y_ticks(ax, b_minor_ticks=True):
    """
    This method formats the ticks on the y-axis.

    Parameters
    ----------
    ax : matplotlib ax
        The ax to which the changes will be applied.
    b_minor_ticks : bool
        Specify whether minor ticks should be included on the y-axis.
    """

    ax.tick_params(axis='y', which='both', direction='in')
    if b_minor_ticks:
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())


def find_zeros(data):
    """
    Returns a list of the indices **just before** the zero,
    i.e. an index i in the list is the data point
    before the zero, and the index i+1 is the data point
    after the zero.

    Parameters
    ----------
    data : np.array
        The input data for which the zeros will be found.
    
    Returns
    -------
    zero_indices : np.array
        A 1D array of indices.
    """

    zero_indices = []
    for i in range(0, data.size-1):
        cur_val = data[i]
        next_val = data[i+1]
        if cur_val < 0 and next_val > 0:
            zero_indices.append(i)
        elif cur_val > 0 and next_val < 0:
            zero_indices.append(i)

    return np.array(zero_indices)


def add_smoothed_curve(ax, line, smoothing_width=None, **kwargs):
    """
    Adds a smoothed version of the given line to the provided axes
    using a moving average.

    Parameters
    ----------
    ax : matplotlib axes
        The axes on which the smoothed curve will be drawn.
    line : line object
        The line containing the data to be smoothed.
    smoothing_width : int
        The window width of the moving average.
    kwargs : dict
        Keyword arguments for the cosmetic features of the line.

    Returns
    -------
    smoothed_line : line object
        The line containing the smoothed data.
    """

    default_kwargs = {'color': 'black', 'linestyle': 'dashed', 'linewidth': 1}
    kwargs = {**default_kwargs, **kwargs}

    if smoothing_width is None:
        smoothing_width = int(utilities.get_config_option('LightcurveSettings', 'SMOOTHING_WIDTH'))
    x, y = line.get_xdata(), line.get_ydata()
    smoothed_y = utilities.moving_average(y, smoothing_width)
    smoothed_line = ax.plot(x, smoothed_y, **kwargs)[0]

    return smoothed_line


def plot_derivative(ax, line, b_add_derivative=True, b_add_smoothed=False, b_add_zeros=False):
    """
    Plots the derivative fo the smooths

    Assumes datetime objects are on the x-axis.

    This method is designed a bit strangely as it tries to account for each
    combination of boolean parameters.

    Parameters
    ----------
    ax : matplotlib axes
        The axes on which the curve(s) will be drawn.
    line : line object
        The line containing the data for the derivative.
    b_add_derivative : bool
        Specifies whether to add the derivative to the plot.
    b_add_smoothed : bool
        Specifies whether to add the smoothed derivative to the plot.
    b_add_zeros : bool
        Specifies whether to add the zero points of the derivative to the plot.

    Returns
    -------
    lines : list
        The list of lines added to the plot.
    """

    lines = []
    dt_times, values = line.get_xdata(), line.get_ydata()

    # Compute the derivative.
    dx = np.array([t.total_seconds() for t in np.diff(dt_times)])
    dy = np.diff(values)
    dydx = dy/dx

    # Create a second y-axis to plot for the derivative.
    ax2 = ax.twinx()
    if (not b_add_derivative) and (not b_add_smoothed):
        ax2.set_yticks([])

    # Plot the lines.
    d_alpha, s_alpha = 0, 0
    if b_add_derivative:
        d_alpha = 0.6
    if b_add_smoothed:
        s_alpha = 0.6

    derivative_line = ax2.plot(dt_times[:-1], dydx,
        color='violet', alpha=d_alpha, label='Derivative')[0]
    smoothed_derivative_line = add_smoothed_curve(ax2, derivative_line,
            color='purple', alpha=s_alpha, label='Smoothed derivative')

    if b_add_derivative:
        lines.append(derivative_line)
        ax2.set_ylabel('Derivative (Counts s${}^{-2}$)')
        ax2.axhline(0, color='darkred', linewidth=0.5, linestyle='dashed')

    if b_add_smoothed:
        lines.append(smoothed_derivative_line)
        ax2.set_ylabel('Derivative (Counts s${}^{-2}$)')
        ax2.axhline(0, color='darkred', linewidth=0.5, linestyle='dashed')
    
    if b_add_zeros:
        smoothed_dydx = smoothed_derivative_line.get_ydata()
        zero_indices = find_zeros(smoothed_dydx)
        zero_times = (dt_times[zero_indices+1] - \
                      dt_times[zero_indices])/2 + \
                      dt_times[zero_indices]

        zeros_line = ax.plot(zero_times, values[zero_indices],
            'rx', markersize=8, mew=2, label='Derivative zeros')[0]
        lines.append(zeros_line)

    return lines


def add_event_lines(axes, start_time, end_time, b_vertical=True, b_shaded=False, **kwargs):
    """
    Adds two vertical lines to the plot using the
    provided input times.

    Parameters
    ----------
    axes : matplotlib axes
        The axes to which the lines will be applied
    start_time : datetime.datetime
        The time of the first vertical line.
    end_time : datetime.datetime
        The time of the second vertical line.
    b_shaded : bool
        Specify if the region between the two vertical lines
        will be shaded.
    kwargs : dict
        Keyword arguments for the cosmetic features of the lines.
    """

    default_kwargs = {'color': 'black', 'linestyle': 'dashed', 'linewidth': 1}
    kwargs = {**default_kwargs, **kwargs}

    x_dat = get_stair_datetimes(axes.patches[0]) # Get stair time edges

    # Add the vertical lines.
    if b_vertical or (start_time == end_time):

        if b_shaded:
            start_index = np.argmax(x_dat>=start_time)
            end_index = np.argmin(x_dat<=end_time)-1
            start_edge = x_dat[start_index]
            end_edge = x_dat[end_index]
        else:
            start_edge = start_time + utilities.timedelta(seconds=(x_dat[2]-x_dat[1]).total_seconds())
            end_edge = end_time - utilities.timedelta(seconds=(x_dat[2]-x_dat[1]).total_seconds())

        axes.axvline(x=start_edge, **kwargs)
        axes.axvline(x=end_edge, **kwargs)
    
    # Shade the region between the lines.
    if b_shaded:
        start_bools = x_dat >= start_time
        end_bools = x_dat <= end_time
        bool_arr = [start_bools[i] and end_bools[i] for i in range(0, np.size(start_bools))]
        axes.fill_between(x_dat, 0, 1, where=bool_arr,
                color=kwargs['color'], alpha=0.2, transform=axes.get_xaxis_transform())


def add_events(ax, events_dict, color='lightsteelblue', frame_labelsize=10):
    """
    Shades the event time intervals for all events contained in events_dict.
    The start frames of each event are marked on the top x-axis (or set 
    frame_labelsize=0 for no marking).

    Parameters
    ----------
    ax : matplotlib axes
        The axes on which the shaded regions will be drawn.
    events_dict : dict
        A dictionary containing the events.
        key = frame number, value = list of events.
    color : str
        The color of the shaded regions.
    frame_labelsize : int
        The font size for the event frame numbers.
    
    Returns
    -------
    xax2 : matplotlib axes or None
        The top x-axis on which the frame numbers were marked.
    """

    starts, ids, time_pairs = [], [], []
    for key in events_dict.keys():
        for event in events_dict[key]:
            start = utilities.convert_string_to_datetime(event.start)
            end = utilities.convert_string_to_datetime(event.end)
            time_pairs.append([start, end])
            if start not in starts:
                starts.append(start)
                ids.append((event.event_id).split('-')[0])

    for pair in time_pairs:
        add_event_lines(ax, pair[0], pair[1],
            b_vertical=True, b_shaded=True, linestyle='dotted', color=color)

    xax2 = None
    if frame_labelsize > 0:
        xax2 = ax.twiny()
        xax2.set(xlim=ax.get_xlim(), xticks=starts, xticklabels=ids)
        xax2.tick_params(axis='x', labelsize=frame_labelsize)

    return xax2

def add_skewnorm(time_edges, ax, amp, a, loc, scale, background=None):
    """
    Adds the fitted skewed gaussian to the lightcurve

    Parameters
    ----------
    time_edges : ptools.np.ndarray
        An array containing the time edges of the bins.
        The times are in NuSTAR time, i.e. the number
        of seconds passed since Jan 1. 2010.
    ax : matplotlib axes
        Provide pre-set axes to impose the fit curve on
    amp : float
        Amplitude of the fit skewed gaussian
    a : float
        Skewness of the fit skewed gaussian
    loc : float
        Center of the fit skewed gaussian
    scale : float
        Width of the fit skewed gaussian
    background : np array
        The background counts of the event
        Included if using residual rates instead of count rates

    Returns
    -------
    ax : matplotlib axes
        The axes on which the plot was made.
    """
    '''
    x_min = utilities.convert_nustar_time_to_datetime(time_edges[0])
    x_max = utilities.convert_nustar_time_to_datetime(time_edges[-1])
    '''

    x = np.linspace(time_edges[0], time_edges[-1], 2000)
    plotx = [utilities.convert_nustar_time_to_datetime(i) for i in x]
    fit = skewnorm(a, loc=loc, scale=scale)
    ploty = fit.pdf(x) * amp
    
    if background is None: #Case of using count rates
        ax.plot(plotx, ploty, color='red', ls='-', lw=5, label='Fit Curve')
    else: #Case of using residual rates
        plot_bkgd = np.zeros(shape=np.shape(x))
        for index in range(len(plotx)): #Iterate through all of the plotting indices
            for i in range(len(time_edges) - 1): #Check all of the time_edges to find correct background bin
                if (x[index] >= time_edges[i]) & (x[index] <= time_edges[i+1]):
                    #Set plotting background to correct bin of background and move to next plotting index
                    plot_bkgd[index] = background[i]
                    break
        
        #Plot residual fit + background rates
        ax.plot(plotx, ploty + plot_bkgd, color='orange', ls='-', lw=5, label='Fit Curve + Background')

    return ax

def skewed_gaussian(x, A, mu, sigma, gamma):

    y = A / (sigma*utilities.math.sqrt(2*utilities.math.pi)) * \
        np.exp(-(x-mu)**2/(2*sigma**2)) * \
            ( 1 + scipy.special.erf(gamma*(x-mu) / (sigma*utilities.math.sqrt(2))) )

    return y


def skewnorm_pdf(x, a, loc, scale):
    #Used for scipy.optimize's curve_fit
    return skewnorm.pdf(x, a, loc=loc, scale=scale)
