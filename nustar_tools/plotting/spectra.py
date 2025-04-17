import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

from astropy.io import fits

from ..utils import time_tools, utilities
from ..plotting import tools as ptools


def array_livetime_correction(hk_file, counts, time_edges):
    """
    Perform the livetime correction on the provided array of counts.

    Parameters
    ----------
    hk_file : str
        The housekeeping file containing the livetime data.
    counts : np array
        The binned photon counts. Axis 0 is the time axis,
        and axis 1 is the counts axis.
    time_edges : np array
        The time edges corresponding to the counts.

    Returns
    -------
    count_rates : np array
        The livetime corrected count rates.
    """

    # Get the livetimes for the observation.
    hk_data, hk_hdr = utilities.get_event_data(
        hk_file, perform_filter=False)
    hk_times = hk_data['time']
    hk_livetimes = hk_data['livetime']

    # Get the average livetime correction for each of the bins from the housekeeping file livetime times.
    time_diff = np.diff(time_edges)
    count_rates = counts.copy()
    livetimes = np.zeros(len(time_edges)-1)
    for t in range(len(time_edges)-1):
        within_time = ((hk_times >= time_edges[t]) & (
            hk_times < time_edges[t+1]))
        livetimes_in_range = hk_livetimes[within_time]
        livetimes[t] = np.average(livetimes_in_range)
        count_rates[t, :] = count_rates[t, :]/(livetimes[t] * time_diff[t])

    return count_rates


def plot_spectrogram(evt_data, time_range=None, energy_range=(3, 10),
                     num_bins=20, cmap=plt.cm.jet, hk_file=None, b_livetime_correction=False,
                     title_append='', fig_dir='./', file_name='spectrogram'):
    """
    Make and plot a spectrogram of the provided data.

    Parameters
    ----------
    evt_data : FITS record
        The data from which the spectrogram will be made.
    time_range : tuple
        The time interval (start, end) to constrain the spectrogram.
        The format for the times must be YYYY-MM-DD HH:MM:SS.
    energy_range : tuple
        The energies to contrain the spectrogram,
        in units of keV.
    num_bins : int
        The number of energy bins (y-axis).
        The x-axis has twice as many bins.
    cmap : matplotlib colormap
        The colormap for the image.
    hk_file : str
        Path to the housekeeping file containing the
        livetime data for the livetime correction.
    b_livetime_correction : bool
        Specifies whether a livetime correction should be performed
        on the spectrogram counts.
    title_append : str
        A string appended to the figure title.
    fig_dir : str
        The output directory of the saved image.
    file_name : str
        The name of the output file (exlcuding extension).

    Returns
    -------
    fig : matplotlib figure
        The figure corresponding to the spectrogram.
    ax : matplotlib axes
        The axes corresponding to the spectrogram.
    """

    ptools.apply_style()

    times = evt_data['TIME']
    pis = evt_data['PI']
    energies = utilities.convert_pi_to_energy(pis)

    if time_range is not None:
        time_range = (
            time_tools.string_to_nustar(time_range[0]),
            time_tools.string_to_nustar(time_range[1])
        )
    else:
        time_range = [times[0], times[-1]]

    x_min = time_tools.nustar_to_datetime(time_range[0])
    x_max = time_tools.nustar_to_datetime(time_range[1])

    H, xedges, yedges = np.histogram2d(times, energies,
                                       bins=(num_bins*2, num_bins),
                                       range=(time_range, energy_range)
                                       )

    title_str = f'Spectrogram{title_append}'
    cbar_str = 'Counts'
    if b_livetime_correction:
        if hk_file is not None:
            H = array_livetime_correction(hk_file, H, xedges)
            title_str += ' Livetime Corrected'
            cbar_str = 'Count rates [ct/s]'
        else:
            print('[plot_spectrogram] HK file needed when \
                performing livetime correction.')

    # Convert the x-edges into datetime format.
    to_datetime = np.vectorize(datetime.datetime.fromtimestamp)
    xedges_datetime = to_datetime(xedges, tz=datetime.timezone.utc)

    # Make the plot.
    fig, ax = plt.subplots()
    norm_max = max(2, np.nanmax(H))
    norm = ptools.matplotlib.colors.LogNorm(1, norm_max)
    spectrogram = ax.pcolor(xedges_datetime, yedges, H.T,
                            cmap=cmap, norm=norm
                            )

    ax.set(ylabel='Energy [keV]', title=title_str)
    ptools.set_x_ticks(ax, x_min, x_max)
    ptools.apply_colorbar(fig, ax, cmap=cmap, norm=norm, label=cbar_str)

    ptools.save_plot(fig, fig_dir, file_name)

    return fig, ax


def plot_photon_spectrum(evt_data, energy_range=None, bin_width=0.2,
                         fig_dir='./', file_name='photon_spectrum'):
    """
    Make and plot a photon spectrum of the provided data.

    Parameters
    ----------
    evt_data : FITS record
        The data from which the spectrogram will be made.
    energy_range : tuple
        The energies to contrain the spectrogram,
        in units of keV.
    bin_width : float
        The width of the energy bins, in keV.
    fig_dir : str
        The output directory of the saved image.
    file_name : str
        The name of the output file (exlcuding extension).

    Returns
    -------
    fig : matplotlib figure
        The figure corresponding to the spectrum.
    ax : matplotlib axes
        The axes corresponding to the spectrum.
    """

    ptools.apply_style()

    energies = 0.04*evt_data['PI'] + 1.6
    if energy_range is None:
        energy_range = (np.floor(np.min(energies)), np.ceil(np.max(energies)))

    num_bins = int((energy_range[1]-energy_range[0])/bin_width)

    fig, ax = plt.subplots()
    counts, edges = np.histogram(energies, bins=num_bins, range=energy_range)
    ax.stairs(counts, edges, color='purple', label=f'{energy_range} keV')

    ax.set(xlabel='Energy (keV)', ylabel='Counts',
           title=f'Photon spectrum {energy_range} keV - {bin_width} keV step',
           xlim=energy_range, yscale='log')
    ax.xaxis.set_minor_locator(ptools.matplotlib.ticker.AutoMinorLocator(5))

    if energy_range[0] < 2.5 and energy_range[1] > 2.5:
        ax.axvline(2.5, color='red', linestyle='dashed', label='2.5 keV')
        ax.legend()

    ptools.save_plot(fig, fig_dir, file_name)

    return fig, ax


def make_grade_spectra(in_dir, fig_dir, fpm, file_name='grade_spectra'):
    """
    Generate a plot showing the grade spectra contained in the
    provided input directory. This method will automatically
    include data from ALL pha files in in_dir.

    If grade 0 and grades 21-24 are in pha files in in_dir,
    then the pileup-corrected curve will automatically be
    plotted.

    Parameters
    ----------
    in_dir : str
        The input directory containing the pha files.
    fig_dir : str
        The directory where the plot will be saved.

    Returns
    -------
    fig : matplotlib figure
        The figure corresponding to the spectrum.
    ax : matplotlib axes
        The axes corresponding to the spectrum.
    """

    in_dir = utilities.verify_path(in_dir)
    fig_dir = utilities.verify_path(fig_dir)

    colors = ['red', 'orange', 'blue', 'purple', 'brown', 'green', 'yellow']
    pha_files, labels = [], []
    sorted_dir = os.listdir(in_dir)
    sorted_dir.sort()

    # Put grade 0 first (sorting doesn't work due to how Python sorts).
    for f in sorted_dir:
        if f.endswith('_sr.pha') and f'{fpm}06' in f:
            if '_g0_sr' in f:
                pha_files.insert(0, in_dir + f)
                labels.insert(0, 'Grade ' + f.split('_g')[1].split('_sr')[0])
            else:
                pha_files.append(in_dir + f)
                labels.append('Grade ' + f.split('_g')[1].split('_sr')[0])

    title_str = ''
    fig, ax = plt.subplots()
    for f, c, l in zip(pha_files, colors, labels):

        with fits.open(f) as hdu:
            hdr, evt = hdu[1].header, hdu[1].data
        bins, counts = evt['CHANNEL'], evt['COUNTS']
        bins = utilities.convert_pi_to_energy(bins)

        ax.step(bins, counts, color=c, label=l)

        start_time = time_tools.nustar_to_astropy(
            hdr['TSTART']).strftime('%Y-%m-%d %H:%M:S')
        end_hhmmss = time_tools.nustar_to_astropy(
            hdr['TSTOP'].strftime('%H:%M:%S'))
        title_str = 'Grade Spectra ' + start_time + '-' + end_hhmmss
        title_str = f'Grade Spectra {start_time} - {end_hhmmss}'

    ax.set(
        xlabel='Energy (keV)', ylabel='Counts',
        title=title_str, yscale='log', xlim=(0, 15))

    # Add the corrected line, if applicable.
    g0_file, g_unphysical_file = None, None
    b_has_grade0, b_has_unphysical = False, False
    for f in pha_files:
        if '_g0_sr' in f:
            b_has_grade0 = True
            g0_file = f
        elif '_g21-24_sr' in f:
            b_has_unphysical = True
            g_unphysical_file = f

    if b_has_grade0 and b_has_unphysical:
        with fits.open(g0_file) as hdu:
            evt = hdu[1].data
        g0_counts = evt['COUNTS']
        with fits.open(g_unphysical_file) as hdu:
            evt = hdu[1].data
        g_unphysical_counts = evt['COUNTS']

        ax.step(bins, g0_counts-0.25*g_unphysical_counts,
                color='cyan', label='G0 - 0.25*G21-24')

    ax.legend()

    ptools.save_plot(fig, fig_dir, file_name)

    return fig, ax
