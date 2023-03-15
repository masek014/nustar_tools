from . import maps
from .movies import make_movies

mtools = maps.mtools
utilities = mtools.utilities

import warnings
warnings.simplefilter('ignore') # Suppress astropy warning about changing dates


INFO_FONT_SIZE = 10


def plot_exposure_maps(evt_data, hdr, spec_region=None,
    b_plot_fov=True, b_plot_detmap=True, b_fit_gaussian=False,
    b_add_contours=False, corners=None,
    fig_dir='', file_name='exposure_map'):
    """
    Make an exposure map for the provided event list.
    An exposure map shows the normalized pixel counts.

    Parameters
    ----------
    evt_data : astropy Table
        The input data to be plotted.
    hdr : header
        The header corresponding to the evt_data.
    spec_region : SkyRegion
        The specified region.
    b_plot_fov : bool
        Specifies whether the field of view should
        be drawn on the map.
    b_plot_detmap : bool
        Specifies whether the detector map should
        be added to the figure. A new map is created
        and placed next to the exposure map.
    b_fit_gaussian : bool
        Specifies whether a 2D gaussian should be fit
        to the exposure map data.
    corners : list
        Sets the limits of the axes. In the format
        [xmin, ymin, xmax, ymin] in arcseconds.
    fig_dir : str
        The output directory for the saved figure.
    file_str : str
        The file name (excluding extension) for
        the saved figure.

    Returns
    -------
    fig : matplotlib figure
        The figure containing the maps.
    map_axes : list
        List containing the map axes.
    """

    mtools.apply_style()
    text_color = 'black'

    nustar_map = maps.make_nustar_map(evt_data, hdr)
    nustar_submap = mtools.get_submap(nustar_map, corners)
    fig = mtools.plt.figure()

    # Adjust the size if plotting both maps.
    if b_plot_detmap:
        fig_size = fig.get_size_inches()
        fig_size[0] = fig_size[0]*2.4
        fig = mtools.plt.figure(figsize=fig_size)

    # Plot the count data.
    ax1 = fig.add_subplot(1, int(b_plot_detmap)+1, 1, projection=nustar_submap)
    map_axes = [ax1]
    cmap1 = mtools.plt.cm.get_cmap('Spectral_r')
    nustar_submap.plot(norm=mtools.mplcolors.Normalize(), cmap=cmap1)
    limb = nustar_submap.draw_limb(color='white', linewidth=1.25, linestyle='dotted', zorder=0, label='Solar disk')
    ax1.set(xlabel='x [arcsec]', ylabel='y [arcsec]')
    cbax1, cb1 = mtools.apply_colorbar(fig, ax1,
        norm=mtools.mplcolors.Normalize(), cmap=cmap1, label='Normalized Counts')

    if b_fit_gaussian:
        mtools.fit_gaussian(nustar_submap, ax1)

    if b_plot_detmap:
        ax2 = fig.add_subplot(122, projection=nustar_submap)
        fig, ax2, det_submap = maps.plot_det_map(evt_data, hdr, fig, ax2, corners=corners)
        ax2.set(xlabel='x [arcsec]', ylabel='y [arcsec]', title='Detector Visualization')
        map_axes.append(ax2)
        text_color = 'white'

    text_str = ''
    if b_plot_fov:
        fov_map = nustar_submap
        if b_plot_detmap:
            fov_map = det_submap
        fov = maps.FOV(evt_data, hdr)
        fov.plot(fov_map, map_axes[-1], edgecolor='pink')
        text_str += fov.get_fov_string()

    if spec_region is not None:
        x, y = spec_region.center.Tx.arcsec, spec_region.center.Ty.arcsec
        r = spec_region.radius.value
        pix_region = spec_region.to_pixel(nustar_submap.wcs)
        for a in map_axes:
            pix_region.plot(ax=a, lw=3, edgecolor='white',
                linestyle='dotted', facecolor='white',
                fill=True, alpha=0.25, label='Spec. Region')

        if b_add_contours:
            smoothed_submap = mtools.apply_gaussian_blur(nustar_submap, 2)
            mtools.draw_nustar_contours(smoothed_submap, a, range(10,100,10), spec_region, out_dir=fig_dir)
        
        if b_plot_detmap:
            dets_in_reg = mtools.find_dets_in_region(det_submap, pix_region)
            if -1 in dets_in_reg:
                dets_in_reg.remove(-1)
            text_str += 'dets in spec. region: {dets_in_reg}\n'

        text_str += f'spec. center: ({x:0.2f}\", {y:0.2f}\") \n'\
            'spec. radius: {r:0.2f}\"\n'

        # Check if the spec reg overlaps with FOV region.
        if b_plot_fov:
            pixel_diff = fov.check_region_outside_fov(spec_region)
            text_str += 'outside FOV: '
            if pixel_diff:
                text_str += 'yes ({pixel_diff} pixels)\n'
            else:
                text_str += 'no\n'

    # mtools.text(0.02, 0.88, text_str, fontsize=INFO_FONT_SIZE,
    #     color=text_color, va='center', transform=map_axes[-1].transAxes)
    mtools.save_map(fig, fig_dir, file_name)

    return fig, map_axes


# TODO: Finish this.
def make_exposure_movie(evt_file, start, number_frames, time_step, exposure_time=60, corners=None, fig_dir=''):
    """
    Make an exposure movie showing the time evolution of the event data.

    Parameters
    ----------
    evt_file : str
        Path to the event file.
    start : str
        The start time of the movie in YYYY-MM-DD HH:MM:SS
    exposure_time : int
        Seconds
    """

    print('Making exposure movie starting at ' + start)
    utilities.create_directory(fig_dir)

    evt_data, hdr = utilities.get_event_data(evt_file)
    if corners is None:
        m = maps.make_nustar_map(evt_data, hdr) # Used only for determining the bounding corners.
        corners = mtools.find_min_max(m.data)

    start_frame = 0
    end_frame = number_frames - 1 - exposure_time//time_step

    for frame in range(start_frame, end_frame):
        frame_start = utilities.add_timedelta_to_string(start, time_step*frame)
        frame_end = utilities.add_timedelta_to_string(frame_start, time_step+exposure_time)
        frame_data = evt_data[utilities.nustar.filter.by_time(evt_data, hdr,
            mtools.Time([frame_start, frame_end], format='iso', scale='utc'))]
        if len(frame_data) == 0:
            continue # Don't plot empty frames
        file_str = str(frame).rjust(len(str(number_frames)),'0')
        if not utilities.os.path.exists(f'{fig_dir}/exposure_map{file_str}.png'):
            plot_exposure_maps(frame_data, hdr, b_plot_fov=True,
                b_plot_detmap=True, b_fit_gaussian=False, corners=corners,
                fig_dir=fig_dir, file_name=f'exopsure_map{file_str}')
        else:
            print('Skipping ' + file_str)

    if fig_dir[-1] == '/':
        fig_dir = fig_dir[:-1]
    ind = utilities.find_nth(fig_dir, '/', fig_dir.count('/'))
    fig_dir = fig_dir[0:ind+1]
    folder_name = fig_dir[ind+1:]
    make_movies(fig_dir, folder_name)


def plot_fpm_maps(evt_file, time_interval=None,
    fig_dir='', file_name='fpm_map'):
    """
    Plots the fields of view of both FPMs against the solar disk.

    Parameters
    ----------
    evt_file : str
        The event file for one of the FPMs.
    time_interval : list or tuple of strings
        Specifies the start and end time desired.
        Has the form [start_time, end_time] with the
        time strings in iso format: 'YYYY-MM-DD HH:MM:SS'
    fig_dir : str
        The output directory of the saved figure.
    """

    mtools.apply_style()

    fig = None
    colors = ['blue', 'red']
    for i, fpm in enumerate(['A', 'B']):

        evt_file = evt_file.replace('A06', fpm+'06')
        evt_file = evt_file.replace('B06', fpm+'06')

        # Get the full map data and the det map.
        evt_data, hdr = utilities.get_event_data(evt_file, time_interval)
        nustar_map = maps.make_nustar_map(evt_data, hdr)

        if fig is None:
            nustar_submap = mtools.get_submap(nustar_map, [-1500, -1500, 1500, 1500])

            fig = mtools.plt.figure()
            ax = fig.add_subplot(111, projection=nustar_submap)
            nustar_submap.data[:,:] = mtools.np.nan
            nustar_submap.plot(norm=mtools.mplcolors.LogNorm(vmin=1,vmax=1))
            limb = nustar_submap.draw_limb(color='black', linewidth=1.25,
                linestyle='dotted', zorder=0, label='Solar disk')

        fov = maps.FOV(evt_data, hdr)
        fov.plot(nustar_submap, ax,
            b_draw_chip_gap=False, b_plot_center=False, b_plot_corners=False,
            fill=True, alpha=0.5, facecolor=colors[i], edgecolor=colors[i],
            linestyle='solid', lw=1, label=f'FPM {fpm}')
        text_str = f'FPM {fpm}\n{fov.get_fov_string()}'

        text_color = 'black'
        x_pos = 0.02 + i*0.5
        mtools.text(x_pos, 0.91, text_str, fontsize=INFO_FONT_SIZE+2,
            color=text_color, va='center', transform=ax.transAxes)

    if time_interval is None:
        end_time = utilities.convert_nustar_time_to_string(evt_data['TIME'][-1]).split(' ')[-1]
    else:
        end_time = time_interval[1].split(' ')[-1]

    ax.set(xlabel='x [arcsecond]', ylabel='y [arcsecond]',
        title=f'{ax.get_title()} - {end_time}')

    # ax.legend(loc=3, handles=artists) # TODO: Fix the labels. They arent showing up
    mtools.save_map(fig, fig_dir, file_name)