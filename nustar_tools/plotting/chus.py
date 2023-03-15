from . import tools as ptools
np = ptools.np
utilities = ptools.utilities
u = utilities.u


def get_evtfile_from_chufile(chu_file, fpm='A'):

    id_dir = utilities.get_id_dir_from_hk_file(chu_file)
    id_num = utilities.get_id_from_id_dir(id_dir)
    evt_file = utilities.EVT_FILE_PATH_FORMAT.format(id_dir=id_dir, id_num=id_num, fpm=fpm)

    return evt_file


def get_chu_data(data_file):
    """
    Reads the CHU data from the provided file and formats/organizes it.

    Parameters
    ----------
    data_file : str
        The absolute path to the input data file.

    Returns
    -------
    cdates : np.ndarray
        Array containing the timestamps for the data.
    newmask : np.ndarray
        Array containing the CHU combination data.
    """

    # Load in the CHU file
    with utilities.fits.open(data_file) as clist:
        cdata1 = clist[1].data
        cdata2 = clist[2].data
        cdata3 = clist[3].data
        chdr = clist[1].header

    mjdref=utilities.Time(chdr['mjdrefi'],format='mjd')
    ctims=utilities.Time(mjdref+cdata3['time']*u.s,format='mjd')

    # Convert to format matplotlib can handle
    # So going astropytime -> datetime -> matplotlibdates
    cdates = ptools.matplotlib.dates.date2num(ctims.datetime)

    # I think this is correct to work out the CHU mask
    # Based on the IDL code ../idl/load_nschu.pro
    # Just assinging 1 if in CHU1, 4 if in CHU2, 9 if in CHU3
    # Also a value of 5=> CHU12, 10 => CHU13, 14 => CHU123 etc.....
    maxres=20
    c1mask=np.all([[cdata1['valid'] == 1],[cdata1['residual'] < maxres],\
                [cdata1['starsfail'] < cdata1['objects']],[cdata1['chuq'][:,3] != 1]],axis=0)
    # These give True or False back so multiplying by number gives number or 0
    c1mask=c1mask[0]*1
    c2mask=np.all([[cdata2['valid'] == 1],[cdata2['residual'] < maxres],\
                [cdata2['starsfail'] < cdata2['objects']],[cdata2['chuq'][:,3] != 1]],axis=0)
    c2mask=(c2mask[0]*4)
    c3mask=np.all([[cdata3['valid'] == 1],[cdata3['residual'] < maxres],\
                [cdata3['starsfail'] < cdata3['objects']],[cdata3['chuq'][:,3] != 1]],axis=0)
    c3mask=(c3mask[0]*9)
    mask=c1mask+c2mask+c3mask

    # Tweak the mask labelling to make plotting easier
    # Maybe not the best way of doing this....
    newmask=np.zeros(len(mask))
    newmask[np.where(mask == 1)] = 1
    newmask[np.where(mask == 4)] = 2
    newmask[np.where(mask == 5)] = 3
    newmask[np.where(mask == 9)] = 4
    newmask[np.where(mask == 10)] = 5
    newmask[np.where(mask == 13)] = 6
    newmask[np.where(mask == 14)] = 7

    return cdates, newmask


# Based on the example at https://github.com/ianan/nustar_sac/blob/master/python/example_hk.ipynb
def make_chu_plot(hk_file, fig_dir='./', file_name='chu',
    xlim=None, axes_position=[]):
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
        List of coordinates used for positioning the CHU plot on a subplot (see combined_plots.py).
    """

    ptools.apply_style()

    id_dir = utilities.get_id_dir_from_hk_file(hk_file)
    cdates, newmask = get_chu_data(hk_file)

    # Use the first and last times in the dataset for the full-range x-axis.
    x_min = ptools.matplotlib.dates.num2date(cdates[0])
    x_max = ptools.matplotlib.dates.num2date(cdates[-1])
    start = x_min.strftime(utilities.DATE_STR_FORMAT)
    start_yyyymmdd, start_hhmmss = start.split(' ')

    if axes_position:
        # The produced CHU plot will be a subplot.
        ax = ptools.plt.axes(axes_position)
    else:
        # The produced CHU plot is a standalone plot.
        fig, ax = ptools.plt.subplots()
        ax.set_title(f'NuSTAR {start_yyyymmdd} CHU States')

    ax.legend_ = None # Turn the legend off
    ax.plot_date(cdates, newmask, color='sienna', linewidth=2, marker='+') # Plot the data, solid line with markers off

    # Configure the ax limits and labels.
    ptools.set_x_ticks(ax, x_min, x_max)
    ptools.set_y_ticks(ax, b_minor_ticks=False)

    ax.set(ylabel='CHU State', yticklabels=[' ','1','2','12','3','13','23','123',' '],
        yticks=np.arange(0, 9, 1.0), ylim=[0.5, 7.5])
    
    # Save the full unmodified CHU plot.
    if not axes_position:
        print(f'Saving CHU plot to {fig_dir}{file_name}_full')
        ptools.save_plot(fig, fig_dir, file_name+'_full')

    # Create the CHU plot with the frame bounds.
    if xlim is not None:
        ax.set_xlim([*xlim])
        ptools.set_x_ticks(ax, *xlim)
        ptools.set_y_ticks(ax, b_minor_ticks=False)

        # Save the livetime plot with the frames.
        if not axes_position:
            print(f'Saving CHU plot to {fig_dir}{file_name}')
            ptools.save_plot(fig, fig_dir, file_name)

    return ax


def generate_chus(id_dir):
    """
    Generate CHU plots using the data in the provided data ID directory.
    """

    # Get the data ID for the set provided directory.
    obs_id = utilities.get_id_from_id_dir(id_dir)
    fig_dir = utilities.FIGURES_DIR_PATH_FORMAT.format(id_dir=id_dir)

    # Create a CHU plot using the specified data files.
    gz_dat_file = f'{id_dir}hk/nu{obs_id}_chu123.fits.gz'
    unzipped_data_file = utilities.gunzip_file(gz_dat_file) # Ensure the data is unzipped
    make_chu_plot(unzipped_data_file, fig_dir=fig_dir)