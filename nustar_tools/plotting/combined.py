from . import chus, livetime, lightcurves
from . import tools as ptools
from ..utils import utilities


CMBN_CONFIG = {
    'PLOT_WIDTH': 12,
    'PLOT_HEIGHT': 12
}


def combine_plots(id_dir, fpm, frame_length, energy_range,
    mp_array=None, fig_dir='./', file_name='combined_plot',
    conf_file=utilities.CONF_FILE):
    """
    Parameters
    ----------
    id_dir : str
        The directory for the observation ID.
    fpm : str
        The FPM of the telescope of interest.
    frame_length : int
        The length of the time bins, in seconds.
    energy_range : tuple
        The energy range (min, max) to be plotted, in keV.
    mp_array : MacropixelArray
        The MacropixelArray containing events. The time ranges of
        the events will be plotted on top of the light curve.
    fig_dir : str
        The output directory of the saved image.
    file_name : str
        The name of the output file (exlcuding extension).
    conf_file : str
        Path to the configuration file containing some formatting parameters.
    """

    ptools.apply_style()
    # utilities.apply_config_settings(CMBN_CONFIG, 'LightcurveSettings', conf_file)

    # Specify plot positions in the grid.
    lightcurve_position = [0.1, 0.655, 0.8, 0.25]
    livetime_position = [0.1, 0.375, 0.8, 0.25]
    chus_position = [0.1, 0.095, 0.8, 0.25]

    id_dir = utilities.clean_directory_string(id_dir)
    id_num = utilities.get_id_from_id_dir(id_dir)
    evt_file = utilities.EVT_FILE_PATH_FORMAT.format(id_dir=id_dir, id_num=id_num, fpm=fpm)
    hk_file = utilities.get_hkfile_from_evtfile(evt_file)

    evt_data, _ = utilities.get_event_data(evt_file)
    xlim = ptools.get_frame_limits(evt_data, frame_length)

    # Create the the combined plots for both FPMs.
    fig = lightcurves.plt.figure(figsize=(4,6))

    _, lightcurve_axis = lightcurves.make_observation_lightcurve(evt_file, hk_file,
        frame_length=frame_length, energy_range=energy_range,
        axes_position=lightcurve_position, mp_array=mp_array)
    lightcurve_axis.set_xticklabels([])
    lightcurve_axis.set_xlabel('')

    gz_ext = fpm + '_fpm.hk.gz'
    gz_dat_file = f'{id_dir}hk/nu{id_num}{gz_ext}'
    unzipped_data_file = utilities.gunzip_file(gz_dat_file)
    livetime_axis = livetime.make_livetime_plot(unzipped_data_file,
        xlim=xlim, axes_position=livetime_position, conf_file=conf_file)
    livetime_axis.set_xticklabels([])

    chu_ext = '_chu123.fits.gz'
    chu_dat_file = f'{id_dir}hk/nu{id_num}{chu_ext}'
    unzipped_data_file = utilities.gunzip_file(chu_dat_file)
    chus_axis = chus.make_chu_plot(unzipped_data_file,
        xlim=xlim, axes_position=chus_position)

    ptools.save_plot(fig, fig_dir, file_name)