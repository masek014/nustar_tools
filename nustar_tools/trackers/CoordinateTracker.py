import datetime
import os
import pathlib
import pickle

import astropy.units as u
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import nustar_pysolar as nustar
import photutils.centroids
import ruptures as rpt

from astropy.io import fits
from astropy.table import QTable, Column
from astropy.time import Time
from matplotlib.collections import PathCollection
from scipy.fft import fft, fftfreq

from ..pixels import PixelArray as pa
from ..plotting import tools as ptools
from ..utils import time_tools, utilities


EVT_FILE_PATH_FORMAT = '{id_dir}/event_cl/nu{id_num:11}{fpm}06_cl.evt'
ATT_FILE_PATH_FORMAT = '{id_dir}/event_cl/nu{id_num:11}_att.fits'
MAST_FILE_PATH_FORMAT = '{id_dir}/event_cl/nu{id_num:11}_mast.fits'
OA_FILE_PATH_FORMAT = '{id_dir}/event_cl/nu{id_num:11}{fpm}_oa.fits'
DET1_FILE_PATH_FORMAT = '{id_dir}/event_cl/nu{id_num:11}{fpm}_det1.fits'


STYLES_DIR = pathlib.Path(__file__).absolute().parent / 'styles'
plt.style.use(STYLES_DIR / 'tracker.mplstyle')


def general_xy_to_radec(x, y, evt_hdr):
    '''A generalized version of nustar_pysolar.convert._xy_to_radec
    that works with any of the NuSTAR time series coordinate data
    (e.g. the optical axis position).

    Conversion function to go from X/Y coordinates
    in the FITS file to RA/Dec coordinates.
    '''
    for field in evt_hdr.keys():
        if field.find('TYPE') != -1:
            if evt_hdr[field] == 'X':
                xval = field[5:8]
            if evt_hdr[field] == 'Y':
                yval = field[5:8]

    xunit = u.Unit(evt_hdr[f'TCUNI{xval}'])
    ra_ref = evt_hdr[f'TCRVL{xval}'] * xunit
    delx = evt_hdr[f'TCDLT{xval}'] * xunit / u.pix
    x0 = evt_hdr[f'TCRPX{xval}'] * u.pix

    yunit = u.Unit(evt_hdr[f'TCUNI{yval}'])
    dec_ref = evt_hdr[f'TCRVL{yval}'] * yunit
    dely = evt_hdr[f'TCDLT{yval}'] * yunit / u.pix
    y0 = evt_hdr[f'TCRPX{yval}'] * u.pix

    # Convert X and Y to RA/dec
    ra_x = ra_ref + (x - x0) * delx / np.cos(dec_ref)
    dec_y = dec_ref + (y - y0) * dely

    return ra_x, dec_y


# TODO: What should tStep be? Do we need to account for the motion/rotation of the Sun?
def radec_to_solar(times, ra, dec) -> tuple[u.Quantity, u.Quantity]:

    x, y = nustar.convert._delta_solar_skyfield(
        ra, dec, Time(times), tStep=1e6)
    x, y = x.to(u.arcsecond), y.to(u.arcsecond)

    return x, y


def make_sky_pixel_array(evt_data) -> np.ndarray:

    arr = [[0 for _ in range(1000)] for _ in range(1000)]
    for evt in evt_data:
        x, y = evt['X'], evt['Y']
        # det_arrs[det][y][x].append(evt) # Track the photon lists
        arr[y][x] = arr[y][x] + 1  # Track the counts

    return np.array(arr)


# TODO: Move this to utilities.py (or someplace).
# TODO: Copy this chunk from the updates nustar_pysolar.
def make_sunpy_array(evtdata, hdr, exp_time=0, on_time=0, shevt_xy=[0, 0]):
    '''This is the portion of the method make_sunpy_map from nustar_pysolar
    that creates the data array for the Sunpy map.

    Parameters
    ----------
    evtdata: FITS data structure
        This should be an hdu.data structure from a NuSTAR FITS file.

    hdr: FITS header containing the astrometric information

    Optional keywords

    exp_time: The exposure time (i.e. livetime, not on-time) no units. 
            If not given then taken from hdr
    on_time: The on-time (i.e. dwell) no units. 
            If not given then taken from hdr        

    norm_map: Normalise the map data by the exposure time (i.e. livetime), 
        giving map in units of DN/s. Defaults to "False" and units of DN

    shevt_xy: 2 element array of x and y arcsec shift to apply to 
        evtdata before making the map (does to nearest pixel),
        defaults to [0,0], so no shift

    Returns
    -------
    H : np.array
        The data array of NuSTAR data.
    '''

    # Parse header keywords
    for field in list(hdr.keys()):
        if field.find('TYPE') != -1:
            if hdr[field] == 'X':
                xval = field[5:8]
            if hdr[field] == 'Y':
                yval = field[5:8]

    min_x = hdr['TLMIN'+xval]
    min_y = hdr['TLMIN'+yval]
    max_x = hdr['TLMAX'+xval]
    max_y = hdr['TLMAX'+yval]

    delx = abs(hdr['TCDLT'+xval])
    dely = abs(hdr['TCDLT'+yval])

    x = evtdata['X'][:]
    y = evtdata['Y'][:]
    # Apply a shift to the data - default is 0
    x = x + round(shevt_xy[0]/delx)
    y = y + round(shevt_xy[1]/dely)

    # Get the exposure and ontimes, just a numbers not units of seconds
    if (exp_time == 0):
        exp_time = hdr['EXPOSURE']
    if (on_time == 0):
        on_time = hdr['ONTIME']

    # Use the native binning for now
    # Assume X and Y are the same size
    resample = 1.0
    scale = delx * resample * 3600
    bins = (max_x - min_x) / (resample)

    H, yedges, xedges = np.histogram2d(y, x, bins=int(
        bins), range=[[min_y, max_y], [min_x, max_x]])

    return H, yedges, xedges


# TODO: Move this to utilities.py (or someplace).
def get_observation_time(
    id_dir: str,
    fpm: str
) -> tuple[datetime.datetime, datetime.datetime]:
    '''
    Reads the first and last time from the photon list corresponding
    to the provided observation ID and FPM.

    Parameters
    ----------
    id_dir : str
        The ID directory of the observation.
    fpm : str
        The FPM, 'A' or 'B'.

    Returns
    -------
    time_range : tuple of datetime
        The time range defining the observation interval.
    '''
    id_dir = utilities.verify_path(id_dir)
    id_num = utilities.get_id_from_id_dir(id_dir)
    evt_file = utilities.EVT_FILE_PATH_FORMAT.format(
        id_dir=id_dir, id_num=id_num, fpm=fpm
    )
    evt_data, _ = utilities.get_event_data(evt_file)
    start, end, _ = utilities.characterize_frames(evt_data, 1)
    time_range = (
        time_tools.nustar_to_datetime(start),
        time_tools.nustar_to_datetime(end)
    )

    return time_range


class CoordinateTracker():

    def __init__(
        self,
        id_dir: str,
        fpm: str,
        data_path_format: str,
        data_keys: list
    ):

        id_dir = utilities.verify_path(id_dir)
        self.id_dir = id_dir
        self.id_num = utilities.get_id_from_id_dir(id_dir)
        if fpm is not None:
            self.fpm = fpm.upper()
        else:
            self.fpm = fpm
        self._data_path_format = data_path_format
        self.data_file = self._data_path_format.format(
            id_dir=self.id_dir, id_num=self.id_num, fpm=self.fpm
        )
        self.data_keys = data_keys
        self._coords = None

    @property
    def timestep(self) -> u.Quantity:
        '''Time between each timestamp.'''
        return (self.times[1] - self.times[0]).total_seconds() * u.s

    @property
    def times(self) -> Column[u.Quantity]:
        '''Timestamps for each data point.'''
        return self._coords['TIME']

    @property
    def x(self) -> u.Quantity:
        '''x coordinate data.'''
        return self._coords[self.x_key]

    @property
    def y(self) -> u.Quantity:
        '''y coordinate data.'''
        return self._coords[self.y_key]

    @property
    def x_key(self) -> str | None:
        '''Key used for the x coordinate data.'''
        for key in self.data_keys:
            if 'X' in key.upper():
                return key

    @property
    def y_key(self) -> str | None:
        '''Key used for the y coordinate data.'''
        for key in self.data_keys:
            if 'Y' in key.upper():
                return key

    @property
    def unit(self) -> str:
        '''The unit defined in the FITS header.'''
        for k in self.hdr.keys():
            if self.hdr[k] == self.data_keys[-1]:
                num = k[5:8]
                return self.hdr[f'TUNIT{num}']

    @property
    def date(self) -> str:
        '''Date in YYYYMMDD format'''
        p = utilities.parse.parse(utilities.ID_DIR_PATH_FORMAT, self.id_dir)
        return p['date']

    @property
    def observation_time(self) -> tuple[datetime.datetime, datetime.datetime]:
        '''Observation interval for the dataset.'''
        return get_observation_time(self.id_dir, self.fpm)

    def _format_pickle_path(self, out_dir: str = './', *args):
        '''Format the pickle path for saving data.'''
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        self.pickle_path = f'./{out_dir}/{self.date}_{self.id_num}'
        for a in args:
            self.pickle_path += f'_{a}'
        self.pickle_path += '.pkl'

    # TODO: Move this utilities?

    def get_field_unit(self, field: str) -> str:
        '''Returns the unit corresponding to the provided FITS header key.'''
        for k in self.hdr.keys():
            if 'TTYPE' in k and self.hdr[k] == field:
                num = k[5:8]
                return self.hdr[f'TUNIT{num}']

    def load_coordinates(self):
        '''Load coordinate data from a pickle file.'''
        if os.path.exists(self.pickle_path):
            print(f'Loading coordinates from pickle: {self.pickle_path}')
            with open(self.pickle_path, 'rb') as in_file:
                self._coords = pickle.load(in_file)
        else:
            print(f'Pickle file does not exist: {self.pickle_path}')

    def save_coordinates(self):
        '''Save coordinate data to a pickle file.'''
        if self._coords is not None:
            print(f'Saving coordinates to pickle: {self.pickle_path}')
            with open(self.pickle_path, 'wb') as in_file:
                pickle.dump(self._coords, in_file)
        else:
            print('Coordinates not yet calculated. Not saving to file.')

    def time_filter(self, time_range: tuple[datetime.datetime, datetime.datetime]):
        '''Filter the data by time.'''
        obs_inds = (self.times >= time_range[0]) & (
            self.times <= time_range[1])
        self._coords = self._coords[obs_inds]

    def read_data(
        self,
        time_range: list[datetime.datetime, datetime.datetime] = None,
        by: int = 1
    ):
        '''Read the data from the file. Filter by time_range, if specified.
        by allows skipping every n data points using numpy indexing,
        e.g. arr[::n].
        '''
        if 'TIME' in self.data_keys:
            self.data_keys.remove('TIME')
        self.data_keys.sort()
        self.data_keys.insert(0, 'TIME')

        with fits.open(self.data_file) as hdu:
            data = hdu[1].data
            self.hdr = hdu[1].header

        cols, units = [], []
        for key in self.data_keys:
            cols.append(data[key])
            units.append(u.Unit(self.get_field_unit(key).lower()))

        # Convert times to datetime.
        # cols[0] = nustar.utils.convert_nustar_time(cols[0])
        cols[0] = [time_tools.nustar_to_datetime(
            t) for t in cols[0]]
        data = QTable(cols, names=self.data_keys, units=units)
        self._coords = data[::by]
        if time_range is not None:
            self.time_filter(time_range)

    def initialize(self):
        '''Loads the coordinates from a pickle, if present.
        If pickle is not present, then this reads the data, then saves the coordinates.
        '''
        self.load_coordinates()
        if self._coords is None:
            self.read_data()
            self.save_coordinates()

    def convert_to_solar(self):
        '''Converts the X and Y coordinates to solar coordinates.'''
        evt_file = EVT_FILE_PATH_FORMAT.format(
            id_dir=self.id_dir, id_num=self.id_num, fpm=self.fpm)
        _, evt_hdr = utilities.get_event_data(evt_file)
        ra, dec = general_xy_to_radec(self.x, self.y, evt_hdr)
        x, y = radec_to_solar(self.times, ra, dec)
        self._coords[self.x_key] = x
        self._coords[self.y_key] = y

    def coordinate_scatter(
        self,
        ax: plt.Axes,
        cbax: plt.Axes | None = None,
        add_hist: bool = True,
        **set_kwargs: dict
    ) -> PathCollection:
        '''Make a scatter plot of the x,y coordinates.'''
        cmap = 'rainbow'
        norm = colors.Normalize()
        sc = ax.scatter(
            self.x, self.y, c=np.arange(0, len(self.x), 1),
            s=5, norm=norm, cmap=cmap)
        ax.set(
            xlabel=f'x-coordinate ({self.x.unit})', ylabel=f'y-coordinate ({self.y.unit})',
            xlim=[np.nanmin(self.x).value, np.nanmax(self.x).value],
            ylim=[np.nanmin(self.y).value, np.nanmax(self.y).value],
            **set_kwargs)

        (ax.get_figure()).colorbar(
            sc, cax=cbax, pad=0.01, aspect=20, label='Time step',
            orientation='horizontal', ticklocation='top')

        if add_hist:
            ax_histx = ax.inset_axes([0, 1.025, 1, 0.1], sharex=ax)
            ax_histy = ax.inset_axes([1.025, 0, 0.1, 1], sharey=ax)
            ax_histx.tick_params(axis='x', labelbottom=False)
            ax_histy.tick_params(axis='y', labelleft=False)

            # TODO: Change this to specify number of bins.
            # binwidth = 0.1
            # xymax = max(np.max(np.abs(self.x.value)), np.max(np.abs(self.y.value)))
            # lim = (int(xymax/binwidth) + 1) * binwidth
            # bins = np.arange(-lim, lim + binwidth, binwidth)

            bins = 100
            ax_histx.hist(
                self.x.value, bins=bins, color='black')
            ax_histy.hist(
                self.y.value, bins=bins, color='black', orientation='horizontal')
        else:
            ax.axis('equal')

        return sc

    def coordinate_timeseries(
        self,
        ax: plt.Axes,
        which_coord: str,
        b_cpd: bool = False,
        plot_kwargs: dict | None = None,
        set_kwargs: dict | None = None
    ) -> list[plt.Line2D]:
        '''plot_kwargs specifies e.g. line color, width, etc.
        set_kwargs is for the ax.set() call, e.g. xlim, ylim, etc.
        '''
        default_plot_kwargs = {'color': 'black', 'lw': 0.75}
        default_set_kwargs = {
            'xlim': [self.times[0], self.times[-1]],
            'xlabel': 'Time',
            'ylabel': f'{which_coord.upper()} ({self.y.unit})'
        }
        if plot_kwargs is None:
            plot_kwargs = {}
        plot_kwargs = {**default_plot_kwargs, **plot_kwargs}
        if set_kwargs is None:
            set_kwargs = {}
        set_kwargs = {**default_set_kwargs, **set_kwargs}

        which_coord = which_coord.lower()
        match which_coord:
            case 'x':
                coord = self.x
            case 'y':
                coord = self.y
            case 'z':
                coord = self.z

        line = ax.plot(self.times, coord, **plot_kwargs)
        ptools.set_x_ticks(ax)
        ptools.set_y_ticks(ax)
        ax.tick_params(axis='x', which='both', direction='in', top=True)
        ax.tick_params(axis='y', which='both', direction='in', right=True)
        ax.set(**set_kwargs)

        if b_cpd:
            # algo = rpt.Pelt(model='rbf', min_size=100).fit(x)
            algo_c = rpt.KernelCPD(kernel='linear', min_size=10).fit(coord)
            result = algo_c.predict(pen=100)
            for r in result[:-1]:
                ax.axvline(self.times[r], color='gray',
                           linestyle='dotted', linewidth=1)

        return line

    def coordinate_fft(self, ax: plt.Axes):
        '''Plot a fast Fourier transform on the x,y data.'''
        N = self.x.size
        tf = fftfreq(N, (self.timestep << u.s).value)[:N//2]
        # Remove nans.
        x = self.x.astype(float)
        x = x[~np.isnan(x)]
        y = self.y.astype(float)
        y = y[~np.isnan(y)]
        xf = fft(x)
        yf = fft(y)
        ax.plot(
            tf, 2.0/N * np.abs(xf[0:N//2]), linewidth=0.75,
            color='darkorange', label='x-coord fft')
        ax.plot(
            tf, 2.0/N * np.abs(yf[0:N//2]), linewidth=0.75,
            color='purple', label='y-coord fft')
        ax.set(
            xlabel='Frequency', ylabel='Power',
            xscale='log', yscale='log')
        ax.grid(True)
        ax.xaxis.grid(which='minor', alpha=0.2)
        ax.legend()

    def make_overview(
        self,
        suptitle: str = 'Coordinate Tracker Overview',
        add_fft: bool = True
    ) -> plt.Figure:

        gridspec_kwargs = {
            'nrows': 4, 'ncols': 1,
            'height_ratios': [0.25, 5, 1, 1],
            'left': 0.1, 'right': 0.9,
            'bottom': 0.1, 'top': 0.9,
            'wspace': 0.005, 'hspace': 0.025
        }
        if add_fft:
            gridspec_kwargs['nrows'] = 5
            gridspec_kwargs['height_ratios'] = [0.25, 5, 1, 1, 3]
        fig = plt.figure(layout='constrained')
        fig.suptitle(suptitle)
        gs = fig.add_gridspec(**gridspec_kwargs)
        shared_ax = None

        cbax0 = fig.add_subplot(gs[0, 0])
        ax0 = fig.add_subplot(gs[1, 0])
        self.coordinate_scatter(ax0, cbax=cbax0, add_hist=False)

        ax1 = fig.add_subplot(gs[2, 0], sharex=shared_ax)
        self.coordinate_timeseries(
            ax1, 'x', set_kwargs={'xlabel': '', 'xticklabels': []})
        # shared_ax = ax1

        ax2 = fig.add_subplot(gs[3, 0], sharex=shared_ax)
        self.coordinate_timeseries(ax2, 'y', set_kwargs={'xlabel': ''})

        if add_fft:
            ax3 = fig.add_subplot(gs[4, 0])
            self.coordinate_fft(ax3)

        return fig


class AttitudeTracker(CoordinateTracker):

    def __init__(self, id_dir: str):
        keys = ['TIME', 'POINTING']
        super().__init__(id_dir, None, ATT_FILE_PATH_FORMAT, keys)

    @property
    def x(self) -> u.Quantity:
        return self._coords['POINTING'][:, 0]

    @property
    def y(self) -> u.Quantity:
        return self._coords['POINTING'][:, 1]

    # This quantity is the pointing angle of the spacecraft.

    @property
    def z(self) -> u.Quantity:
        return self._coords['POINTING'][:, 2]

    @property
    def observation_time(self) -> tuple[datetime.datetime, datetime.datetime]:
        return get_observation_time(self.id_dir, 'A')

    @property
    def nominal_coordinate(self) -> tuple[float, float]:

        # coords = ['RA_NOM', 'DEC_NOM']
        # units = []
        # for coord in coords:
        #     comment = self.hdr.comments[coord].lower()
        #     if 'deg' in comment:
        #         units.append(u.deg)
        #     elif 'arcmin' in comment:
        #         units.append(u.arcmin)
        #     elif 'arcsec' in comment:
        #         units.append(u.arcsec)

        return (self.hdr['RA_NOM'], self.hdr['DEC_NOM'])

    # TODO: Figure out why it's saving self._coords in deg instead of arcsec.

    def convert_to_solar(self):
        '''Convert pointing coordinates to solar coordinate frame.'''
        x, y = radec_to_solar(self.times, self.x, self.y)
        self._coords['POINTING'][:, 0] = x
        self._coords['POINTING'][:, 1] = y

    def coordinate_timeseries(
        self,
        ax: plt.Axes,
        which_coord: str,
        plot_kwargs: dict | None = None,
        set_kwargs: dict | None = None
    ) -> list[plt.Line2D]:
        '''plot_kwargs specifies e.g. line color, width, etc.
        set_kwargs is for the ax.set() call, e.g. xlim, ylim, etc.
        which_coord specifies the X, Y, and Z coordinate, which denote
        the RA, Dec, and Angle of the attitude.
        '''
        which_coord = which_coord.lower()
        match which_coord:
            case 'x':
                label = 'RA'
            case 'y':
                label = 'Dec'
            case 'z':
                label = 'Angle'
            case _:
                label = 'coord'
        default_set_kwargs = {
            'xlabel': 'Time',
            'ylabel': f'{label} ({self.y.unit})'
        }
        if set_kwargs is None:
            set_kwargs = {}
        set_kwargs = {**default_set_kwargs, **set_kwargs}
        return super().coordinate_timeseries(
            ax, which_coord, plot_kwargs=plot_kwargs, set_kwargs=set_kwargs)

    def make_overview(
        self,
        suptitle: str = 'Attitude Tracker Overview',
        add_fft: bool = True
    ) -> plt.Figure:

        gridspec_kwargs = {
            'nrows': 5, 'ncols': 1,
            'height_ratios': [0.25, 5, 1, 1, 1],
            'left': 0.1, 'right': 0.9,
            'bottom': 0.1, 'top': 0.9,
            'wspace': 0.005, 'hspace': 0.025
        }
        if add_fft:
            gridspec_kwargs['nrows'] = 6
            gridspec_kwargs['height_ratios'] = [0.05, 5, 1, 1, 1, 3]

        fig = plt.figure(layout='constrained')
        fig.suptitle(suptitle)
        gs = fig.add_gridspec(**gridspec_kwargs)
        shared_ax = None

        cbax0 = fig.add_subplot(gs[0, 0])
        ax0 = fig.add_subplot(gs[1, 0])
        self.coordinate_scatter(ax0, cbax=cbax0, add_hist=False)

        ax1 = fig.add_subplot(gs[2, 0], sharex=shared_ax)
        self.coordinate_timeseries(
            ax1, 'x', set_kwargs={'xlabel': '', 'xticklabels': []})

        ax2 = fig.add_subplot(gs[3, 0], sharex=shared_ax)
        self.coordinate_timeseries(
            ax2, 'y', set_kwargs={'xlabel': '', 'xticklabels': []})

        ax3 = fig.add_subplot(gs[4, 0], sharex=shared_ax)
        self.coordinate_timeseries(ax3, 'z', set_kwargs={'xlabel': ''})

        if add_fft:
            ax4 = fig.add_subplot(gs[5, 0])
            self.coordinate_fft(ax4)

        return fig


class MastTracker(CoordinateTracker):

    def __init__(self, id_dir: str):
        keys = ['TIME', 'T_FBOB']
        super().__init__(id_dir, None, MAST_FILE_PATH_FORMAT, keys)

    @property
    def x(self) -> u.Quantity:
        return self._coords['T_FBOB'][:, 0]

    @property
    def y(self) -> u.Quantity:
        return self._coords['T_FBOB'][:, 1]


class OpticalAxisTracker(CoordinateTracker):

    def __init__(self, id_dir: str, fpm: str):
        keys = ['TIME', 'X_OA', 'Y_OA']
        super().__init__(id_dir, fpm, OA_FILE_PATH_FORMAT, keys)


class ApertureStopTracker(CoordinateTracker):

    def __init__(self, id_dir: str, fpm: str):
        keys = ['TIME', 'X_APSTOP', 'Y_APSTOP']
        super().__init__(id_dir, fpm, OA_FILE_PATH_FORMAT, keys)


class Det2ApertureStopTracker(CoordinateTracker):

    def __init__(self, id_dir: str, fpm: str):
        keys = ['TIME', 'DET2X_APSTOP', 'DET2Y_APSTOP']
        super().__init__(id_dir, fpm, OA_FILE_PATH_FORMAT, keys)


class Det1Tracker(CoordinateTracker):

    def __init__(self, id_dir: str, fpm: str):
        keys = ['TIME', 'X_DET1', 'Y_DET1']
        super().__init__(id_dir, fpm, DET1_FILE_PATH_FORMAT, keys)

    def read_data(self, time_range: list = None):
        super().read_data(time_range=time_range)


class CentroidTracker(CoordinateTracker):
    '''Contains data for tracking the coordinate centroid over time.'''

    def __init__(
        self,
        id_dir: str,
        fpm: str,
        frame_length: int,
        which_pixels: str,
        out_dir: str = 'centroid_coordinates'
    ):
        '''which_pixels is either RAW, SKY, or SOL'''

        self.which_pixels = which_pixels.upper()
        if self.which_pixels == 'RAW' or self.which_pixels == 'SOL':
            keys = ['TIME', 'RAWX', 'RAWY']
            file_format = utilities.EVT_FILE_PATH_FORMAT
        elif self.which_pixels == 'SKY':
            keys = ['TIME', 'X', 'Y']
            file_format = '{id_dir}event_cl/nu{id_num}{fpm}06_cl.evt'
        else:
            raise ValueError(
                'Attribute \'which_pixels\' must be either \'RAW\', \'SKY\, or \'SOL\'')

        super().__init__(id_dir, fpm, file_format, keys)
        self.frame_length = frame_length
        self.out_dir = out_dir
        self._format_pickle_path(
            out_dir, f'fpm{self.fpm}', f'{self.frame_length}s', self.which_pixels, 'centroid')

    def make_overview(self, add_fft: bool = True):
        super().make_overview(
            f'Centroid {self.which_pixels} Coordinate Overview', add_fft=add_fft)

    def read_data(self, time_range: tuple = None):

        evt_data, hdr = utilities.get_event_data(self.data_file)
        start, end, num_frames = utilities.characterize_frames(
            evt_data, self.frame_length)
        start_dt = time_tools.nustar_to_datetime(start)
        end_dt = time_tools.nustar_to_datetime(end)

        # TODO: Test if this works.
        if time_range is not None:
            if time_range[0] > start_dt and time_range[0] < end_dt:
                start = (
                    time_range[0] - time_tools.get_reference_time()).total_seconds()
            if time_range[1] > start_dt and time_range[1] < end_dt:
                end = (time_range[1] -
                       time_tools.get_reference_time()).total_seconds()

        times, x, y = [], [], []
        time_edges = np.arange(start, end+self.frame_length, self.frame_length)
        for i in range(num_frames):
            start = time_tools.nustar_to_datetime(time_edges[i])
            mid_time = start + datetime.timedelta(seconds=self.frame_length/2)
            end = time_tools.nustar_to_datetime(time_edges[i+1])
            frame_data = evt_data[nustar.filter.by_time(evt_data, hdr,
                                                        Time((start, end), format='datetime', scale='utc'))]

            # Skip the SAA passings and data gaps.
            if len(frame_data) > 0:
                if self.which_pixels == 'RAW':
                    rp_arr = pa.RawPixelArray(frame_data, hdr)
                    det_arrs = rp_arr.make_det_counts()
                    arr = pa.merge_det_arrs(det_arrs)
                elif self.which_pixels == 'SKY':
                    arr = make_sky_pixel_array(frame_data)
                elif self.which_pixels == 'SOL':
                    arr, _, _ = make_sunpy_array(frame_data, hdr)
                com_x, com_y = list(photutils.centroids.centroid_com(arr))
            else:
                com_x, com_y = [np.nan, np.nan]

            times.append(mid_time)
            x.append(com_x)
            y.append(com_y)

        units = [u.s] + [u.pix]*(len(self.data_keys)-1)
        self._coords = QTable([times, x, y], names=self.data_keys, units=units)

    def plot_centroid(self, time_limits: tuple[str, str] = None, b_cpd: bool = False):

        if time_limits is None:
            xlim = [self.times[0], self.times[-1]]
        else:
            xlim = [Time(t, scale='utc').datetime for t in time_limits]
            # xlim = [utilities.convert_string_to_datetime(
            #     t) for t in time_limits]

        labels = [
            f'{self.which_pixels} pixel x-coordinate ',
            f'{self.which_pixels} pixel y-coordinate ',
            'distance'
        ]
        shift_colors = ['red', 'blue', 'purple']

        fig, ax = plt.subplots(2, 1, figsize=(12, 12))
        for i, a in enumerate(ax):
            coords = self._coords[:, i+1]
            a.scatter(self.times, coords, color='black', s=2)
            a.set(
                xlabel='Time',
                ylabel=f'centroid {labels[i]}',
                title=f'Evolution of centroid {labels[i]}',
                xlim=xlim
            )
            ptools.set_x_ticks(a, self.times[0], self.times[-1])

            if b_cpd:
                algo = rpt.Pelt(model='rbf').fit(coords)
                result = algo.predict(pen=5)
                for r in result[:-1]:
                    a.axvline(self.times[r], color=shift_colors[i],
                              linestyle='dotted', linewidth=1)

        fig.tight_layout()
        ptools.save_plot(fig, './', 'centroid_coordinates')

        self.fig, self.ax = fig, ax


class TrackerAnimation():

    def __init__(self, tracker: CoordinateTracker):

        self.tracker = tracker
        self.num_frames = len(self.tracker.times)
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set(xlabel='x-coordinate', ylabel='y-coordinate')
        self.ax.axis('equal')

        self.sc = self.ax.scatter([], [],
                                  )
        self.ax.set(xlim=(np.min(self.tracker.x)*.9, np.max(self.tracker.x)*1.1),
                    ylim=(np.min(self.tracker.y)*.9,
                          np.max(self.tracker.y)*1.1)
                    )
        self.xdata, self.ydata, self.colors = [], [], []
        self.sc.set_clim(0, self.num_frames)
        # plt.colorbar(self.sc, ax=self.ax, cmap=cmap, pad=0.01, aspect=100, label='Time step')

        self.cmap = plt.get_cmap('rainbow')
        self.cmap_values = self.cmap(np.arange(0, len(self.tracker.times), 1))
        sm = plt.cm.ScalarMappable(cmap=self.cmap)
        sm.set_clim(0, len(self.tracker.times))
        plt.colorbar(sm, ax=self.ax, pad=0.01, aspect=100, label='Time step')
        print('Number of frames:', self.num_frames)

    # def initialize_plot(self):

    #     cmap = 'rainbow'
    #     norm = colors.Normalize(vmin=0, vmax=self.num_frames)
    #     self.sc = self.ax.scatter(self.tracker.x[0], self.tracker.y[0],
    #         s=5, cmap=cmap, norm=norm)

    #     return self.sc,

    def update_plot(self, frame) -> tuple[PathCollection]:

        print(frame)
        self.xdata.append(self.tracker.x[frame])
        self.ydata.append(self.tracker.y[frame])
        self.colors.append(self.cmap_values[frame])

        self.sc.set_offsets(np.c_[self.xdata, self.ydata])
        self.sc.set_facecolors(self.colors)

        return self.sc,

    def animate(self):

        self.anim = animation.FuncAnimation(
            self.fig, self.update_plot,
            # init_func=self.initialize_plot,
            frames=self.num_frames, interval=1, blit=True)
        self.anim.save(
            'tracker_animation_test.mp4', fps=60, dpi=80,
            extra_args=['-vcodec', 'libx264'])
