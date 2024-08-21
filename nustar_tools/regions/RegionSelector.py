import sunpy.map
import numpy as np
import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.time import Time
from astropy.coordinates import SkyCoord

from .deconvolve import deconvolve

from ..mapping import maps, exposuremaps, tools as mtools
from ..plotting import tools as ptools
from ..trackers import CoordinateTracker as ct
from ..pixels import PixelArray as pa
from ..utils import utilities


DATE_FORMAT = '%Y-%m-%d'
TIME_FORMAT = '%H:%M:%S'
LIGHTCURVE_STYLES_DIR = ptools.STYLES_DIR
MAP_STYLES_DIR = mtools.STYLES_DIR
PSF_FILE = '/home/reed/Documents/research/nustar/psf/nuA2dpsfen1_20100101v001.fits' # TODO: Remove this?

FPM_CMAPS = dict(
    A = 'Greys',
    B = 'Reds'
)


class RegionSelector():


    def __init__(
        self,
        id_dir: str,
        fpm: str,
        time_range: tuple[str, str] = None,
        energy_range: tuple[float, float] = (0,79)*u.keV,
        grade_range: tuple[int, int] = (0, 32),
        b_livetime_correction: bool = True,
        out_dir: str= None,
    ):

        if id_dir[-1] != '/':
            id_dir = id_dir + '/'
        self.id_dir = id_dir
        self.id_num = id_dir.split('/')[-2]
        self.fpm = fpm
        self.out_dir = out_dir
        self._set_data_files(b_livetime_correction)

        if time_range is None:
            self.time_range = None
        else:
            self.time_range = Time(time_range)

        self.energy_range = energy_range
        self.grade_range = grade_range
        self.pixel_array = None
        self.region = None
        self.region_label = 'no_region'

        self._read_data()
        self._make_map()
        # self._coordinate_frame = self._nustar_map.coordinate_frame


    # TODO: Implement this.
    # @property
    # def energy_range(self):
    #     return self._energy_range


    # @property.setter
    # def energy_range(self, energy_range):

    #     self._energy_range = energy_range
    #     self._energy_filter()
    #     self._make_map()
    #     self._update_region_frame()
    #     self._make_pixel_array()


    @property
    def region_evts(self):

        if self.pixel_array is not None:
            return self.pixel_array.evts
        else:
            return None


    @property
    def map_width(self):
        return self._map_corners[2] - self._map_corners[0]


    @property
    def map_height(self):
        return self._map_corners[3] - self._map_corners[1]


    @property
    def _param_str(self):

        start = self.time_range[0].strftime('%H%M%S')
        end = self.time_range[1].strftime('%H%M%S')
        
        return f'{start}-{end}_fpm{self.fpm}_{self.energy_range[0].value}-{self.energy_range[1].value}keV'


    @property
    def _nustar_times(self):
        return [utilities.convert_string_to_nustar_time(t.strftime(f'{DATE_FORMAT} {TIME_FORMAT}')) for t in self.time_range]


    def _set_data_files(self, b_livetime_correction):

        self.evt_file = utilities.EVT_FILE_PATH_FORMAT.format(
            id_dir=self.id_dir,
            id_num=self.id_num,
            fpm=self.fpm
        )
        if b_livetime_correction:
            self.hk_file = utilities.HK_FILE_PATH_FORMAT.format(
                id_dir=self.id_dir,
                id_num=self.id_num,
                fpm=self.fpm
            )
        else:
            self.hk_file = None


    def _read_data(self):

        self._full_evt_data, self.hdr = utilities.get_event_data(self.evt_file, self.time_range)
        self._energy_filter()

        if self.time_range is None:
            start = utilities.convert_nustar_time_to_astropy(self.evt_data['TIME'][0])
            end = utilities.convert_nustar_time_to_astropy(self.evt_data['TIME'][-1])
            self.time_range = (start, end)
            print('Time range not specified. Found time range:', start, end)


    def _energy_filter(self):

        inds = utilities.nustar.filter.by_energy(self._full_evt_data, *self.energy_range.value)
        self.evt_data = self._full_evt_data[inds]
        self._grade_filter()

    
    def _grade_filter(self):

        inds = (self.evt_data['GRADE'] >= self.grade_range[0]) & (self.evt_data['GRADE'] <= self.grade_range[1])
        self.evt_data = self.evt_data[inds]


    def _make_map(self):

        self._nustar_map = maps.make_nustar_map(self.evt_data, self.hdr)
        self._map_corners = list(mtools.find_min_max(self._nustar_map.data, 1)) * u.arcsec


    def _convert_position_to_fraction(self, pos):

        frac_pos = [q << u.arcsec for q in pos]
        frac_pos[0] = (frac_pos[0] - self._map_corners[0]) / self.map_width
        frac_pos[1] = (frac_pos[1] - self._map_corners[1]) / self.map_height
        frac_pos[2] = frac_pos[2] / self.map_width
        frac_pos[3] = frac_pos[3] / self.map_height

        return frac_pos


    def _fit_inset_pos(self, inset_pos, pad_mult=1.05):

        xfrac = inset_pos[0] + inset_pos[2]
        yfrac = inset_pos[1] + inset_pos[3]

        # Refit the corners if necessary.
        inset_pos = list(inset_pos)
        if inset_pos[0] < 0.1:
            width = self._map_corners[2] - self._map_corners[0]
            xdiff = 0.1 - inset_pos[0]
            self._map_corners[0] -= width * xdiff * pad_mult
            inset_pos[0] += xdiff
        if inset_pos[1] < 0.1:
            height = self._map_corners[3] - self._map_corners[1]
            ydiff = 0.1 - inset_pos[1]
            self._map_corners[1] -= height * ydiff * pad_mult
            inset_pos[1] += ydiff
        if xfrac > 0.95:
            width = self._map_corners[2] - self._map_corners[0]
            xdiff = xfrac - 0.95
            self._map_corners[2] += width * xdiff * pad_mult
            inset_pos[0] -= xdiff
        if yfrac > 0.95:
            height = self._map_corners[3] - self._map_corners[1]
            ydiff = yfrac - 0.95
            self._map_corners[3] += height * ydiff * pad_mult
            inset_pos[1] -= ydiff

        return inset_pos


    def _make_pixel_array(self):
        
        if self.region is not None:
            map_ = self._region_map
        else:
            map_ = self._nustar_map
            
        self.pixel_array = pa.PixelArray(
            self.evt_data,
            self.hdr,
            map_=map_,
            region=self.region
        )


    def _make_title(self, plot_type):

        t1 = self.time_range[0].strftime(f'{DATE_FORMAT} {TIME_FORMAT}')
        t2 = self.time_range[1].strftime(TIME_FORMAT)

        return f'{plot_type} NuSTAR FPM {self.fpm} {t1}-{t2}, '\
            f'{self.energy_range[0]} - {self.energy_range[1]}'


    def save_fig(self, fig_type):

        utilities.create_directory(self.out_dir)
        plt.savefig(f'{self.out_dir}{self.region_label}_{fig_type}_{self._param_str}.png')


    def set_energy_range(self, energy_range):

        self.energy_range = energy_range.to(u.keV)
        self._energy_filter()
        self._make_map()
        self._make_pixel_array()


    def set_region(self, region_label, RegionClass, center, **kwargs):
        """
        We need to keep track of _region_map since sky-to-pixel conversions
        cannot work for regions at the limb if the region and map have
        different observers.
        """

        self._region_map = self._nustar_map
        self.region_label = region_label
        self.region = RegionClass(
            center=SkyCoord(*center, frame=self._region_map.coordinate_frame),
            **kwargs
        )
        self._make_pixel_array()


    def plot_region(self, ax, map_, **kwargs):

        default_kwargs = dict(
            color='lightblue',
            edgecolor=None,
            ls='dashed',
            lw=4,
            fill=True,
            alpha=0.4
        )

        kwargs = {**default_kwargs, **kwargs}

        self.region.to_pixel(map_.wcs).plot(
            ax=ax,
            **kwargs
        )


    def plot_overview_map(
        self,
        fig=None,
        index=111,
        region_kw={},
        map_kw={}
    ):

        default_map_kw = dict(
            b_contours=True,
            corners=list(self._map_corners.value),
            b_colorbar=False,
            b_blur=True,
            blur_size=2,
            label='Normalized Counts', 
            norm=colors.Normalize(1, 1e3)
        )
        map_kw = {**default_map_kw, **map_kw}

        nustar_map = maps.make_nustar_map(self.evt_data, self.hdr)
        nustar_submap, fig, ax, _ = mtools.apply_map_settings(
            nustar_map,
            fig=fig,
            index=index,
            **map_kw
        )
        ax.set_title(self._make_title(''))

        if self.region is not None:
            self.plot_region(ax, nustar_submap, **region_kw)

        return fig, ax, nustar_submap
    
    
    # TODO: Add diagnostic image/figure showing the OA and source positions, the PSF array, num. iterations (etc.?).
    def plot_deconvolved_map(
        self,
        psf_file=PSF_FILE,
        iterations=10,
        fig=None,
        index=111,
        region_kw={},
        map_kw={},
    ):

        default_map_kw = dict(
            b_contours=True,
            corners=list(self._map_corners.value),
            b_colorbar=False,
            b_blur=False,
            label='Normalized Counts', 
            norm=colors.Normalize(1, 1e3)
        )
        map_kw = {**default_map_kw, **map_kw}

        # plt.style.use(f'{MAP_STYLES_DIR}/map.mplstyle')

        oa_tracker = ct.OpticalAxisTracker(self.id_dir, self.fpm)
        oa_tracker.read_data(self.time_range)
        oa_tracker.convert_to_solar()
        oa_position = (np.mean(oa_tracker.x), np.mean(oa_tracker.y))
        source_position = (self.region.center.Tx, self.region.center.Ty)

        nustar_map = maps.make_nustar_map(self.evt_data, self.hdr)
        submap = mtools.get_submap(nustar_map, map_kw['corners'])

        deconv_data, psf_array = deconvolve(submap, psf_file, source_position, oa_position, it=iterations)
        submap.data[:] = deconv_data

        submap, fig, ax, _ = mtools.apply_map_settings(
            submap,
            fig=fig,
            index=index,
            **map_kw
        )
        ax.set_title(self._make_title(f'Deconvolved {iterations} it.,\n'))
        self.plot_region(ax, submap, **region_kw)
        
        deconv_params = dict(
            oa_position=oa_position,
            source_position=source_position,
            psf_array=psf_array
        )

        return fig, ax, submap, deconv_params
        

    def plot_exposure_map(self):

        fig, ax = exposuremaps.plot_exposure_maps(
            self.pixel_array.evts,
            self.hdr,
            b_plot_fov=False,
            fig_dir=self.out_dir,
            file_name=f'{self.region_label}_exposure_{self._param_str}'
        )

        return fig, ax


    def make_lightcurve(self, frame_length):
        
        time_edges, values, values_err = self.pixel_array.make_lightcurve(
            frame_length,
            self._nustar_times,
            self.energy_range,
            self.hk_file
        )

        return time_edges, values, values_err


    def plot_lightcurve(self, frame_length, ax=None, **kwargs):
        
        default_kwargs = dict(
            title = self._make_title('')
        )

        kwargs = {**default_kwargs, **kwargs}

        fig, ax = self.pixel_array.plot_lightcurve(
            frame_length,
            self._nustar_times,
            self.energy_range,
            self.hk_file,
            ax=ax,
            **kwargs
        )

        return fig, ax


    def plot_stacked_lightcurve(self, frame_length, energy_ranges, ax=None, **kwargs):

        default_kwargs = dict(
            cmap=FPM_CMAPS[self.fpm]
        )
        kwargs = {**default_kwargs, **kwargs}

        orig_energy_range = self.energy_range
        step = 1 / (len(energy_ranges) + 1)

        cmap = plt.get_cmap(kwargs.pop('cmap'))
        colors = cmap(np.linspace(step, 1-step, len(energy_ranges)))
        colors = reversed(colors)
        
        for color, energy_range in zip(colors, energy_ranges):
            try:
                self.set_energy_range(energy_range)
            except pa.EmptyPixelArrayError as e:
                print(f'WARNING [plot_stacked_lightcurve]: {e} Skipping energy range {energy_range}')
                continue
            fig, ax = self.pixel_array.plot_lightcurve(
                frame_length,
                self._nustar_times,
                energy_range,
                self.hk_file,
                ax=ax,
                color=color,
                label=f'{energy_range[0].value} - {energy_range[1].value} keV',
                **kwargs
            )

        ax.legend()

        self.set_energy_range(orig_energy_range)

        return fig, ax


    def plot_combined(self, frame_length, energy_ranges=None, lc_kwargs={}):

        fig = plt.figure(figsize=(10,14), layout='constrained')
        gs = fig.add_gridspec(2, 1,
            height_ratios=(3, 1),
            hspace=0.0,
        )
        
        plt.style.use(f'{LIGHTCURVE_STYLES_DIR}/lightcurve.mplstyle')
        _, ax_map, _ = self.plot_overview_map(fig=fig, index=gs[0,0])
        
        ax_lc = fig.add_subplot(gs[1,0])

        if self.hk_file is None:
            lc_unit = 'Counts'
        else:
            lc_unit = 'Counts/s'
        lc_kwargs_default = dict(
            frame_length=frame_length,
            fig=fig,
            ax=ax_lc,
            title=f'Region lightcurve ({frame_length}s bins)',
            lw=2
        )
        lc_kwargs = {**lc_kwargs_default, **lc_kwargs}
        if energy_ranges is None:
            lightcurve_method = self.plot_lightcurve
        else:
            lightcurve_method = self.plot_stacked_lightcurve
            lc_kwargs['energy_ranges'] = energy_ranges
        
        lightcurve_method(**lc_kwargs)

        ax_lc.set_ylabel(lc_unit)
        ptools.set_x_ticks(ax_lc)

        return fig, ax_map, ax_lc


    def plot_inset_lightcurve(
        self,
        frame_length,
        energy_ranges: tuple[tuple] = None,
        fig: matplotlib.figure.Figure = None,
        ax: matplotlib.axes.Axes = None,
        submap: sunpy.map.Map = None,
        inset_pos: tuple[float] | list[float] = [0.4, 0.4, 0.2, 0.2],
        region_kw: dict = {}
    ):

        b_quantity = isinstance(inset_pos[0], u.Quantity)
        if b_quantity:
            orig_inset_pos = inset_pos
            inset_pos = self._convert_position_to_fraction(inset_pos)

        if fig is None:
            inset_pos = self._fit_inset_pos(inset_pos)
            fig, ax, submap = self.plot_overview_map(region_kw=region_kw)

        # TODO: See if we can clean this part up a bit.
        if b_quantity:
            inset_pos = orig_inset_pos
            bl = SkyCoord( *(inset_pos[0], inset_pos[1]), frame=submap.coordinate_frame)
            tr = SkyCoord( *(inset_pos[0]+inset_pos[2], inset_pos[1]+inset_pos[3]), frame=submap.coordinate_frame)
            bl = submap.wcs.world_to_pixel(bl)
            tr = submap.wcs.world_to_pixel(tr)
            inset_pos = (bl[0], bl[1], tr[0]-bl[0], tr[1]-bl[1])
            transform = ax.get_transform(submap.wcs)
        else:
            transform = None
            
        plt.style.use(f'{LIGHTCURVE_STYLES_DIR}/inset_lightcurve.mplstyle')
        axins = ax.inset_axes(
            inset_pos,
            transform=transform,
            facecolor='white'
        )
        
        if self.hk_file is None:
            lc_unit = 'ct'
        else:
            lc_unit = 'ct/s'
        lc_kwargs = dict(
            frame_length=frame_length,
            fig=fig,
            ax=axins,
            title=f'Region lightcurve ({lc_unit})'
        )
        if energy_ranges is None:
            lightcurve_method = self.plot_lightcurve
        else:
            lightcurve_method = self.plot_stacked_lightcurve
            lc_kwargs['energy_ranges'] = energy_ranges
        
        lightcurve_method(**lc_kwargs)
        ptools.set_x_ticks(axins)

        pad_mult = 1.05
        indic_col = (0.5, 0.5, 0.5, 0.75)

        # TODO: Make agnostic to region type.
        bottom_left = SkyCoord(
            *( (self.region.center.Tx.value,self.region.center.Ty.value)*u.arcsec - self.region.radius*pad_mult ) << u.arcsec,
            frame=submap.coordinate_frame
        )
        top_right = SkyCoord(
            *( (self.region.center.Tx.value,self.region.center.Ty.value)*u.arcsec + self.region.radius*pad_mult ) << u.arcsec,
            frame=submap.coordinate_frame
        )
        x0, y0 = bottom_left.to_pixel(submap.wcs)
        x1, y1 = top_right.to_pixel(submap.wcs)
        ax.indicate_inset(
            bounds=(x0, y0, x1 - x0, y1 - y0),
            inset_ax=axins,
            edgecolor=indic_col,
            linewidth=2,
            alpha=indic_col[-1],
            zorder=10
        )

        return fig, ax, axins, submap


    def save_lightcurve(self, frame_length):

        header = 'unix_time,counts,counts_err'

        time_edges, counts, counts_err = self.pixel_array.make_lightcurve(
            frame_length,
            self._nustar_times,
            self.energy_range,
        )

        # Convert timestamps to Unix time.
        dt_edges = [utilities.convert_nustar_time_to_datetime(t) for t in time_edges]
        unix_edges = Time(Time(dt_edges), format='unix').value

        columns = [unix_edges, counts, counts_err]

        if self.hk_file is not None:
            time_edges, rates, rates_err = self.pixel_array.make_lightcurve(
                frame_length,
                self._nustar_times,
                self.energy_range,
                self.hk_file,
            )
            header += ',count_rates,count_rates_err'
            columns += [rates, rates_err]
            
        for i in range(1, len(columns)):
            columns[i] = np.append(columns[i], np.nan)
        
        # Combine into two column array.
        arr = np.vstack(columns).T
        arr = arr[arr[:,0].argsort()] # Sort by timestamp

        np.savetxt(
            f'{self.out_dir}{self.region_label}_lightcurve_{frame_length}s_{self._param_str}.csv',
            arr,
            fmt=['%.5f']*len(columns),
            delimiter=',',
            header=header,
            comments=''
        )