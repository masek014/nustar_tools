import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os
import sunpy.map

from astropy.io import fits
from astropy.table import Table, Column, vstack
from regions import PixCoord, SkyRegion

from dataclasses import dataclass

import nustar_pysolar as nustar

from ..utils import utilities
from ..plotting import lightcurves

STYLES_DIR = os.path.dirname(os.path.realpath(lightcurves.__file__)) + '/styles/'


def normalize(arr):
    """
    Normalizes the provided array.
    """
    
    norm = np.linalg.norm(arr)

    return arr / norm


def merge_det_arrs(
    det_arrs: list[np.ndarray],
    b_normalize: bool = True
) -> np.ndarray:
    """
    Combines the detector arrays into a single array.
    Each quadrant it normalized with respect to itself.
    """

    def dummy(p):
        return p

    # TODO: Better way to do this?
    if b_normalize:
        norm = normalize
    else:
        norm = dummy
    
    row1 = np.hstack((norm(det_arrs[1]), norm(det_arrs[0])))
    row2 = np.hstack((norm(det_arrs[2]), norm(det_arrs[3])))
    arr = np.vstack((row1, row2))

    return arr


def get_region_pixels(map_: sunpy.map.GenericMap, reg: SkyRegion) -> np.ndarray:
    """
    Returns the coordinates of the pixels within the region in an
    [n, 2] numpy array, where n is the number of rows equal to the
    number of pixels within the region. Each row is the (x,y)
    coordinate of a pixel.
    """

    if map_.data.shape != (2999, 2999):
        print('WARNING: To get pixel values corresponding to the photon\
            list in the FITS file, you must provide full-sized NuSTAR \
            map of (2999, 2999) pixels.')

    pix_reg = reg.to_pixel(map_.wcs)
    reg_mask = pix_reg.to_mask()
    xmin, ymin = reg_mask.bbox.ixmin, reg_mask.bbox.iymin
    reg_pixels = np.array(np.where(reg_mask.data==1)).T
    reg_pixels[:,0] += ymin
    reg_pixels[:,1] += xmin
    reg_pixels = np.fliplr(reg_pixels)

    return reg_pixels


class EmptyPixelArrayError(ValueError): pass


@dataclass
class Pixel:
    coord: PixCoord
    evts: Table


class PixelArray():

    
    def __init__(
        self,
        evt_data: Table | fits.fitsrec.FITS_rec,
        hdr: fits.header.Header,
        x_key: str = 'X',
        y_key: str = 'Y',
        keep_cols: list[str] = ['PI'],
        map_: sunpy.map.GenericMap = None,
        region: SkyRegion = None,
        filters: dict = {}
    ):

        self.hdr = hdr
        self.x_key = x_key
        self.y_key = y_key
        self.map_ = map_
        self.region = region
        
        self.evt_data = Table(evt_data)
        self.evt_data.keep_columns(
            ['TIME', 'DET_ID', 'X', 'Y', 'GRADE', x_key, y_key] + keep_cols
        )

        for col, col_range in filters.items():
            inds = (self.evt_data[col] >= col_range[0]) & (self.evt_data[col] <= col_range[1])
            self.evt_data = self.evt_data[inds]

        if len(self.evt_data) < 2:
            raise EmptyPixelArrayError('Filter removed all events from the '
                                       'table. Please loosen the filter.')

        self.time_range = (
            utilities.convert_nustar_time_to_astropy(self.evt_data['TIME'][0]),
            utilities.convert_nustar_time_to_astropy(self.evt_data['TIME'][-1])
        )

        self._create_pixels()
        self._combine_evts()


    def _get_pixel_data(self) -> tuple[np.ndarray, Column, Column]:

        X, Y = self.evt_data[self.x_key], self.evt_data[self.y_key]
        if self.region is None:
            coords = np.array([X,Y]).T
            pixel_coords = np.vstack([tuple(row) for row in coords])
            pixel_coords = np.unique(pixel_coords, axis=0)
        else:
            X, Y = self.evt_data['X'], self.evt_data['Y']
            pixel_coords = get_region_pixels(self.map_, self.region)
            x_within = (X > np.min(pixel_coords[:,0])) & (X < np.max(pixel_coords[:,0]))
            y_within = (Y > np.min(pixel_coords[:,1])) & (Y < np.max(pixel_coords[:,1]))
            self.evt_data = self.evt_data[x_within & y_within]
            X, Y = self.evt_data[self.x_key], self.evt_data[self.y_key]
        
        return pixel_coords, X, Y
    
    
    def _create_pixels(self):

        pixel_coords, X, Y = self._get_pixel_data()
        self.pixels = []
        for coord in pixel_coords:
            x, y = coord
            inds = (X == x) & (Y == y)
            if inds.any():
                evts = self.evt_data[inds]
                pixel = Pixel(PixCoord(x, y), evts)
                self.pixels.append(pixel)


    def _combine_evts(self):

        evts = []
        for pixel in self.pixels:
            evts.append(pixel.evts)

        if len(evts) < 2:
            raise EmptyPixelArrayError(f'PixelArray must have two or more photons (found {len(evts)} photons).')

        self.evts = vstack(evts)
        self.evts.sort('TIME')


    def make_lightcurve(
        self,
        frame_length: float,
        time_range: tuple[float, float] = None,
        energy_range: tuple[float, float] = None,
        hk_file: str = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        if energy_range is not None:
            energy_range = energy_range.value

        _, _, num_frames = utilities.characterize_frames(self.evts, frame_length)

        if hk_file is None:
            time_edges, values, values_err = lightcurves.calculate_lightcurve(
                self.evts,
                num_frames,
                time_range,
                energy_range
            )
        else:
            time_edges, values, values_err = lightcurves.calculate_lightcurve_rates(
                self.evts,
                hk_file,
                num_frames,
                time_range,
                energy_range
            )

        return time_edges, values, values_err
    

    def plot_lightcurve(
        self,
        frame_length: float,
        time_range: tuple[float, float] = None,
        energy_range: tuple[float, float] = None,
        hk_file: str = None,
        ax: plt.Axes = None,
        b_show_error: bool = True,
        b_normalize: bool = False,
        **kwargs
    ) -> tuple[plt.Figure, plt.Axes]:

        default_kwargs = dict(
            color='black',
            ylabel='Counts' if hk_file is None else 'Counts/s',
            title='NuSTAR Lightcurve',
            b_title_add_date=False
        )
        kwargs = {**default_kwargs, **kwargs}

        time_edges, values, values_err = self.make_lightcurve(
            frame_length,
            time_range,
            energy_range,
            hk_file
        )

        if b_normalize:
            norm = 1/np.nanmax(values)
            values -= np.nanmin(values)
            values = norm * values
            values_err *= norm

        if not b_show_error:
            values_err = None

        fig, ax, line = lightcurves.make_lightcurve_plot(
            time_edges,
            values,
            values_err,
            ax=ax,
            **kwargs
        )

        if b_normalize:
            ax.set_title(f'Normalized {ax.get_title()}')

        return fig, ax


class RawPixelArray(PixelArray):

    
    def __init__(
            self,
            evt_data: Table | fits.fitsrec.FITS_rec,
            hdr: fits.header.Header,
            keep_cols: list[str] = ['GRADE', 'PI'],
            map_: sunpy.map.GenericMap = None,
            region: SkyRegion = None,
            filters: dict = {}
        ):
        super().__init__(evt_data, hdr, 'RAWX', 'RAWY', keep_cols, map_, region, filters)


    def make_det_counts(self, time_range: tuple[float, float] = None) -> np.ndarray:

        if time_range is None:
            time_range = self.time_range

        dets = [0, 1, 2, 3]
        det_arrs = []
        for det in dets:
            arr = [ [0 for _ in range(32)] for _ in range(32)]
            det_arrs.append(arr)

        inds = nustar.filter.by_time(self.evt_data, self.hdr, time_range)
        for evt in self.evt_data[inds]:
            det = evt['DET_ID']
            x, y = evt[self.x_key], evt[self.y_key]
            # det_arrs[det][y][x].append(evt) # Track the photon lists
            det_arrs[det][y][x] = det_arrs[det][y][x] + 1 # Track the counts

        dets.reverse()
        for det in dets:
            a = np.array(det_arrs[det])
            for _ in range(np.abs(3-det)):
                a = np.rot90(a, k=1, axes=(1,0)) # Rotate clockwise
            det_arrs[det] = a
        
        return np.array(det_arrs)
    
    
    def plot_det_counts(
        self,
        time_range: tuple[float, float] = None,
        axs: list[np.ndarray, np.ndarray, np.ndarray, np.ndarray] = None,
        cmaps=['Blues', 'Greens', 'Oranges', 'Reds']
    ) -> tuple[np.ndarray[plt.Axes], np.ndarray[np.ndarray]]:

        if time_range is None:
            time_range = self.time_range

        det_arrs = self.make_det_counts(time_range)
        mats = []
        dets = [0, 1, 2, 3]

        if axs is None:
            fig, axs = plt.subplots(2, 2,
                figsize=(4,4),
                gridspec_kw=dict(
                    left=0.05, right=0.95,
                    bottom=0.05, top=0.95,
                    wspace=0.02, hspace=0.02
                )
            )
            # Use figtext instead of suptitle since suptitle messes with the layout boundaries.
            self.fig_text = plt.figtext(0.5, 0.96, '', horizontalalignment='center', fontsize=8)

        self.fig_text.set_text(f'{time_range[0]}-{time_range[1]}')

        for det in dets:
            
            row = int(det // 2)
            col = det % 2 - ( (det%2) + (det%2-1) )*(row) # Fun way of reversing the bottom row cols :)

            ax = axs[row,col]
            arr = det_arrs[det]
            mat = ax.matshow(np.fliplr(arr),
                interpolation='none',
                cmap=cmaps[det],
                # norm=colors.LogNorm(vmin=1, vmax=1e3)
            )

            ax.set(xticks=[], yticks=[])
            ax.set_xticks(np.arange(arr.shape[1]+1)-.5, minor=True)
            ax.set_yticks(np.arange(arr.shape[0]+1)-.5, minor=True)
            ax.tick_params(which='both', length=0)
            ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)

            mats.append(mat)

        return axs, mats
