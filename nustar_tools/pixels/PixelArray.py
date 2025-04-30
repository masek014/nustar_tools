'''
Contains classes and functions for fine control over the pixel data from a
NuSTAR photon event list (evt file).
'''

import os

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import nustar_pysolar as nustar
import sunpy.map

from astropy.io import fits
from astropy.table import Table, Column, vstack
from astropy.time import Time
from regions import PixCoord, SkyRegion

from ..plotting import lightcurves
from ..utils import time_tools, utilities


STYLES_DIR = os.path.dirname(
    os.path.realpath(lightcurves.__file__)) + '/styles/'


def normalize(arr: np.ndarray) -> np.ndarray:
    '''Normalizes the provided array.'''
    return arr / np.linalg.norm(arr)


def merge_det_arrs(
    det_arrs: list[np.ndarray],
    apply_normalization: bool = True
) -> np.ndarray:
    '''Combines the detector arrays into a single array.
    Each quadrant it normalized with respect to itself.
    '''
    def dummy(p):
        return p
    norm = normalize if apply_normalization else dummy
    row1 = np.hstack((norm(det_arrs[1]), norm(det_arrs[0])))
    row2 = np.hstack((norm(det_arrs[2]), norm(det_arrs[3])))
    arr = np.vstack((row1, row2))

    return arr


def get_region_pixels(map_: sunpy.map.GenericMap, reg: SkyRegion) -> np.ndarray:
    '''Returns the coordinates of the pixels within the region in an
    [n, 2] numpy array, where n is the number of rows equal to the
    number of pixels within the region. Each row is the (x,y)
    coordinate of a pixel.
    '''
    if map_.data.shape != (2999, 2999):
        print('WARNING: To get pixel values corresponding to the photon \
            list in the FITS file, you must provide full-sized NuSTAR \
            map of (2999, 2999) pixels.')
    pix_reg = reg.to_pixel(map_.wcs)
    reg_mask = pix_reg.to_mask()
    xmin, ymin = reg_mask.bbox.ixmin, reg_mask.bbox.iymin
    reg_pixels = np.array(np.where(reg_mask.data == 1)).T
    reg_pixels[:, 0] += ymin
    reg_pixels[:, 1] += xmin
    reg_pixels = np.fliplr(reg_pixels)

    return reg_pixels


class EmptyPixelArrayError(ValueError):
    '''Dummy for instances where a pixel does not have enough counts.'''


@dataclass
class Pixel:
    '''Contains coordinate and photon event data for a single pixel.'''
    coord: PixCoord
    evts: Table


class PixelArray():
    '''Robust class for manipulating data
    pertaining to the pixels in NuSTAR data.
    '''

    def __init__(
        self,
        evt_data: Table | fits.fitsrec.FITS_rec,
        hdr: fits.header.Header,
        x_key: str = 'X',
        y_key: str = 'Y',
        keep_cols: list[str] | None = None,
        map_: sunpy.map.GenericMap = None,
        region: SkyRegion = None,
        filters: dict[str, tuple] | None = None
    ):

        keep_cols = keep_cols or ('PI',)
        self.hdr = hdr
        self.x_key = x_key
        self.y_key = y_key
        self.map_ = map_
        self.region = region
        self.evt_data = Table(evt_data)
        self.evt_data.keep_columns(
            ['TIME', 'DET_ID', 'X', 'Y', 'GRADE', x_key, y_key] + list(keep_cols))

        if filters is not None:
            for col, col_range in filters.items():
                inds = (self.evt_data[col] >= col_range[0]) & (
                    self.evt_data[col] <= col_range[1])
                self.evt_data = self.evt_data[inds]
            if len(self.evt_data) < 2:
                raise EmptyPixelArrayError(
                    'Filter removed all events from the table. Please loosen the filter.')

        self.time_range = (
            time_tools.nustar_to_astropy(self.evt_data['TIME'][0]),
            time_tools.nustar_to_astropy(self.evt_data['TIME'][-1]))
        self._create_pixels()
        self._combine_evts()

    def _get_pixel_data(self) -> tuple[np.ndarray, Column, Column]:
        '''Returns the data contained within each pixel.'''
        x, y = self.evt_data[self.x_key], self.evt_data[self.y_key]
        if self.region is None:
            coords = np.array([x, y]).T
            pixel_coords = np.vstack([tuple(row) for row in coords])
            pixel_coords = np.unique(pixel_coords, axis=0)
        else:
            x, y = self.evt_data['X'], self.evt_data['Y']
            pixel_coords = get_region_pixels(self.map_, self.region)
            x_within = (x > np.min(pixel_coords[:, 0])) & (
                x < np.max(pixel_coords[:, 0]))
            y_within = (y > np.min(pixel_coords[:, 1])) & (
                y < np.max(pixel_coords[:, 1]))
            self.evt_data = self.evt_data[x_within & y_within]
            x, y = self.evt_data[self.x_key], self.evt_data[self.y_key]

        return pixel_coords, x, y

    def _create_pixels(self):
        '''Create the pixels containing the data.'''
        pixel_coords, pix_x, pix_y = self._get_pixel_data()
        self.pixels = []
        for coord in pixel_coords:
            x, y = coord
            inds = (pix_x == x) & (pix_y == y)
            if inds.any():
                evts = self.evt_data[inds]
                pixel = Pixel(PixCoord(x, y), evts)
                self.pixels.append(pixel)

    def _combine_evts(self):
        '''Combine all the pixel data into a single table.'''
        evts = []
        for pixel in self.pixels:
            evts.append(pixel.evts)
        if len(evts) < 2:
            raise EmptyPixelArrayError(
                f'PixelArray must have two or more photons (found {len(evts)} photons).')
        self.evts = vstack(evts)
        self.evts.sort('TIME')

    def make_lightcurve(
        self,
        frame_length: float,
        time_range: tuple[float, float] = None,
        energy_range: tuple[float, float] = None,
        hk_file: str = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''Compute a lightcurve for the observation.'''
        if energy_range is not None:
            energy_range = energy_range.value
        _, _, num_frames = utilities.characterize_frames(
            self.evts, frame_length)
        if hk_file is None:
            time_edges, values, values_err = lightcurves.calculate_lightcurve(
                self.evts, num_frames, time_range, energy_range)
        else:
            time_edges, values, values_err = lightcurves.calculate_lightcurve_rates(
                self.evts, hk_file, num_frames, time_range, energy_range)

        return time_edges, values, values_err

    def plot_lightcurve(
        self,
        frame_length: float,
        time_range: tuple[float, float] = None,
        energy_range: tuple[float, float] = None,
        hk_file: str = None,
        ax: plt.Axes = None,
        show_error: bool = True,
        apply_normalization: bool = False,
        **kwargs
    ) -> tuple[plt.Figure, plt.Axes]:
        '''Plot a lightcurve for the observation.'''
        default_kwargs = {
            'color': 'black',
            'ylabel': 'Counts' if hk_file is None else 'Counts/s',
            'title': 'NuSTAR Lightcurve',
            'b_title_add_date': False
        }
        kwargs = {**default_kwargs, **kwargs}
        time_edges, values, values_err = self.make_lightcurve(
            frame_length, time_range, energy_range, hk_file)
        if apply_normalization:
            norm = 1/np.nanmax(values)
            values -= np.nanmin(values)
            values = norm * values
            values_err *= norm
        if not show_error:
            values_err = None
        fig, ax, _ = lightcurves.make_lightcurve_plot(
            time_edges, values, values_err, ax=ax, **kwargs)
        if apply_normalization:
            ax.set_title(f'Normalized {ax.get_title()}')

        return fig, ax


class RawPixelArray(PixelArray):
    '''Used for tracking and plotting data pertaining to the "raw" pixels,
    i.e. the physical **detector** grid.
    '''

    def __init__(
        self,
        evt_data: Table | fits.fitsrec.FITS_rec,
        hdr: fits.header.Header,
        keep_cols: list[str] | None = None,
        map_: sunpy.map.GenericMap = None,
        region: SkyRegion = None,
        filters: dict | None = None
    ):
        super().__init__(evt_data, hdr, 'RAWX', 'RAWY', keep_cols, map_, region, filters)

    def make_det_counts(self, time_range: tuple[float, float] = None) -> np.ndarray:
        '''Make counts array, organized by detector number (i.e. 0, 1, 2, 3).'''
        if time_range is None:
            time_range = self.time_range
        dets = [0, 1, 2, 3]
        det_arrs = []
        for det in dets:
            arr = [[0 for _ in range(32)] for _ in range(32)]
            det_arrs.append(arr)
        self.evt_data.rename_column('TIME', 'time')  # Stupid...
        inds = nustar.filter.by_time(self.evt_data, self.hdr, time_range)
        self.evt_data.rename_column('time', 'TIME')
        for evt in self.evt_data[inds]:
            det = evt['DET_ID']
            x, y = evt[self.x_key], evt[self.y_key]
            # det_arrs[det][y][x].append(evt) # Track the photon lists
            det_arrs[det][y][x] = det_arrs[det][y][x] + 1  # Track the counts
        dets.reverse()
        for det in dets:
            a = np.array(det_arrs[det])
            for _ in range(np.abs(3-det)):
                a = np.rot90(a, k=1, axes=(1, 0))  # Rotate clockwise
            det_arrs[det] = a

        return np.array(det_arrs)

    def plot_det_counts(
        self,
        time_range: tuple[Time, Time] = None,
        fig: plt.Figure | None = None,
        axs: list[np.ndarray, np.ndarray, np.ndarray, np.ndarray] = None,
        cmaps: tuple[str] | None = None
    ) -> tuple[np.ndarray[plt.Axes], list[np.ndarray]]:
        '''Plot the counts in each detector pixel.
        If axs is None, a new fig will be created regardless of what
        the value passed to the fig argument is.

        If passing to cmaps, it must be an interable of four strings
        corresponding to matplotlib colormaps.
        '''
        cmaps = cmaps or ('Blues', 'Greens', 'Oranges', 'Reds')
        if time_range is None:
            time_range = self.time_range
        det_arrs = self.make_det_counts(time_range)
        mats = []
        dets = [0, 1, 2, 3]
        if axs is None:
            fig, axs = plt.subplots(
                2, 2, figsize=(4, 4),
                gridspec_kw={
                    'left': 0.05, 'right': 0.95,
                    'bottom': 0.05, 'top': 0.95,
                    'wspace': 0.01, 'hspace': 0.01
                })

        start = time_range[0].strftime('%Y-%m-%d %H:%M:%S')
        end = time_range[-1].strftime('%Y-%m-%d %H:%M:%S')
        fig.suptitle(f'{start}-{end}', fontsize=8)
        for det in dets:
            row = int(det // 2)
            # Fun way of reversing the bottom row cols :)
            col = det % 2 - ((det % 2) + (det % 2-1))*(row)
            ax = axs[row, col]
            arr = det_arrs[det]
            mat = ax.matshow(
                np.fliplr(arr), interpolation='none', cmap=cmaps[det])
            ax.set(xticks=[], yticks=[])
            ax.set_xticks(np.arange(arr.shape[1]+1)-.5, minor=True)
            ax.set_yticks(np.arange(arr.shape[0]+1)-.5, minor=True)
            ax.tick_params(which='both', length=0)
            ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
            mats.append(mat)

        return axs, mats
