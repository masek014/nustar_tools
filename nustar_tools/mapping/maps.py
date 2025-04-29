'''
Contains general mapping functions for different styles of mapping NuSTAR data.
'''

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import nustar_pysolar as nustar
import sunpy.map

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from regions import CircleSkyRegion, PixCoord, RectanglePixelRegion, SkyRegion

from . import tools as mtools
from ..utils import utilities
from ..utils.MinimumBoundingBox import MinimumBoundingBox


def make_nustar_map(
    evt_data: fits.FITS_rec,
    hdr: fits.Header,
    time_range: tuple[Time, Time] | None = None,
    normalize: bool = False
) -> sunpy.map.GenericMap:
    '''Makes a map for the provided data.

    Parameters
    ----------
    evt_data : FITS record
        The data to be plotted.
    hdr : FITS header
        The header corresponding to the evt_data.
    time_range : tuple of string or Astropy time
        The time range (start, end) to filter the data around.
    normalize: bool
        Specifies whether to normalize the map data by the
        exposure time (i.e. livetime), giving map in units of DN/s.
    '''
    filtered_data = evt_data
    if time_range is not None:
        time_range = Time(list(time_range), format='iso', scale='utc')
        filtered_data = evt_data[nustar.filter.by_time(
            evt_data, hdr, time_range)]
    # NOTE: Each data array associated with a full-sized map takes up about 71 MB of memory.
    nustar_map = nustar.map.make_sunpy(filtered_data, hdr, norm_map=normalize)

    return nustar_map


def plot_observation_map(
    evt_file: str,
    normalize: bool = False,
    fig_dir: str = './',
    file_name: str = 'observation_map',
    **cb_kwargs
) -> tuple[sunpy.map.GenericMap, plt.Figure, plt.Axes, list]:
    '''Makes a map for the provided evt file and plots
    it on a figure.

    Parameters
    ----------
    evt_file : str
        The path to the photon event file containing
        the data to be plotted.
    normalize: bool
        Specifies whether to normalise the map data by the
        exposure time (i.e. livetime), giving map in units of DN/s.
    fig_dir : str
        The directory in which the figure will be saved.
        Leave as none for no figure to be saved.
    cb_kwargs : dict
        Keywords for the colorbar creation.

    Returns
    -------
    nuster_submap : Sunpy map
        The map of the plotted data. 
    fig : matplotlib figure
        The figure corresponding to the map.
    ax : matplotlib axes
        The axes corresponding to the map.
    axes_limits : list
        Axes limits to use for the maps.
        Useful for giving all maps uniform axis limits.
    '''
    evt_data, hdr = utilities.get_event_data(evt_file)
    nustar_map = make_nustar_map(evt_data, hdr, normalize=normalize)
    nustar_submap, fig, ax, axes_limits = mtools.apply_map_settings(
        nustar_map, **cb_kwargs)
    mtools.save_map(fig, fig_dir, file_name)

    return nustar_submap, fig, ax, axes_limits


def make_sunpy_with_array(
    evt_data: fits.FITS_rec,
    hdr: fits.Header,
    data_array: np.ndarray,
    exp_time: float = 0,
    on_time: float = 0,
    rebin_size: float = 1.0,
    normalized: bool = False
) -> sunpy.map.GenericMap:
    '''This function is very similar to nustar_pysolar's 'make_sunpy'
    function with the primary difference being that this function creates
    a sunpy map using a Numpy data array instead of a FITS record.
    As such, much of the code in this function is copied and pasted
    from the 'make_sunpy()' function.

    data_array is the array of data obtained from an
    **already created** SunPy map (it doesn't have to be, but that's
    the intended purpose of this function). This function requires the
    evt_data and hdr variables so that the map of data_array has
    the same parameters (time, references coordinates, etc.)
    as its parent map. Note that the map created by this function does
    **not** use evt_data and hdr to create the map. However, evt_data
    and hdr **should** be related to data_array in some manner.

    This function was designed for the purpose of creating maps of
    boxcar averaged frames and/or the residual data obtained by
    subtracting the boxcar average from the original   .

    Parameters
    ----------
    evt_data : FITS record array
        The FITS record array from the input data file.
    hdr : FITS header
        The header object corresponding to evt_data.
    data_array : np array
        The array containing the data to be mapped.
    exp_time : float
        Exposure time.
    on_time : float
        On time.
    rebin_size : float
        Size used for rebinning the data read from the event file.
    normalized : bool
        Boolean specifying whether or not the map should be normalized.
        Note: this method does **not** perform a normalization.
        normalize should specify whether the parent map of data_array
        was normalized.

    Returns
    -------
    data_map : Sunpy map
        The map object that was created using data_array.
    '''
    sunpy_header = mtools.make_sunpy_header(
        evt_data, hdr, exp_time, on_time, rebin_size, normalized)
    data_map = sunpy.map.Map(data_array, sunpy_header)

    return data_map


def make_det_array(evt_data: fits.FITS_rec) -> np.ndarray:
    '''Makes an array tracking the most recent det
    that apppears in each pixel.

    Parameters
    ----------
    evt_data : FITS record
        The photon event list containing the data.

    Returns
    -------
    arr : np array
        The array containing the det information.
        It is structured as follows: arr[y,x] = det
        for each pixel (x,y) that appears in the grid.
        The null value (no photon event) is -1.
    '''
    arr = np.full((2999, 2999), -1)
    x, y = evt_data['X'], evt_data['Y']
    dets = evt_data['DET_ID']
    arr[y, x] = dets

    return arr


def make_det_array3d(evt_data: fits.FITS_rec) -> dict:
    '''Makes a dictionary tracking the dets that appear
    in each pixel. Tracks a list for each coordinate
    so that no data is overwritten, hence "3D".

    Parameters
    ----------
    evt_data : FITS record
        The photon event list containing the data.

    Returns
    -------
    d : dict
        The dictionary containing the det information for each pixel.
        It is keyed by the coordinate similar to an array: d[y][x],
        where the value is a list containing all the dets that appear
        in that pixel.
    '''
    d = {}
    X, Y = evt_data['X'], evt_data['Y']
    dets = evt_data['DET_ID']
    for x, y, det in zip(X, Y, dets):
        if y not in d:
            d[y] = {}
        if x not in d[y]:
            d[y][x] = []
        d[y][x].append(det)

    return d


def make_det_map(
    evt_data: fits.FITS_rec,
    hdr: fits.Header
) -> sunpy.map.GenericMap:
    '''Makes a map consisting of the det information for each pixel.
    Note that if photons from different detectors hit the same pixel,
    the most recent value is what will be contained within the pixel.

    Parameters
    ----------
    evt_data : FITS record
        The photon list containing the det information.
    hdr : FITS header
        The header corresponding to evt_data.

    Returns
    -------
    det_map : Sunpy map
        The map containing the det information for each pixel.
    '''
    det_map = make_nustar_map(evt_data, hdr)
    det_map.data[:] = make_det_array(evt_data)

    return det_map


def plot_det_map(
    evt_data: fits.FITS_rec,
    hdr: fits.Header,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    corners: list = None
) -> tuple[plt.Figure, plt.Axes, sunpy.map.GenericMap]:
    '''Makes and plots a det map.

    Parameters
    ----------
    evt_data : FITS record
        The photon list containing the det information.
    hdr : FITS header
        The header corresponding to evt_data.
    fig : matplotlib figure
        A figure on which the map will be plotted.
    ax : matplotlib axes
        Axes on which the map will be plotted.
    corners : list
        List of corners bounding the plotting area.
        [xmin, ymin, xmax, ymax]

    Returns
    -------
    fig : matplotlib figure
        The figure corresponding to the map.
    ax : matplotlib axes
        The axes corresponding to the map.
    det_submap : Sunpy map
        The map of the plotted data. 
    '''
    det_map = make_det_map(evt_data, hdr)
    det_map.data[:] += 1
    det_submap = mtools.get_submap(det_map, corners=corners)
    det_map.data[:] -= 1

    if fig is None:
        mtools.apply_style()
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection=det_submap)

    cmap = plt.get_cmap('jet')
    det_submap.plot(cmap=cmap)
    _ = det_submap.draw_limb(
        color='white', linewidth=1.25,
        linestyle='dotted', zorder=0, label='Solar disk')

    ax.set(xlabel='x [arcsec]', ylabel='y [arcsec]')
    mtools.apply_discrete_colorbar(fig, ax, 5, -1, 3, cmap=cmap, width=0.05)

    return fig, ax, det_submap


def generate_maps(id_dir: str):
    '''Generate maps using the data in the provided data ID directory.'''
    mtools.apply_style()
    fig_dir = f'{id_dir}/figures/'
    obs_id = utilities.get_id_from_id_dir(id_dir)
    plot_observation_map(
        f'{id_dir}event_cl/nu{obs_id}A06_cl_sunpos.evt',
        fig_dir=fig_dir, file_name='observation_mapA')
    plot_observation_map(
        f'{id_dir}event_cl/nu{obs_id}B06_cl_sunpos.evt',
        fig_dir=fig_dir, file_name='observation_mapB')


class FOV():
    '''This class is designed to fit and maintain a field of view around the
    provided input NuSTAR event. This class also includes methods for
    fitting a region within the FOV and plotting the FOV.
    '''

    def __init__(self, evt_data: fits.FITS_rec, hdr: fits.Header):
        '''Initialize the object with the data and the time of interest.
        The FOV is then fit accordingly.

        Parameters
        ----------
        evt_data : FITS rec array
            Data to be plotted.
        time_interval : tuple or list
            # Specify the start and end time of an interval of interest.
            The resulting FOV is fit only around data within the specified
            interval. The format is [start_time, end_time] in the format:
            'YYYY-MM-DD HH:MM:SS'.
        '''

        self.evt_data, self.hdr = evt_data, hdr
        self.data_map = make_nustar_map(evt_data, hdr)
        self.fit_fov()

    def fit_fov(self):
        '''Fit a FOV around the data map.

        The FOV is determined by the data contained by data_map.
        The coordinates and side lengths of the FOV are then
        converted to the units used by input_map. This allows
        for a rect to be made for a macropixel map.
        '''
        # Get the bounding box around the data.
        data_map = self.data_map
        data_coords = mtools.get_data_coords(data_map.data)
        bbox = MinimumBoundingBox(data_coords)
        c1, c2, c3, c4 = np.array([list(x) for x in list(bbox.corner_points)])
        rot_angle = np.degrees(bbox.unit_vector_angle)

        # Find two opposite corners to get the center point.
        corners = [c1, c2, c3, c4]
        max_dist = -1
        opposite_pair = []
        for i, c in enumerate(corners):
            for j in range(i+1, len(corners)):
                c_comp = corners[j]
                dist = mtools.compute_distance(
                    c[0], c[1], c_comp[0], c_comp[1])
                if dist > max_dist:
                    max_dist = dist
                    opposite_pair = [c, c_comp]

        # The center point in pixel coordinates.
        center_x_pix = (opposite_pair[0][0] + opposite_pair[1][0]) / 2
        center_y_pix = (opposite_pair[0][1] + opposite_pair[1][1]) / 2
        rect = RectanglePixelRegion(
            center=PixCoord(center_x_pix, center_y_pix),
            width=bbox.length_parallel,
            height=bbox.length_orthogonal,
            angle=rot_angle*u.deg)

        # Store the attribute as a RectangleSkyRegion.
        self.rect = rect.to_sky(data_map.wcs)

    def fit_coordinate_center(self, region: CircleSkyRegion) -> CircleSkyRegion:
        '''Moves the center of the provided region to the brightest pixel.'''
        reg_data = mtools.get_region_data(
            region.to_pixel(self.data_map.wcs),
            self.data_map.data, full_size=True)
        # Ensure the array does not contain only zeros.
        if reg_data.any():
            idxs = np.argsort(reg_data.ravel())[-20:]  # 20 brightest pixels
            rows, cols = idxs//reg_data.shape[0], idxs % reg_data.shape[1]
            x, y = np.mean(cols), np.mean(rows)
            com_sky = PixCoord(x, y).to_sky(self.data_map.wcs)
            new_region = CircleSkyRegion(com_sky, radius=region.radius)
            test_reg = CircleSkyRegion(
                center=new_region.center, radius=1*u.arcsec)
            if self.check_region_outside_fov(test_reg) > 0:
                print('WARNING: new center was outside FOV. Defaulting to input region')
                new_region = region  # Default to original region
        else:
            print('Submap contains only zeros. Not fitting region center.')
            new_region = region

        return new_region

    def check_region_outside_fov(self, region: SkyRegion) -> float:
        '''Check whether the given region extends beyond the FOV.

        This method uses the intersection between two regions in order
        to determine whether the FOV fully contains the provided region.
        If the FOV *does* fully contain the region, then the intersection
        and the FOV regions will have the *same* shape. Otherwise, the
        intersection region will have a larger shape in one or both
        dimensions.

        The value of the largest size difference is returned.
        This can be thought of as providing True or False.
        If the value is nonzero, then it is True that the
        provided region is outside the FOV.

        Technically, this method should work with ANY regions
        provided as parameters, but it's intended to be used
        for checking for overlap between a region and the FOV.

        Parameters
        ----------
        region : SkyRegion
            The region to test if its center is within the FOV.

        Returns
        -------
        The amount by which the region extends beyond the FOV. 
        '''
        reg_pix = region.to_pixel((self.data_map).wcs)
        fov_pix = (self.rect).to_pixel((self.data_map).wcs)
        intersection = fov_pix.intersection(reg_pix)
        intersection_mask = intersection.to_mask()
        fov_mask = fov_pix.to_mask()
        # Compute difference between the shapes in each direction.
        diff = [abs(v1 - v2) for v1, v2 in zip((fov_mask.data).shape,
                                               (intersection_mask.data).shape)]

        return np.max(diff)

    def fit_region_within_edges(
        self,
        region: CircleSkyRegion,
        minimum_radius: u.Quantity = 40 * u.arcsec
    ) -> CircleSkyRegion:
        '''Fit the inital region within the FOV by reducing the radius
        if the region extends beyond the FOV edges.

        Parameters
        ----------
        region : CircleSkyRegion
            The intial region.
        minimum_radius : u.Quantity
            The minimum allowable radius.

        Returns
        -------
        The region after fitting the FOV edges.
        '''
        reg_pix = region.to_pixel((self.data_map).wcs)
        fov_pix = (self.rect).to_pixel((self.data_map).wcs)
        intersection = fov_pix.intersection(reg_pix)
        frac = np.sum(intersection.to_mask()) / reg_pix.area
        while (frac < 0.8) and (region.radius > minimum_radius):
            region.radius -= 2 * u.arcsec
            new_pix = region.to_pixel(self.data_map.wcs)
            intersection = fov_pix.intersection(new_pix)
            frac = np.sum(intersection.to_mask()) / new_pix.area

        return region

    def fit_region_within_chipgap(
        self,
        region: CircleSkyRegion
    ) -> CircleSkyRegion:
        '''Fit the inital region within the chip gap by reducing the radius
        if the region contains events from more than one detector.

        Parameters
        ----------
        region : CircleSkyRegion
            The intial region.

        Returns
        -------
        The region after fitting within the chipgap.
        '''
        det_map = make_det_map(self.evt_data, self.hdr)
        dets = mtools.find_dets_in_region(
            det_map, region.to_pixel(det_map.wcs))
        while len(dets) > 2 and region.radius.value > 25:
            # Decrement by 2.5 arcseconds per iteration since
            # each detector pixel is about 2.5 arcseconds
            region.radius = (region.radius.value - 2.5)*u.arcsec
            dets = mtools.find_dets_in_region(
                det_map, region.to_pixel(det_map.wcs))
            if -1 in dets:
                dets.remove(-1)

        return region

    def fit_region(
        self,
        region_kwargs: dict,
        fit_coordinate_center: bool = True,
        fit_within_detector_edges: bool = True,
        fit_within_chipgap: bool = True
    ) -> SkyRegion:
        '''Fit the inital region within the chip gap by reducing the radius
        if the region contains events from more than one detector.

        Parameters
        ----------
        region : tuple
            Contains the initial region coordinate in the format:
            (center_x, center_y, radius) in arcseconds.

        Returns
        -------
        reg : tuple
            Contains the new coordinates in the format:
            (center_x, center_y, radius) in arcseconds.
        '''
        region_class = region_kwargs.pop('region_class')
        center = region_kwargs.pop('center')
        center = SkyCoord(*center, frame=self.data_map.coordinate_frame)
        region = region_class(center=center, **region_kwargs)
        if fit_coordinate_center:
            region = self.fit_coordinate_center(region)
        if fit_within_detector_edges:
            region = self.fit_region_within_edges(region)
        if fit_within_chipgap:
            region = self.fit_region_within_chipgap(region)

        return region

    def get_fov_string(self) -> str:
        '''Returns a string containing information about the FOV.'''
        x, y = self.rect.center.Tx.arcsec, self.rect.center.Ty.arcsec
        width, height = self.rect.width.value/60, self.rect.height.value/60
        angle = self.rect.angle.value
        s = f'FOV center: ({x:0.1f}\", {y:0.1f}\")\n'\
            f'FOV side length: {width:0.1f}\' by {height:0.1f}\' \n'\
            f'FOV rotation: {angle:0.1f} deg\n'

        return s

    def plot(
        self,
        input_map: sunpy.map.GenericMap,
        ax: plt.Axes,
        draw_chip_gap: bool = True,
        plot_corners: bool = False,
        plot_center: bool = False,
        **kwargs
    ):
        '''
        Plot the FOV on the provided map.

        Parameters
        ----------
        input_map : Sunpy map
            The map on which the FOV will be plotted.
        ax : matplotlib axes
            The axes on which the FOV will be plotted.
            Must be associated with input_map.
        draw_chip_gap : bool
            Specify whether to draw the chip gap.
        plot_corners : bool
            Specify whether to draw dots on the corners of the FOV .
        plot_center : bool
            Specify whether to draw a center point in the FOV.
        kwargs
            Keyword arguments for the cosmetic features of the circle.
        '''
        default_kwargs = {'edgecolor': 'purple',
                          'linestyle': 'dashed', 'lw': 1}
        kwargs = {**default_kwargs, **kwargs}
        m_size = input_map.scale[0].value
        fov_pix = self.rect.to_pixel(input_map.wcs)
        fov_pix.plot(ax=ax, **kwargs)  # Plot the FOV.
        if plot_center:
            mtools.plot_point(
                fov_pix.center.xy[0], fov_pix.center.xy[1], ax, radius=1/m_size)
        if plot_corners:
            for c in fov_pix.corners:
                mtools.plot_point(c[0], c[1], ax, radius=1/m_size)
        if draw_chip_gap:
            cg_rect1 = RectanglePixelRegion(PixCoord(*fov_pix.center.xy),
                                            10/m_size, fov_pix.height, angle=fov_pix.angle)
            cg_rect1.plot(ax=ax, facecolor='none',
                          edgecolor=kwargs['edgecolor'], linestyle='dashed', lw=1)
            cg_rect2 = RectanglePixelRegion(PixCoord(*fov_pix.center.xy),
                                            fov_pix.width, 10/m_size, angle=fov_pix.angle)
            cg_rect2.plot(ax=ax, facecolor='none',
                          edgecolor=kwargs['edgecolor'], linestyle='dashed', lw=1)
