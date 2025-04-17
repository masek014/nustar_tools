import gc
import math
import os
import pickle

import astropy
import astropy.units as u
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
import numpy as np
import photutils
import scipy.optimize as opt
import sunpy.map

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from pylab import text
from regions import PixCoord, CirclePixelRegion, RectangleSkyRegion, CircleSkyRegion, PixelRegion
from scipy import ndimage

from ..utils import utilities


STYLES_DIR = f'{utilities.os.path.dirname(utilities.os.path.realpath(__file__))}/styles/'


def apply_style():
    plt.style.use(f'{STYLES_DIR}map.mplstyle')


def save_map(
    fig: plt.Figure,
    fig_dir: str,
    fig_name: str,
    type: str = 'png'
):
    '''Saves the provided figure to a file.

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
    '''
    fig_dir = utilities.verify_path(fig_dir)
    fig_name = f'{fig_name}.{type}'
    fig_path = f'{fig_dir}{fig_name}'
    plt.savefig(fig_path)
    print(f'Saved map to {fig_path}')

    # Clean up memory.
    fig.clf()
    plt.close()
    gc.collect()


def get_pixel_conversion(bin_size: int) -> float:
    '''Returns the pixel to arcsecond conversion factor based on
    the given bin_size.

    Parameters
    ----------
    bin_size : int
        The size of the bins in units of raw pixels.

    Returns
    -------
    pixel_size: float
        arcseconds per pixel
    '''
    return 2.45810736 * bin_size


# TODO: Incorporate this better.
def draw_nustar_contours(map_, ax, levels, region, out_dir='./'):

    apply_style()
    bl, tr = get_subregion(
        map_,
        (region.center.Tx.value, region.center.Ty.value),
        region.radius.value
    )
    region_submap = map_.submap(bottom_left=bl, top_right=tr)
    levels = levels << u.percent
    fig, ax = plt.subplots(subplot_kw={'projection': region_submap})
    paths = region_submap.draw_contours(
        levels,
        axes=ax,
        cmap='Grays'
    ).get_paths()
    region_submap.plot(axes=ax)
    plt.savefig(os.path.join(out_dir, 'contour_map.png'))

    cdelt = map_.scale[0].to(u.arcsec/u.pix)
    areas = {}
    for i, level in enumerate(levels):
        contour = paths[i]
        if contour:
            vs = contour.vertices
            x, y = vs[:, 0], vs[:, 1]
            area = 0.5*np.sum(
                y[:-1]*np.diff(x) - x[:-1] * np.diff(y)) << u.pix**2
            area = np.abs(area) * cdelt * cdelt
            areas[level] = area
        else:
            areas[level] = np.nan
    pickle_path = os.path.join(out_dir, 'contours.pkl')
    with open(pickle_path, 'wb') as outfile:
        pickle.dump(areas, outfile)

    return areas


def rebin_data(in_map, new_size):
    '''Rebins the data to the specified rebin size.'''
    dimensions = [new_size, new_size] * u.pixel

    return in_map.superpixel(dimensions)


def apply_gaussian_blur(
    in_map: sunpy.map.GenericMap,
    std_dev: float = 1.0
) -> sunpy.map.GenericMap:
    '''Applies a Gaussian blur to the provided SunPy map.

    Parameters
    ----------
    in_map : Sunpy map
        Input map to which the blur will be applied.
    std_dev : float
        The standard deviation of the function.

    Returns
    -------
    A new Sunpy map with the blur applied.
    '''
    smoothed_data = ndimage.gaussian_filter(
        in_map.data, std_dev, mode='nearest')

    return sunpy.map.Map(smoothed_data, in_map.meta)


def apply_contour(
    submap: sunpy.map.GenericMap,
    cmap: str,
    dmin: float,
    dmax: float
):
    '''Applies contour lines to the map.

    Parameters
    ----------
    submap : Sunpy map
        The input map.
    cmap : str
        The name of the colormap to use.
    dmin : float
        The minimum of the colorbar.
    dmax : float
        The maximum of the colorbar.
    '''

    # Make a copy of our NuSTAR map.
    submap2 = sunpy.map.Map(submap.data, submap.meta)

    # Make a colour map with all the colours the same for plotting the contours.
    cm2 = mplcolors.LinearSegmentedColormap.from_list(
        'simple', [(1, 1, 1), (1, 1, 1)], N=2)

    # Setup up the map norm and colors and contour colours.
    submap.plot_settings['cmap'] = plt.get_cmap(cmap)
    submap.plot_settings['norm'] = mplcolors.LogNorm(vmin=dmin, vmax=dmax)
    submap2.plot_settings['cmap'] = cm2

    # Map the composite map.
    comp_map = sunpy.map.Map(submap, submap2, composite=True)

    # Set the second map as the contours and set the limits.
    # comp_map.set_levels(index=1, levels=[5e-3,1e-2,1e-1])
    comp_map.set_levels(index=1, levels=[0.5, 1, 2, 10], percent=True)
    comp_map.set_alpha(index=1, alpha=0.5)
    comp_map.plot()


def apply_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    norm: object,
    cmap: object,
    **kwargs
):
    '''Adds a colorbar to the map plot.
    A new axes object is created to house the colorbar.

    Parameters
    ----------
    fig : matplotlib figure
        The figure to which to colorbar will be added.
    ax : matplotlib axes
        The axes to which the colorbar is applied.
    norm : object
        The normalization to use, e.g. matplotlib.colors.Normalize.
    cmap : object
        The colormap to use, e.g. matplotlib.pyplot.get_cmap('Grays').
    kwargs : dict
        Keyword arguments to the ColorbarBase method.

    Returns
    -------
    cbar : matplotlib colorbar
        The newly created colorbar.
    '''
    default_kwargs = {'aspect': 20, 'pad': 0.05}
    kwargs = {**default_kwargs, **kwargs}
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, **kwargs)
    cbar.ax.yaxis.set_minor_formatter('')

    return cbar


def apply_discrete_colorbar(
        fig, ax, num_segments, cb_min, cb_max,
        cmap=plt.get_cmap('jet'), width=0.05, format='%li', label=''):
    '''
    Adds a colorbar to the map plot.
    A new axes object is created to house the colorbar.
    Inspired from: https://stackoverflow.com/a/14779462

    Parameters
    ----------
    fig : matplotlib figure
        The figure to which to colorbar will be added.
    ax : matplotlib axes
        The axes to which the colorbar is applied.
    num_segments : int
        Number of segments on the colorbar.
    cb_min : float
        The colobar minimum value.
    cb_max : float
        The colobar maximum value.
    cmap : matplotlib cmap
        The colormap to use.

    Returns
    -------
    cbax : matplotlib axes
        The axes containing the colorbar.
    cb : matplotlib colorbar
        The newly created colorbar.
    '''
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'Det ID', cmaplist, cmap.N)
    ticks = np.linspace(cb_min, cb_max, num_segments)
    step = (ticks[1]-ticks[0])/2
    bounds = np.linspace(cb_min-step, cb_max+step, num_segments+1)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    cb = apply_colorbar(
        fig, ax,
        norm=norm, cmap=cmap, label=label,
        ticks=ticks, boundaries=bounds, format=format)

    return cb


def find_min_max(
    map_data: np.ndarray,
    rebin_size: int = 1,
    make_square: bool = True,
    padding: bool = True
) -> list[float, float, float, float]:
    '''
    Finds the two coordinate pairs specifying the minimum bounding
    box for the provided data. The purpose of this method is to
    "trim" the excess, zero-valued data so the resulting plot focuses
    on the area of the FOV of the instrument. The resulting
    coordinates are in arcseconds.

    Parameters
    ----------
    map_data : np array
        The input data.
    rebin_size : int
        This is the amount by which the rebinning modified the
        data, if the original (2999 x 2999) array was rebinned.
    make_square : bool
        If true, the resulting coordinates of the minimum bounding
        box are modified so that the returned coordinates form a
        square rather than a rectangle (primarily for
        aesthetic purposes).
    padding : bool
        Specifies whether padding should be added around the edges.

    Returns
    -------
    x_min
    y_min
    x_max
    y_max
    '''

    num_rows, num_cols = np.shape(map_data)
    top_index = 0
    bottom_index = num_rows
    left_index = num_cols
    right_index = 0
    found_bottom = False

    # Identify the indeces of the bounding data.
    for i, row in enumerate(map_data):
        # indeces = np.nonzero(row)[0] # Get indeces of non-zero values in the row
        indeces = np.where(row > 0)[0]  # Ignores values <= 0.
        if indeces.size != 0:
            min_index = indeces[0]
            max_index = indeces[-1]
            if not found_bottom:
                found_bottom = True
                bottom_index = i
            if top_index < i:
                top_index = i
            if left_index > min_index:
                left_index = min_index
            if right_index < max_index:
                right_index = max_index

    # Transform the coordinates from the pixel index to arcseconds with some padding.
    # The tanh function is used to determine the extra padding.
    # + 500*math.tanh(0.2*rebin_size)
    y_max = ((top_index/num_rows) * 7200 - 3600)
    # - 500*math.tanh(0.2*rebin_size)
    y_min = ((bottom_index/num_rows) * 7200 - 3600)
    # - 500*math.tanh(0.2*rebin_size)
    x_min = ((left_index/num_cols) * 7200 - 3600)
    # + 500*math.tanh(0.2*rebin_size)
    x_max = ((right_index/num_cols) * 7200 - 3600)

    if padding:
        y_max += 500*math.tanh(0.2*rebin_size)
        y_min -= 500*math.tanh(0.2*rebin_size)
        x_min -= 500*math.tanh(0.2*rebin_size)
        x_max += 500*math.tanh(0.2*rebin_size)

    # Round the coordinates so they are whole numbers.
    x_min = math.ceil(x_min)
    y_min = math.ceil(y_min)
    x_max = math.ceil(x_max)
    y_max = math.ceil(y_max)

    # Make the coordinates form a square with the same center point as the original.
    if make_square:
        horizontal_length = x_max - x_min
        vertical_length = y_max - y_min
        diff = abs(horizontal_length - vertical_length)
        half = math.ceil(diff/2)
        if horizontal_length > vertical_length:
            # It's a horizontal rectangle, so the vertical length needs to be stretched.
            y_max += half
            y_min -= half
        else:
            x_max += half
            x_min -= half

        # Sometimes the side lengths may differ by 1 unit.
        # Do one more check to make sure the side lengths are equal.
        horizontal_length = x_max - x_min
        vertical_length = y_max - y_min
        diff = abs(horizontal_length - vertical_length)
        if horizontal_length > vertical_length:
            # The horizontal length is greater, so add to vertical length
            y_max = y_max + abs(diff)
        elif vertical_length > horizontal_length:
            x_max = x_max + abs(diff)

    return x_min, y_min, x_max, y_max


def get_submap(
    map_: sunpy.map.GenericMap,
    corners: list[float, float, float, float] | None = None
) -> sunpy.map.GenericMap:
    '''Create a submap of the provided map object.
    If corners are not provided, then they will
    be defined to fit the data in the map using
    find_min_max().

    Parameters
    ----------
    map_ : Sunpy map
        The input map to be submapped.
    corners : list
        A list containing the corners of the map in units of
        arcseconds: [x_min, y_min, x_max, y_max].

    Returns
    A Sunpy map that is a submap of map_ defined around corners.
    '''
    if corners is None:
        corners = find_min_max(map_.data)
    bl = SkyCoord(
        corners[0], corners[1], unit=u.arcsec, frame=map_.coordinate_frame)
    tr = SkyCoord(
        corners[2], corners[3], unit=u.arcsec, frame=map_.coordinate_frame)

    return map_.submap(bottom_left=bl, top_right=tr)


def set_ticks(ax: plt.Axes, hide_axes: bool = False):
    '''Configure the ticks on the provided axes.

    Parameters
    ----------
    ax : matplotlib axes
        The axes to configure.
    hide_axes : bool
        Specifies whether to show or hide the axes.
    '''
    lon = ax.coords[0]
    lat = ax.coords[1]

    lon.set_axislabel('X [arcsec]')
    lat.set_axislabel('Y [arcsec]')
    lon.set_ticks(number=6, color='black')
    lat.set_ticks(number=6, color='black')
    lon.set_minor_frequency(4)
    lat.set_minor_frequency(4)

    # Coordinate grid
    lon.grid(color='grey', alpha=0.5, linestyle='solid')
    lat.grid(color='grey', alpha=0.5, linestyle='solid')

    if hide_axes:
        lon.set_axislabel('')
        lat.set_axislabel('')
        lon.set_ticks_visible(False)
        lat.set_ticks_visible(False)
        lon.set_ticklabel_visible(False)
        lat.set_ticklabel_visible(False)


def add_overlay(ax: plt.Axes):
    '''Manually plots the heliographic overlay on the provided axes.'''
    overlay = ax.get_coords_overlay('heliographic_stonyhurst')
    lon = overlay[0]
    lat = overlay[1]
    lon.set_ticks_visible(False)
    lat.set_ticks_visible(False)
    lat.set_ticklabel_visible(False)
    lon.set_ticklabel_visible(False)
    lon.coord_wrap = 180 * u.deg
    lon.set_major_formatter('dd')
    overlay.grid(color='grey', linewidth=1, linestyle='dotted')


# TODO: Add map figsize parameter?
def apply_map_settings(
        nustar_map, bin_size=1, corners=[],
        blur_size=1, b_blur=False, b_contours=False, b_colorbar=True,
        hide_axes=False, fig=None, index=111, **cb_kwargs):
    '''
    Applies the map settings that are specified in the configuration.

    Parameters
    ----------
    nustar_map : Sunpy map
        The map to which the settings will be applied.
    bin_size : int
        The size of the map pixels in units of native pixels.
    corners : list
        Limits to specify the boundaries of the map.
        This is primarily used for ensuring all frame
        maps have the same boundaries.
    blue_size : int
        The smoothing size of the Gaussian blur.
    b_blue : bool
        Specifies whether to perform a Gaussian blur.
    b_contours : bool
        Specifies whether to add contour lines.
    b_colorbar : bool
        Specifies whether to add a colorbar.
    hide_axes : bool
        Specifies whether the show or hide the axes.
    cb_kwargs : dict
        Keywords for the colorbar creation.

    Returns
    -------
    nustar_submap : Sunpy map
        The new map with the settings applied.
    fig : matplotlib figure
        Figure of the plot.
    ax : matplotlib axes
        Axes object for the plot.
    corners_list : list
        List of the limits used for the axes.
    '''

    default_kwargs = {
        'cmap': 'plasma',
        'norm': mplcolors.LogNorm(
            np.min(nustar_map.data[nustar_map.data != 0]), np.max(nustar_map.data)),
        'label': nustar_map.meta['pixlunit']
    }
    cb_kwargs = {**default_kwargs, **cb_kwargs}

    # Rebin the data if a new bin is specified
    if bin_size != 1.:
        nustar_map = rebin_data(nustar_map, bin_size)

    # Define the bounding box of the map.
    if not corners:
        corners = find_min_max(nustar_map.data, bin_size)

    x_min, y_min, x_max, y_max = corners
    bl = SkyCoord(
        x_min*u.arcsec, y_min*u.arcsec, frame=nustar_map.coordinate_frame)
    tr = SkyCoord(
        x_max*u.arcsec, y_max*u.arcsec, frame=nustar_map.coordinate_frame)
    nustar_submap = nustar_map.submap(bottom_left=bl, top_right=tr)

    # Create the figure with the applied settings
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(index, projection=nustar_submap)

    if b_blur:
        nustar_submap = apply_gaussian_blur(nustar_submap, blur_size)
    if b_contours:
        apply_contour(
            nustar_submap, cb_kwargs['cmap'], cb_kwargs['norm'].vmin, cb_kwargs['norm'].vmax)
    else:
        # Use a different, easier to see color scheme if there is no contour.
        nustar_submap.plot(norm=cb_kwargs['norm'], cmap=cb_kwargs['cmap'])

    # Draw the solar limb
    # nustar_submap.draw_limb(color='black', linewidth=1.25, linestyle='dotted', zorder=0)

    set_ticks(ax, hide_axes)
    add_overlay(ax)

    # Tweak the title
    title_obsdate = nustar_submap.meta["date-obs"][:-4]
    ax.set_title('NuSTAR ' + title_obsdate)

    if b_colorbar:
        apply_colorbar(fig, ax, **cb_kwargs)

    return nustar_submap, fig, ax, corners


def get_region_data(
    reg: PixelRegion,
    data: np.ndarray,
    fill_val: float = 0,
    full_size: bool = False
) -> np.ndarray:
    '''Get the data contained within the provided region.

    Parameters
    ----------
    reg : PixelRegion
        The bounding region.
    data : np.ndarray
        The array containing the pixel information,
        e.g. from a Sunpy map.
    fill_val : float
        The default null value in indices outside the region.
    full_size : bool
        Specifies whether the returned array, reg_data,
        is the same shape as the input array, data.
        The default is False since it is wasteful in memory.

    Returns
    -------
    reg_data : np.ndarray
        An array containing only the pixel information within
        the provided reg.
    '''
    reg_mask = reg.to_mask()
    xmin, xmax = reg_mask.bbox.ixmin, reg_mask.bbox.ixmax
    ymin, ymax = reg_mask.bbox.iymin, reg_mask.bbox.iymax
    reg_data = np.where(
        reg_mask.data == 1, data[ymin:ymax, xmin:xmax], fill_val)
    if full_size:
        a = np.full(
            shape=data.shape, fill_value=fill_val, dtype=reg_data.dtype)
        a[ymin:ymax, xmin:xmax] = reg_data
        reg_data = a

    return reg_data


def find_dets_in_region(
    det_map: sunpy.map.GenericMap,
    region: CirclePixelRegion
) -> list[int]:
    '''Determines which detectors appear within the given region.

    Parameters
    ----------
    det_map : Sunpy map
        The map containing the detector data.
    region : CirclePixelRegion
        The region of interest.

    Returns
    -------
    dets : list
        A list containing the dets that are within the region.
    '''
    det_data = get_region_data(region, det_map.data, fill_val=-1)
    dets = list(np.unique(det_data))
    dets.sort()

    return dets


def twoD_Gaussian(
    xdata_tuple: tuple,
    amplitude: float,
    xo: float,
    yo: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    offset: float
):
    '''Function for a 2D Gaussian.'''

    (x, y) = xdata_tuple
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp(- (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                                     + c*((y-yo)**2)))

    return g.ravel()


def fit_gaussian(
    map_: sunpy.map.GenericMap,
    ax: plt.Axes,
    num_contours: int = 6
):
    '''Fits a 2D Gaussian to the data contained in map_.
    Plots the results as contour lines.

    Fitting method from https://stackoverflow.com/a/21566831

    Parameters
    ----------
    map_ : Sunpy map
        The map containing the data that the Gaussian will be fit to.
    ax : matplotlib axes
        The axes corresponding to map_opj.
    num_contours : int
        The number of contours to draw around the fit.

    Returns
    -------
    None
    '''
    data = map_.data
    data = np.hstack((data, np.zeros((data.shape[0], 1))))
    x = np.linspace(0, data.shape[1], data.shape[1])
    y = np.linspace(0, data.shape[0], data.shape[0])
    x, y = np.meshgrid(x, y)
    xdata = np.vstack((x.ravel(), y.ravel()))
    # initial_guess = (3,100,100,20,40,0,10)
    com = photutils.centroids.centroid_com(data)
    initial_guess = (1, com[0], com[1], 1, 1, 1, 1)
    popt, pcov = opt.curve_fit(
        twoD_Gaussian, xdata, data.ravel(), p0=initial_guess)
    data_fitted = twoD_Gaussian(xdata, *popt)

    ratio = popt[3]/popt[4]
    text(0.5, 0.95, r'$\sigma$ ratio: {ratio:0.3f}, {1/ratio:0.3f}',
         fontsize=12, color='white', va='center', transform=ax.transAxes)
    ax.contour(x, y, data_fitted.reshape(data.shape), num_contours,
               colors='blue', alpha=0.2)


def compute_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    '''Computes the distance between two coordinates.'''
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


def compute_farthest_pixel(
    arr: np.ndarray,
    ref_x: int,
    ref_y: int
) -> tuple[int, int, float]:
    '''Find the farthest pixel from (ref_x, ref_y) that has
    nonzero data in the input array.

    Parameters
    ----------
    arr : numpy array
        The input data array.
    ref_x : int
        The x-coordinate of the reference pixel.
    ref_y : int
        The y-coordinate of the reference pixel.

    Returns
    -------
    far_x : int
        The x-coordinate of the farthest pixel.
    far_y : int
        The y-coordinate of the farthest pixel.
    far_dist : float
        The distance between the reference pixel and the farthest pixel.
    '''
    num_rows, num_cols = arr.shape
    far_x, far_y = -1, -1
    far_dist = 0
    for y in range(num_rows):
        for x in range(num_cols):
            if arr[y, x] > 0:
                dist = compute_distance(x, y, ref_x, ref_y)
                if dist > far_dist:
                    far_dist = dist
                    far_x, far_y = x, y

    return far_x, far_y, far_dist


def plot_point(
    x: float,
    y: float,
    ax: plt.Axes,
    radius: float = 2,
    **kwargs
) -> CirclePixelRegion:
    '''Plots a point on the axis at the specified pixel point.

    Parameters
    ----------
    x : float
        The pixel x-coordinate.
    y : float
        The pixel y-coordinate.
    ax : matplotlib axes object
        The axes on which the point will be plotted.
    radius : float
        The radius, in units of pixels.

    Returns
    -------
    point : CirclePixelRegion
        The astropy regions circle object representing the point.
    '''
    default_kwargs = {'facecolor': 'blue', 'edgecolor': 'blue', 'lw': 2}
    kwargs = {**default_kwargs, **kwargs}
    kwargs['facecolor'] = kwargs['edgecolor']
    kwargs['fill'] = True  # Force True, otherwise it's just a circle
    point = CirclePixelRegion(PixCoord(x, y), radius=radius)
    point.plot(ax=ax, **kwargs)

    return point


def plot_rectangle(
    map_: sunpy.map.GenericMap,
    ax: plt.Axes,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    angle: float,
    **kwargs
) -> RectangleSkyRegion:
    '''Plot a rectangle on the provided map.

    Parameters
    ----------
    map_ : Sunpy map
        The map on which the rectangle will be drawn.
    ax : matplotlib axes object
        The associated axes object.
    center_x : float
        The x-coordinate of the center, in arcseconds.
    center_y : float
        The y-coordinate of the center, in arcseconds.
    width : float
        The width of the rectangle, in arcseconds.
    height : float
        The height of the rectangle, in arcseconds.
    angle : float
        The rotation angle of the rectangle, in degrees.
        The rotation is in the anti-clockwise direction.

    Returns
    -------
    rect : RectangleSkyRegion
        The astropy regions rectangle object.
    '''
    rect = RectangleSkyRegion(
        center=SkyCoord(
            center_x, center_y, unit='arcsec', frame=map_.coordinate_frame),
        width=width * u.arcsec, height=height * u.arcsec,
        angle=angle * u.deg
    )
    rect.to_pixel(map_.wcs).plot(ax=ax, **kwargs)

    return rect


def get_data_coords(arr: np.ndarray) -> np.ndarray:
    '''Obtain the coordinates of all nonzero data points in the provided data.

    Parameters
    ----------
    arr : np.ndarray
        The input data.

    Returns
    -------
    nonzero_points : np.ndarray
        An array where each element is a coordinate (x, y)
    '''
    nonzero_y, nonzero_x = np.nonzero(arr)
    nonzero_points = np.vstack((nonzero_x, nonzero_y)).T

    return nonzero_points


def make_sunpy_header(
    evt_data: fits.FITS_rec,
    hdr: fits.Header,
    exp_time: float = 0,
    on_time: float = 0,
    rebin_size: float = 1.0,
    norm_map: bool = True
):
    '''Make the header for a fits file.'''
    for field in list(hdr.keys()):
        if field.find('TYPE') != -1:
            if hdr[field] == 'X':
                xval = field[5:8]
            if hdr[field] == 'Y':
                yval = field[5:8]
    min_x = hdr[f'TLMIN{xval}']
    min_y = hdr[f'TLMIN{yval}']
    max_x = hdr[f'TLMAX{xval}']
    max_y = hdr[f'TLMAX{yval}']
    delx = abs(hdr[f'TCDLT{xval}'])
    dely = abs(hdr[f'TCDLT{yval}'])

    met = evt_data['TIME'][:]*u.s
    mjdref = hdr['MJDREFI']
    mid_obs_time = astropy.time.Time(mjdref*u.d+met.mean(), format='mjd')
    sta_obs_time = astropy.time.Time(mjdref*u.d+met.min(), format='mjd')

    # Get the exposure and ontimes, just a number not units of seconds
    if (exp_time == 0):
        exp_time = hdr['EXPOSURE']
    if (on_time == 0):
        on_time = hdr['ONTIME']

    # Assume X and Y are the same size
    delx = u.Quantity(delx, hdr['TCUNI14']).to(u.arcsec).value
    scale = delx * rebin_size
    bins = (max_x - min_x) / (rebin_size)

    # Normalise the data with the exposure (or live) time?
    pixluname = 'DN/s' if norm_map else 'DN'
    dict_header = {
        "DATE-OBS": sta_obs_time.iso,
        "EXPTIME": exp_time,
        "ONTIME": on_time,
        "CDELT1": scale,
        "NAXIS1": bins,
        "CRVAL1": 0.,
        "CRPIX1": bins*0.5,
        "CUNIT1": "arcsec",
        "CTYPE1": "HPLN-TAN",
        "CDELT2": scale,
        "NAXIS2": bins,
        "CRVAL2": 0.,
        "CRPIX2": bins*0.5 + 0.5,
        "CUNIT2": "arcsec",
        "CTYPE2": "HPLT-TAN",
        "PIXLUNIT": pixluname,
        "DETECTOR": "NuSTAR",
        "HGLT_OBS": sunpy.coordinates.sun.B0(mid_obs_time).value,
        "HGLN_OBS": 0,
        "RSUN_OBS": sunpy.coordinates.sun.angular_radius(mid_obs_time).value,
        "RSUN_REF": sunpy.sun.constants.radius.value,
        "DSUN_OBS": sunpy.coordinates.sun.earth_distance(mid_obs_time).to_value('m')
    }
    header = sunpy.util.MetaDict(dict_header)

    return header


def apply_counts_threshold(data_array: np.ndarray, threshold: float) -> np.ndarray:
    '''Change all values below the given threshold in the provided data array to zero.
    This is essentially a counts filter.

    Parameters
    ----------
    data_array : np array
        Data to filter.
    threshold : float
        Threshold in count/s specifying the minimum allowed value.

    Returns
    -------
    data_array : np array
        Data array with the threshold applied.
    '''
    sub_threshold_indices = data_array < threshold
    data_array[sub_threshold_indices] = 0

    return data_array


def get_subregion(
    map_: sunpy.map.GenericMap,
    center: float,
    radius: float
) -> tuple[SkyCoord, SkyCoord]:
    '''Obtain the bottom left and top right coordinates of the subregion
    specified by the provided coordinates.

    Parameters
    ----------
    map_ : Sunpy map
        The input map that the region is based on.
    center : tuple
        Coordinates for the region center, (x,y), in arcseconds.
    radius : float
        The radius of the circular region.

    Returns
    -------
    bl : SkyCoord
        The bottom left point of the subregion.
    tr : SkyCoord
        The top right point of the subregion.
    '''
    x, y = center
    bl_x, bl_y = x - radius, y - radius
    bl = SkyCoord(
        bl_x*u.arcsecond, bl_y*u.arcsecond, frame=map_.coordinate_frame)
    tr_x, tr_y = x + radius, y + radius
    tr = SkyCoord(
        tr_x*u.arcsecond, tr_y*u.arcsecond, frame=map_.coordinate_frame)

    return bl, tr


def get_sky_from_arcseconds(time, solar_pos, **kwargs):
    '''
    This is the inverse of nustar_pysolar.convert._delta_solar_skyfield().
    Convert arcsecond coordinates to sky coordinates.

    This operation inverts the sky-to-solar coordinate conversion that is
    initially performed on the NuSTAR evt file prior to the flare finding.

    Parameters
    ----------
    time : astropy time
        The time at which the coordinate needs to be converted.
    solar_pos : astropy quantity
        The (x,y) coordinate as an astropy quantity with units
        of arcseconds to be converted to a sky coordinate.
    kwargs : dict
        The keywords for time stepping and calibration files.

    Returns
    -------
    sky_pos : astropy Quantity
        The converted coordinate in units of degrees.
    '''

    from nustar_pysolar.utils import skyfield_ephem
    from sunpy.coordinates import sun

    tStep = kwargs.get('tStep', 5.0)
    tStep = tStep * u.s

    load_path = kwargs.get('load_path', None)
    observer, TheSun, ts = skyfield_ephem(
        load_path=load_path, parallax_correction=True, utc=time)
    tcheck = ts.from_astropy(time)
    astrometric = observer.at(tcheck).observe(TheSun)
    this_ra, this_dec, dist = astrometric.radec()

    # Get the center of the Sun, and assign it degrees.
    # Doing it this was is necessary to do the vector math below.
    sun_pos = np.array(
        [this_ra.to(u.deg).value, this_dec.to(u.deg).value])*u.deg
    sun_np = sun.P(time)
    rotMatrix = np.array([
        [np.cos(sun_np), np.sin(sun_np)],
        [-np.sin(sun_np),  np.cos(sun_np)]])
    delta_offset = solar_pos*[-1., 1.]
    offset = ((np.dot(delta_offset, np.linalg.inv(rotMatrix))).to(u.deg))
    sky_pos = offset + sun_pos

    return sky_pos


def get_arcsecond_coordinates(region: CircleSkyRegion) -> tuple[float, float, float]:
    '''Returns the x, y, radius coordinates of the circle.'''
    x = region.center.Tx.arcsec
    y = region.center.Ty.arcsec
    r = region.radius.value

    return x, y, r


def pixcoord_map_transform(
    map1: sunpy.map.GenericMap,
    map2: sunpy.map.GenericMap,
    x: float,
    y: float,
    map_pixcoord: PixCoord | None = None
) -> PixCoord:
    '''Convert a PixCoord from one map frame to another.

    Parameters
    ----------
    map1 : sunpy map
        The original map that contains the provided (x,y) coordinate.
    map2 : sunpy map
        The map to which the provided (x,y) coordinate will be converted to.
    x : float
        The x-coordinate in pixels.
    y : float
        The y-coordinate in pixels.
    map_pixcoord : regions.PixCoord object
        A PixCoord of the pixel coordinate in the map1 frame.

    Returns
    -------
    submap_pixcoord : regions.PixCoord object
        A PixCoord of the pixel coordinate in the map2 frame.
    '''
    if map_pixcoord is None:
        map_pixcoord = PixCoord(x, y)
    map_skycoord = map_pixcoord.to_sky(map1.wcs)
    submap_pixcoord = PixCoord.from_sky(map_skycoord, map2.wcs)

    return submap_pixcoord


def draw_circle(
    map1: sunpy.map.GenericMap,
    map2: sunpy.map.GenericMap,
    axes: plt.Axes,
    center_x: float,
    center_y: float,
    radius: float,
    plot_center: bool = True,
    **kwargs
) -> CirclePixelRegion:
    '''Plot a circle on the provided axes. The provided center coordinates are converted
    from map1 pixels to submap pixels.

    Parameters
    ----------
    map1 : Sunpy map
        The map that the center_x and center_y coordinates are from.
    submap : Sunpy map
        The map on which the circle will be drawn.
    axes : matplotlib axes object
        The axes on which the circle will be drawn.
    center_x : float
        The x-coordinate of the circle center.
    center_y : float
        The y-coordinate of the circle center.
    radius : float
        The radius of the circle in pixels.
    plot_center : bool
        Speicfy whether to plot the center point of the circle.
    kwargs
        Keyword arguments for the cosmetic features of the circle.

    Returns
    -------
    reg : CirclePixelRegion
        The circle region that was plotted.
    '''
    submap_center_pix = pixcoord_map_transform(
        map1, map2, center_x, center_y)
    reg = CirclePixelRegion(submap_center_pix, radius=radius)
    patch = reg.as_artist(**kwargs)
    axes.add_patch(patch)
    if plot_center:
        _ = plot_point(*submap_center_pix.xy, axes, 0.05, **kwargs)

    return reg


def plot_nominal_coordinate(
    evt_file: str,
    map_: sunpy.map.GenericMap,
    axes: plt.Axes,
    time: Time | None = None
) -> CirclePixelRegion:
    '''Plots the nominal coordinate on the provided map.'''
    x, y = utilities.get_nominal_coordinate(evt_file, time)
    center_pix = map_.wcs.world_to_pixel(
        SkyCoord(*(x, y), frame=map_.coordinate_frame))
    # For some reason, each coordinate is contained in an array
    center_pix = [float(c) for c in center_pix]
    point = plot_point(*center_pix, axes, 0.1, color='brown')

    return point
