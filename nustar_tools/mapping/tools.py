import gc
import math

import astropy
import photutils
import sunpy.map
import matplotlib
import numpy as np
import astropy.units as u
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
from pylab import cm, text
from scipy import ndimage
from astropy.time import Time
from astropy.coordinates import SkyCoord
from regions import PixCoord, CirclePixelRegion, CircleSkyRegion, RectanglePixelRegion, RectangleSkyRegion

from ..utils import utilities


STYLES_DIR = f'{utilities.os.path.dirname(utilities.os.path.realpath(__file__))}/styles/'


def verify_config(config_dict):

    checks = [('BIN_SIZE', int), ('GAUSS_BLUR_SIZE', int), ('BOXCAR_WIDTH', int)]
    utilities.verify_config_types(config_dict, checks)


def apply_style():
    plt.style.use(f'{STYLES_DIR}map.mplstyle')


def save_map(fig, fig_dir, fig_name, type='png'):
    """
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

    Ex: save_map(fig, '/Users/rbmasek/nustar/data/20150901/20110116001/figures/', 'nu20110116001A06_cl_sunpos', '.png')
        makes an image file with the path /Users/rbmasek/nustar/data/20150901/20110116001/figures/nu20110116001A06_cl_sunpos.png
    """

    fig_dir = utilities.verify_path(fig_dir)
    fig_name = fig_name + '.' + type
    fig_path = fig_dir + fig_name
    # fig.patch.set_alpha(0) # Make the border transparent
    plt.savefig(fig_path, bbox_inches='tight')
    print(f'Saved map to {fig_path}')

    # Clean up memory.
    fig.clf()
    plt.close()
    gc.collect()


def get_pixel_conversion(bin_size):
    """
    Returns the pixel to arcsecond conversion factor based on
    the given bin_size.

    Parameters
    ----------
    bin_size : int
        The size of the bins in units of raw pixels.
    """
    
    return 2.45810736 * bin_size


# TODO: Incorporate this better.
def draw_nustar_contours(map_, ax, levels, region, out_dir='./'):

        bl, tr = get_subregion(
            map_,
            (region.center.Tx.value, region.center.Ty.value),
            region.radius.value
        )
        region_submap = map_.submap(bottom_left=bl, top_right=tr)
       
        cs = region_submap.draw_contours(
            levels*u.percent,
            axes=ax,
        )
        
        cdelt = map_.scale[0].to(u.arcsec/u.pix)
        areas = ''
        for i in range(len(levels)):
            contour = cs.collections[i]
            if contour.get_paths():
                vs = contour.get_paths()[0].vertices
                x = vs[:,0]
                y = vs[:,1]
                area = 0.5*np.sum(y[:-1]*np.diff(x) - x[:-1]*np.diff(y)) * (u.pix)**2
                area = np.abs(area)*cdelt*cdelt
                # print(f'{levels[i]}%: {area.value:.02f}')
                areas += (f'{area.value:0.2f},')
            else:
                areas += 'nan,'

        areas = areas[:-1]
        # with open(f'{out_dir}contours.txt', 'w') as outfile:
            # outfile.write(areas)
        
        return areas


def rebin_data(in_map, new_size):
    """
    Rebins the data to the specified rebin size.
    """

    dimensions = u.Quantity([new_size,new_size], u.pixel)

    return in_map.superpixel(dimensions)


def apply_gaussian_blur(in_map, std_dev=1.0):
    """
    Applies a Gaussian blur to the provided SunPy map.

    Parameters
    ----------
    in_map : Sunpy map
        Input map to which the blur will be applied.
    std_dev : float
        The standard deviation of the function.
    
    Returns
    -------
    A new Sunpy map with the blur applied.
    """

    return sunpy.map.Map(ndimage.gaussian_filter(in_map.data, std_dev, mode='nearest'), in_map.meta)


def apply_contour(submap, cmap, dmin, dmax):
    """
    Applies contour lines to the map.

    Parameters
    ----------
    submap : Sunpy map
        The input map.
    dmin : float
        The minimum of the colorbar.
    dmax : float
        The maximum of the colorbar.
    """
        
    # Make a copy of our NuSTAR map.
    submap2 = sunpy.map.Map(submap.data, submap.meta)

    # Make a colour map with all the colours the same for plotting the contours.
    cm2 = mplcolors.LinearSegmentedColormap.from_list('simple', [(1,1,1),(1,1,1)], N=2)

    # Setup up the map norm and colors and contour colours.
    submap.plot_settings['cmap'] = cm.get_cmap(cmap)
    submap.plot_settings['norm'] = mplcolors.LogNorm(vmin=dmin,vmax=dmax)
    submap2.plot_settings['cmap'] = cm2

    # Map the composite map.
    comp_map = sunpy.map.Map(submap, submap2, composite=True)

    # Set the second map as the contours and set the limits.
    # comp_map.set_levels(index=1, levels=[5e-3,1e-2,1e-1])
    comp_map.set_levels(index=1, levels=[0.5,1,2,10], percent=True)
    comp_map.set_alpha(index=1, alpha=0.5)
    comp_map.plot()


def apply_colorbar(fig, ax, width=0.005, **kwargs):
    """
    Adds a colorbar to the map plot.
    A new axes object is created to house the colorbar.

    Parameters
    ----------
    fig : matplotlib figure
        The figure to which to colorbar will be added.
    ax : matplotlib axes
        The axes to which the colorbar is applied.
    width : float
        The colorbar width as a fraction of the plot width.
    kwargs : dict
        Keyword arguments to the ColorbarBase method.

    Returns
    -------
    cbax : matplotlib axes
        The axes containing the colorbar.
    cb : matplotlib colorbar
        The newly created colorbar.
    """

    default_kwargs = {
        'spacing': 'uniform'
    }
    kwargs = {**default_kwargs, **kwargs}

    cbax = ax.inset_axes([1.01, 0, width, 1])
    cb = matplotlib.colorbar.ColorbarBase(cbax, **kwargs)

    return cbax, cb


def apply_discrete_colorbar(fig, ax, num_segments, cb_min, cb_max,
    cmap=plt.cm.get_cmap('jet'), width=0.05, format='%li', label=''):
    """
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
    """

    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Det ID', cmaplist, cmap.N)
    ticks = np.linspace(cb_min, cb_max, num_segments)
    step = (ticks[1]-ticks[0])/2
    bounds = np.linspace(cb_min-step, cb_max+step, num_segments+1)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    cbax, cb = apply_colorbar(fig, ax, width=width,
        norm=norm, label=label, cmap=cmap,
        ticks=ticks, boundaries=bounds, format=format)

    return cbax, cb


def find_min_max(map_data, rebin_size=1, b_make_square=True, b_padding=True):
    """
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
    b_make_square : bool
        If true, the resulting coordinates of the minimum bounding
        box are modified so that the returned coordinates form a
        square rather than a rectangle (primarily for
        aesthetic purposes).
    b_padding : bool
        Specifies whether padding should be added around the edges.

    Returns
    -------
    x_min
    y_min
    x_max
    y_max
    """

    num_rows, num_cols = np.shape(map_data)

    top_index = 0
    bottom_index = num_rows
    left_index = num_cols
    right_index = 0
    found_bottom = False

    # Identify the indeces of the bounding data.
    for i,row in enumerate(map_data):
        # indeces = np.nonzero(row)[0] # Get indeces of non-zero values in the row
        indeces = np.where(row>0)[0] # Ignores values <= 0.
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
    y_max = ((top_index/num_rows) * 7200 - 3600)# + 500*math.tanh(0.2*rebin_size)
    y_min = ((bottom_index/num_rows) * 7200 - 3600)# - 500*math.tanh(0.2*rebin_size)
    x_min = ((left_index/num_cols) * 7200 - 3600)# - 500*math.tanh(0.2*rebin_size)
    x_max = ((right_index/num_cols) * 7200 - 3600)# + 500*math.tanh(0.2*rebin_size)

    if b_padding:
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
    if b_make_square:
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


def get_submap(map_obj, corners=None):
    """
    Create a submap of the provided map object.
    If corners are not provided, then they will
    be defined to fit the data in the map using
    find_min_max().

    Parameters
    ----------
    map_obj : Sunpy map
        The input map to be submapped.
    corners : list
        A list containing the corners of the map in units of
        arcseconds: [x_min, y_min, x_max, y_max].

    Returns
    A Sunpy map that is a submap of map_obj defined around corners.
    """
    
    if corners is None:
        corners = find_min_max(map_obj.data)
    
    bl = SkyCoord(corners[0], corners[1], unit=u.arcsec, frame=map_obj.coordinate_frame)
    tr = SkyCoord(corners[2], corners[3], unit=u.arcsec, frame=map_obj.coordinate_frame)

    return map_obj.submap(bottom_left=bl, top_right=tr)


def set_ticks(ax, b_hide_axes=False):
    """
    Configure the ticks on the provided axes.

    Parameters
    ----------
    ax : matplotlib axes
        The axes to configure.
    b_hide_axes : bool
        Specifies whether to show or hide the axes.
    """

    lon = ax.coords[0]
    lat = ax.coords[1]

    lon.set_axislabel('X [arcsec]')
    lat.set_axislabel('Y [arcsec]')
    lon.set_ticks(number=6, color='black')
    lat.set_ticks(number=6, color='black')
    lon.set_minor_frequency(4)
    lat.set_minor_frequency(4)

    # Coordinate grid
    lon.grid(color='grey', alpha=0.1, linestyle='solid')
    lat.grid(color='grey', alpha=0.1, linestyle='solid')

    if b_hide_axes:
        lon.set_axislabel('')
        lat.set_axislabel('')
        lon.set_ticks_visible(False)
        lat.set_ticks_visible(False)
        lon.set_ticklabel_visible(False)
        lat.set_ticklabel_visible(False)


def add_overlay(ax):
    """
    Manually plots the heliographic overlay on the provided axes.
    """

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
def apply_map_settings(nustar_map, bin_size=1, corners=[],
    blur_size=1, b_blur=False, b_contours=False, b_colorbar=True,
    b_hide_axes=False, fig=None, index=111, **cb_kwargs):
    """
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
    b_hide_axes : bool
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
    """

    default_kwargs = {
        'width': 0.02,
        'cmap': 'plasma',
        'norm': mplcolors.LogNorm(np.min(nustar_map.data[nustar_map.data!=0]), np.max(nustar_map.data)),
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
    bl = SkyCoord(x_min*u.arcsec, y_min*u.arcsec, frame=nustar_map.coordinate_frame)
    tr = SkyCoord(x_max*u.arcsec, y_max*u.arcsec, frame=nustar_map.coordinate_frame)
    nustar_submap = nustar_map.submap(bottom_left=bl, top_right=tr)
    
    # Create the figure with the applied settings
    if fig is None:
        fig = plt.figure(figsize=(12,12), constrained_layout=True)
    ax = fig.add_subplot(index, projection=nustar_submap)

    if b_blur:
        nustar_submap = apply_gaussian_blur(nustar_submap, blur_size)
    if b_contours:
        apply_contour(nustar_submap, cb_kwargs['cmap'], cb_kwargs['norm'].vmin, cb_kwargs['norm'].vmax)
    else:
        # Use a different, easier to see color scheme if there is no contour.
        nustar_submap.plot(norm=cb_kwargs['norm'], cmap=cb_kwargs['cmap'])
    
    # Draw the solar limb
    # nustar_submap.draw_limb(color='black', linewidth=1.25, linestyle='dotted', zorder=0)
    
    set_ticks(ax, b_hide_axes)
    add_overlay(ax)

    # Tweak the title
    title_obsdate = nustar_submap.meta["date-obs"][:-4]    
    ax.set_title('NuSTAR ' + title_obsdate)

    if b_colorbar:
        apply_colorbar(fig, ax, **cb_kwargs)

    return nustar_submap, fig, ax, corners


def get_region_data(reg, data, fill_val=0, b_full_size=False):
    """
    Get the data contained within the provided region.

    Parameters
    ----------
    reg : PixelRegion
        The bounding region.
    data : np.ndarray
        The array containing the pixel information,
        e.g. from a Sunpy map.
    fill_val : float
        The default null value in indices outside the region.
    b_full_size : bool
        Specifies whether the returned array, reg_data,
        is the same shape as the input array, data.
        The default is False since it is wasteful in memory.

    Returns
    -------
    reg_data : np.ndarray
        An array containing only the pixel information within
        the provided reg.
    """

    reg_mask = reg.to_mask()
    xmin, xmax = reg_mask.bbox.ixmin, reg_mask.bbox.ixmax
    ymin, ymax = reg_mask.bbox.iymin, reg_mask.bbox.iymax
    reg_data = np.where(reg_mask.data==1, data[ymin:ymax, xmin:xmax], fill_val)

    if b_full_size:
        a = np.full(shape=data.shape, fill_value=fill_val, dtype=reg_data.dtype)
        a[ymin:ymax, xmin:xmax] = reg_data
        reg_data = a

    return reg_data


def find_dets_in_region(det_map, region):
    """
    Determines which detectors appear within the given region.

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
    """

    det_data = get_region_data(region, det_map.data, fill_val=-1)
    dets = list(np.unique(det_data))
    dets.sort()

    return dets


def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    Function for a 2D Gaussian.
    """

    (x, y) = xdata_tuple                                                        
    xo = float(xo)                                                              
    yo = float(yo)                                                              
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)   
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)    
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)   
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)         
                        + c*((y-yo)**2)))                                   
    return g.ravel()


def fit_gaussian(map_obj, ax, num_contours=6):
    """
    Fits a 2D Gaussian to the data contained in map_obj.
    Plots the results as contour lines.

    Parameters
    ----------
    map_obj : Sunpy map
        The map containing the data that the Gaussian will be fit to.
    ax : matplotlib axes
        The axes corresponding to map_opj.
    num_contours : int
        The number of contours to draw around the fit.
    
    Returns
    -------
    None
    """

    # Fitting method: https://stackoverflow.com/a/21566831
    data = map_obj.data
    data = np.hstack((data, np.zeros((data.shape[0],1))))
    x = np.linspace(0, data.shape[1], data.shape[1])
    y = np.linspace(0, data.shape[0], data.shape[0])
    x, y = np.meshgrid(x, y)
    xdata = np.vstack((x.ravel(),y.ravel()))
    # initial_guess = (3,100,100,20,40,0,10)
    com = photutils.centroids.centroid_com(data)
    initial_guess = (1, com[0], com[1], 1, 1, 1, 1)
    popt, pcov = opt.curve_fit(twoD_Gaussian, xdata, data.ravel(), p0=initial_guess)
    data_fitted = twoD_Gaussian(xdata, *popt)

    ratio = popt[3]/popt[4]
    text(0.5, 0.95, r'$\sigma$ ratio: {ratio:0.3f}, {1/ratio:0.3f}',
        fontsize=12, color='white', va='center', transform=ax.transAxes)

    ax.contour(x, y, data_fitted.reshape(data.shape), num_contours,
        colors='blue', alpha=0.2)


def compute_distance(x1, y1, x2, y2):
    """
    Computes the distance between two coordinates.
    """

    return np.sqrt( (x1-x2)**2 + (y1-y2)**2 )


def compute_farthest_pixel(arr, ref_x, ref_y):
    """
    Find the farthest pixel from (ref_x, ref_y) that has nonzero data in the input array.

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
    """

    num_rows, num_cols = arr.shape
    far_x, far_y = -1, -1
    far_dist = 0
    for y in range(num_rows):
        for x in range(num_cols):
            if arr[y,x] > 0:
                dist = compute_distance(x, y, ref_x, ref_y)
                if dist > far_dist:
                    far_dist = dist
                    far_x, far_y = x, y

    return far_x, far_y, far_dist


def plot_point(x, y, ax, radius=2, **kwargs):
    """
    Plots a point on the axis at the specified pixel point.

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
    """

    default_kwargs = {'facecolor': 'blue', 'edgecolor': 'blue', 'lw': 2}
    kwargs = {**default_kwargs, **kwargs}
    kwargs['facecolor'] = kwargs['edgecolor']
    kwargs['fill'] = True # Force True, otherwise it's just a circle

    point = CirclePixelRegion(PixCoord(x, y), radius=radius)
    point.plot(ax=ax, **kwargs)

    return point


def plot_rectangle(map_obj, ax, center_x, center_y, width, height, angle, **kwargs):
    """
    Plot a rectangle on the provided map.

    Parameters
    ----------
    map_obj : Sunpy map
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
    """

    rect = RectangleSkyRegion(
        center=SkyCoord(center_x, center_y, unit='arcsec',frame=map_obj.coordinate_frame),
        width=width*u.arcsec, height=height*u.arcsec,
        angle=angle*u.deg
    )
    rect.to_pixel(map_obj.wcs).plot(ax=ax, **kwargs)

    return rect


def get_data_coords(arr):
    """
    Obtain the coordinates of all nonzero data points in the provided data.

    Parameters
    ----------
    arr : np.ndarray
        The input data.
    
    Returns
    -------
    nonzero_points : np.ndarray
        An array where each element is a coordinate (x, y)
    """

    nonzero_y, nonzero_x = np.nonzero(arr)
    nonzero_points = np.vstack((nonzero_x, nonzero_y)).T

    return nonzero_points


def make_sunpy_header(evt_data, hdr, exp_time=0, on_time=0, rebin_size=1.0, norm_map=True):
    """
    Make the header for a fits file.
    """

    for field in list(hdr.keys()):
        if field.find('TYPE') != -1:
            if hdr[field] == 'X':
                xval = field[5:8]
            if hdr[field] == 'Y':
                yval = field[5:8]

    min_x= hdr['TLMIN'+xval]
    min_y= hdr['TLMIN'+yval]
    max_x= hdr['TLMAX'+xval]
    max_y= hdr['TLMAX'+yval]
    delx = abs(hdr['TCDLT'+xval])
    dely = abs(hdr['TCDLT'+yval])

    met = evt_data['TIME'][:]*u.s
    mjdref=hdr['MJDREFI']

    mid_obs_time = astropy.time.Time(mjdref*u.d+met.mean(), format = 'mjd')
    sta_obs_time = astropy.time.Time(mjdref*u.d+met.min(), format = 'mjd')

    # Get the exposure and ontimes, just a number not units of seconds
    if (exp_time == 0):
        exp_time=hdr['EXPOSURE']
    if (on_time == 0):
        on_time=hdr['ONTIME']

    delx = u.Quantity(delx, hdr['TCUNI14']).to(u.arcsec).value
    # multiplier = 1
    # if delx == 0.0006828076:
    #     multiplier = 3600

    # Assume X and Y are the same size
    scale = delx * rebin_size
    bins = (max_x - min_x) / (rebin_size)

    # Normalise the data with the exposure (or live) time?
    if norm_map:
        pixluname='DN/s'
    else:
        pixluname='DN'

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
    "DETECTOR":"NuSTAR",
    "HGLT_OBS": sunpy.coordinates.sun.B0(mid_obs_time).value,
    "HGLN_OBS": 0,
    "RSUN_OBS": sunpy.coordinates.sun.angular_radius(mid_obs_time).value,
    "RSUN_REF": sunpy.sun.constants.radius.value,
    "DSUN_OBS": sunpy.coordinates.sun.earth_distance(mid_obs_time).to_value('m')
    }

    header = sunpy.util.MetaDict(dict_header)
    
    return header


def apply_counts_threshold(data_array, threshold):
    """
    Change all values below the given threshold in the provided data array to zero.
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
    """

    # https://stackoverflow.com/questions/28430904/set-numpy-array-elements-to-zero-if-they-are-above-a-specific-threshold
    sub_threshold_indices = data_array < threshold
    data_array[sub_threshold_indices] = 0

    return data_array


def get_subregion(map_obj, center, radius):
    """
    Obtain the bottom left and top right coordinates of the subregion
    specified by the provided coordinates.

    Parameters
    ----------
    map_obj : Sunpy map
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
    """

    x, y = center
    bl_x, bl_y = x - radius, y - radius
    bl = SkyCoord(bl_x*u.arcsecond, bl_y*u.arcsecond,
        frame=map_obj.coordinate_frame)

    tr_x, tr_y = x + radius, y + radius
    tr = SkyCoord(tr_x*u.arcsecond, tr_y*u.arcsecond,
        frame=map_obj.coordinate_frame)

    return bl, tr


def get_sky_from_arcseconds(time, solar_pos, **kwargs):
    """
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
    """

    from nustar_pysolar.utils import skyfield_ephem
    from sunpy.coordinates import sun

    tStep=kwargs.get('tStep', 5.0)
    tStep = tStep * u.s

    load_path = kwargs.get('load_path', None)
    observer, TheSun, ts = skyfield_ephem(load_path=load_path,
                                            parallax_correction=True,
                                            utc=time)
    tcheck = ts.from_astropy(time)
    astrometric = observer.at(tcheck).observe(TheSun)
    this_ra, this_dec, dist = astrometric.radec()

    # Get the center of the Sun, and assign it degrees.
    # Doing it this was is necessary to do the vector math below.
    sun_pos = np.array([this_ra.to(u.deg).value,
        this_dec.to(u.deg).value])*u.deg
    sun_np = sun.P(time)

    rotMatrix = np.array([[np.cos(sun_np), np.sin(sun_np)],
                        [-np.sin(sun_np),  np.cos(sun_np)]])

    delta_offset = solar_pos*[-1., 1.]
    offset = ((np.dot(delta_offset, np.linalg.inv(rotMatrix))).to(u.deg))
    sky_pos = offset + sun_pos

    return sky_pos


def pixcoord_map_transform(map1, map2, x, y, map_pixcoord=None):
    """
    Convert a PixCoord from one map frame to another.

    Parameters
    ----------
    map_obj : sunpy map
        The original map that contains the provided (x,y) coordinate.
    submap : sunpy map
        The map to which the provided (x,y) coordinate will be converted to.
    x : float
        The x-coordinate in pixels.
    y : float
        The y-coordinate in pixels.
    map_pixcoord : regions.PixCoord object
        A PixCoord of the pixel coordinate in the map_obj frame.

    Returns
    -------
    submap_pixcoord : regions.PixCoord object
        A PixCoord of the pixel coordinate in the submap frame.
    """

    if map_pixcoord is None:
        map_pixcoord = PixCoord(x, y)

    map_skycoord = map_pixcoord.to_sky(map1.wcs)
    submap_pixcoord = PixCoord.from_sky(map_skycoord, map2.wcs)

    return submap_pixcoord


def draw_circle(map_obj, submap, axes, center_x, center_y, radius, b_plot_center=True, **kwargs):
    """
    Plot a circle on the provided axes. The provided center coordinates are converted
    from map_obj pixels to submap pixels.

    Parameters
    ----------
    map_obj : Sunpy map
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
    b_plot_center : bool
        Speicfy whether to plot the center point of the circle.
    kwargs
        Keyword arguments for the cosmetic features of the circle.

    Returns
    -------
    reg : PixelCircleRegion
        The circle region that was plotted.
    """

    submap_center_pix = pixcoord_map_transform(map_obj, submap, center_x, center_y)
    reg = CirclePixelRegion(submap_center_pix, radius=radius)
    patch = reg.as_artist(**kwargs)
    axes.add_patch(patch)

    if b_plot_center:
        reg_center = plot_point(*submap_center_pix.xy, axes, 0.05, **kwargs)

    return reg


def plot_nominal_coordinate(evt_file, submap, axes, time=None):

    x, y = utilities.get_nominal_coordinate(evt_file, time)

    center_pix = submap.wcs.world_to_pixel(SkyCoord(*(x,y), frame=submap.coordinate_frame))
    center_pix = [float(c) for c in center_pix] # For some reason, each coordinate is contained in an array

    point = plot_point(*center_pix, axes, 0.1, color='brown')

    return point