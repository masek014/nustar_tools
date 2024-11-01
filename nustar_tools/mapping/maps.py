import astropy.units as u
import copy
import math
import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
import numpy as np
import nustar_pysolar as nustar
import photutils
import sunpy.map

from astropy.io import fits
from astropy.time import Time
from regions import CircleSkyRegion, PixCoord, RectanglePixelRegion, SkyRegion

from . import tools as mtools
from ..utils import utilities
from ..utils.MinimumBoundingBox import MinimumBoundingBox

import warnings
warnings.simplefilter('ignore')


def make_nustar_map(
    evt_data: fits.FITS_rec,
    hdr: fits.Header,
    time_range: tuple[Time, Time] | None = None,
    normalize: bool = False
) -> sunpy.map.GenericMap:
    """
    Makes a map for the provided data.

    Parameters
    ----------
    evt_data : FITS record
        The data to be plotted.
    hdr : FITS header
        The header corresponding to the evt_data.
    time_range : tuple
        The time range (start, end) to filter the data around.
        In the format YYYY-MM-DD HH:MM:SS in UTC.
    normalize: bool
        Specifies whether to normalize the map data by the
        exposure time (i.e. livetime), giving map in units of DN/s.
    """

    filtered_data = evt_data
    if time_range is not None:
        time_range = Time(list(time_range), format='iso', scale='utc')
        filtered_data = evt_data[nustar.filter.by_time(evt_data, hdr, time_range)]

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
    """
    Makes a map for the provided evt file and plots
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
    """

    evt_data, hdr = utilities.get_event_data(evt_file)
    nustar_map = make_nustar_map(evt_data, hdr, normalize=normalize)
    nustar_submap, fig, ax, axes_limits = mtools.apply_map_settings(nustar_map, **cb_kwargs)
    
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
):
    """
    This method is very similar to nustar_pysolar's 'make_sunpy'
    method with the primary difference being that this method creates
    a sunpy map using a Numpy data array instead of a FITS record.
    As such, much of the code in this method is copied and pasted
    from the 'make_sunpy()' method.

    data_array is the array of data obtained from an
    **already created** SunPy map (it doesn't have to be, but that's
    the intended purpose of this method). This method requires the
    evt_data and hdr variables so that the map of data_array has
    the same parameters (time, references coordinates, etc.)
    as its parent map. Note that the map created by this method does
    **not** use evt_data and hdr to create the map. However, evt_data
    and hdr **should** be related to data_array in some manner.

    This method was designed for the purpose of creating maps of
    boxcar averaged frames and/or the residual data obtained by
    subtracting the boxcar average from the original   .

    Inputs:
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
    """    

    sunpy_header = mtools.make_sunpy_header(evt_data, hdr, exp_time, on_time, rebin_size, normalized)    
    data_map = sunpy.map.Map(data_array, sunpy_header)

    return data_map


def make_det_array(evt_data: fits.FITS_rec) -> np.ndarray:
    """
    Makes an array tracking the most recent det
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
    """

    arr = np.full((2999,2999), -1)

    x, y = evt_data['X'], evt_data['Y']
    dets = evt_data['DET_ID']
    arr[y,x] = dets

    return arr


def make_det_array3d(evt_data: fits.FITS_rec) -> dict:
    """
    Makes a dictionary tracking the dets that appear
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
    """
    
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
    """
    Makes a map consisting of the det information for each pixel.
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
    """

    # Replace the map data with the DET data.
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
    """
    Makes and plots a det map.

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
    """
    
    det_map = make_det_map(evt_data, hdr)
    det_map.data[:] += 1
    det_submap = mtools.get_submap(det_map, corners=corners)
    det_map.data[:] -= 1

    if fig is None:
        mtools.apply_style()
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection=det_submap)

    cmap = plt.get_cmap('jet')
    det_submap.plot(cmap=cmap)
    limb = det_submap.draw_limb(color='white', linewidth=1.25,
        linestyle='dotted', zorder=0, label='Solar disk')
    
    ax.set(xlabel='x [arcsec]', ylabel='y [arcsec]')

    mtools.apply_discrete_colorbar(fig, ax, 5, -1, 3, cmap=cmap, width=0.05)

    return fig, ax, det_submap


def make_cluster_map_data(cluster, detection_val=1.0, connection_val=0.01):
    """
    This method uses the macropixels contained in a Cluster object
    to generate a data array marking which pixels are contained
    in the Cluster object.

    The brightest pixel in the cluster is marked with a value of 1.0,
    and the connecting pixels are marked with a value of 0.01.
    All pixels not in the cluster have a value of 0.0.
    The different values are used to differentiate between
    the initial, detected macropixel and the connecting macropixels.
    Note: the values may need to be changed depending
    on the colormap used for plotting.

    Parameters
    ----------
    cluster : Cluster object
        The cluster used to make the data array.
    detection_val : float
        The value for the brightest (initial) macropixel.
    connection_val : float
        The value for the connecting macropixels.

    Returns
    -------
    data_array : np array
        An array marking which pixels are in the cluster.
    """

    rebin_size = cluster.pixel_list[0].bin_size
    coordinate_pairs = cluster.get_coordinate_pairs()
    data_array = np.zeros(shape=(math.ceil(2999/rebin_size),math.ceil(2999/rebin_size)))
    b_first = True
    
    for pair in coordinate_pairs:
        row_num = pair[0]
        col_num = pair[1]

        if b_first:
            data_array[row_num][col_num] = detection_val
            b_first = False
        else:
            data_array[row_num][col_num] = connection_val

    return data_array


# TODO: Rework this.
def make_cluster_map(evt_file, fig_dir, clusters_list, axes_limits=[],
    cmap='Blues', config_file=utilities.CONF_FILE):
    """
    General method used to make maps of clusters.

    Parameters
    ----------
    evt_file : str
        The event file used to make the Cluster objects.
        The data from this file is used for obtaining
        logistical data. It's not directly used for mapping the data.
    fig_dir : str
        The directory where the resulting map files will be saved.
    clusters_list : list
        The Cluster objects in this list will be used to make the map.
        The data from each pixel will be combined into one array to
        make a single map figure. Alternatively, a single Cluster
        object can be supplied here, and a map will be made using the
        data from the single object.
    axes_limits : list
        Axes limits to bound the mapping area.
    config_file : str
        The configuration file containing the map settings.
    """


    if not isinstance(clusters_list, list):
        clusters_list = [clusters_list]
    
    frame_number = clusters_list[0].frame_number
    fig_path = f'{fig_dir}clustermap_Frame{frame_number}.png'

    if not utilities.os.path.exists(fig_path):

        macropixel_bin_size = int(utilities.get_config_option('GeneralSettings', 'BIN_SIZE', config_file))
        frame_length = int(utilities.get_config_option('GeneralSettings', 'FRAME_LENGTH', config_file))
        if not axes_limits:
            # Change the observation map save directory so it isn't in the cluster dir.
            obs_map_dir = utilities.os.path.normpath(fig_dir + utilities.os.sep + utilities.os.pardir)+'/'
            nustar_submap, fig, ax, axes_limits = plot_observation_map(evt_file, fig_dir=obs_map_dir)

        total_data_array = np.zeros(shape=(math.ceil(2999/macropixel_bin_size),math.ceil(2999/macropixel_bin_size)))

        for cluster in clusters_list:
            cluster_array = make_cluster_map_data(cluster)
            total_data_array = np.add(total_data_array, cluster_array)

        # Read configuration settings.
        config_dict = {}
        utilities.apply_config_settings(config_dict, 'MacropixelClusterMapSettings', config_file)

        evt_data, hdr = utilities.get_event_data(evt_file)
        frame_map = make_sunpy_with_array(evt_data, hdr, total_data_array, rebin_size=macropixel_bin_size)
        cluster_submap, fig, ax, _ = mtools.apply_map_settings(frame_map, macropixel_bin_size,
            corners=axes_limits, cmap=cmap)

        # Update the title to feature the frame time,
        # instead of the observation start time.
        prev_title = ax.get_title()
        time_index = utilities.find_nth(prev_title, '-', 1) - 4
        prev_prefix = prev_title[:time_index]
        prev_time = prev_title[time_index:]
        frame_time = utilities.datetime.fromisoformat(prev_time) + \
                        utilities.timedelta(0, frame_number*frame_length)
            
        if (clusters_list[0].parent_array).b_extrapolate_background:
            frame_time += utilities.timedelta(0, -1*frame_length*(clusters_list[0].parent_array).boxcar_width)
        ax.set_title(prev_prefix + str(frame_time))
        
        # Add frame number to upper left-hand corner of the frame.
        if config_dict['B_FRAME_LABEL']:
            mtools.text(0.05, 0.95, str(frame_number), fontsize=config_dict['FRAME_NUMBER_FONT_SIZE'],
                ha='center', va='center', transform=ax.transAxes)

        mtools.save_map(fig, fig_dir, f'clustermap_frame{frame_number}')
    
    else:
        print(f'Cluster map {fig_path} already exists. Skipping.')


def make_event_map(event, axes_limits=[], b_add_spec_region=True, b_add_fov=True, b_zoom=False):
    """
    Plots the event heatmap on the solar disk.

    Parameters
    ----------
    event : Event object
        The event to be mapped.
    axes_limits : list
        The limits bounding the plotting area,
        in arcseconds (xmin, ymin, xmax, ymax).
    b_add_spec_region : bool
        Specifies whether the emission region should
        be plotted on the map.
    b_add_fov : bool
        Specifies whether NuSTAR's FOV during the event
        should be plotted on the map.
    b_zoom : bool
        Specifies whether the map should be zoomed in around the macropixels.
        If false, the axes bounds will be determined based on the photon hits.

    Returns
    -------
    fig : matplotlib figure
        The figure corresponding to the map.
    ax : matplotlib axes
        The axes corresponding to the map.
    event_submap : Sunpy map
        The map containing the plotted event.
    """

    mtools.apply_style()

    data_array = event.make_counts_array()

    if not b_zoom and not axes_limits:
        corners = mtools.find_min_max(event.nustar_map.data)
        axes_limits = list(corners)

    event_submap, fig, ax, _ = mtools.apply_map_settings(event.event_map, corners=axes_limits,
        cmap='viridis', norm=mplcolors.Normalize(0, np.max(data_array)),
        label='Counts')
    ax.set_title(f'NuSTAR Eventmap {event.event_id} {event.start}')

    if b_add_spec_region:
        
        pix_reg = event.spec_region.to_pixel(event_submap.wcs)
        init_pix_reg = event.initial_spec_region.to_pixel(event_submap.wcs)
        center_x, center_y, radius = mtools.get_arcsecond_coordinates(event.spec_region)
        coord_str = f'c: ({center_x:.2f}\", {center_y:.2f}\") r: {radius:.2f}\"'
        pix_reg.plot(ax=ax, edgecolor='red', linestyle='dashed',
            label=f'spec coords:, {coord_str}')
        init_pix_reg.plot(ax=ax, edgecolor='blue', linestyle='dashed')
        
        pix_reg = event.background_spec_region.to_pixel(event_submap.wcs)
        init_pix_reg = event.initial_spec_region.to_pixel(event_submap.wcs)
        center_x, center_y, radius = mtools.get_arcsecond_coordinates(event.background_spec_region)
        coord_str = f'c: ({center_x:.2f}\", {center_y:.2f}\") r: {radius:.2f}\"'
        pix_reg.plot(ax=ax, edgecolor='orange', linestyle='dotted',
            label=f'spec coords:, {coord_str}')

    if b_add_fov:
        fov = FOV(event.evt_data, event.hdr)
        fov.plot(event_submap, ax)
        spec_str = f'Spec. center: ({center_x:.2f}\", {center_y:.2f}\") Spec. radius: {radius:.2f}\"'
        fov_str = fov.get_fov_string()
        # text(0.02, 0.90, spec_str + '\n' + fov_str, fontsize=12,
        #   color='black', va='center', transform=ax.transAxes)

    ax.grid(False)
    mtools.save_map(fig, event.event_dir, 'eventmap')

    return fig, ax, event_submap


def generate_maps(id_dir):
    """
    Generate maps using the data in the provided data ID directory.
    """

    mtools.apply_style()

    # Get the ID number and generate the maps for both FPMs.
    fig_dir = id_dir + 'figures/'
    obs_id = utilities.get_id_from_id_dir(id_dir)

    # Make maps for both FPMs.
    plot_observation_map(f'{id_dir}event_cl/nu{obs_id}A06_cl_sunpos.evt', fig_dir=fig_dir, file_name='observation_mapA')
    plot_observation_map(f'{id_dir}event_cl/nu{obs_id}B06_cl_sunpos.evt', fig_dir=fig_dir, file_name='observation_mapB')

    
class FOV():
    """
    This class is designed to fit and maintain a field of view around the
    provided input NuSTAR event. This class also includes methods for
    fitting a region within the FOV and plotting the FOV.
    """

    def __init__(self, evt_data: fits.FITS_rec, hdr: fits.Header):
        """
        Initialize the object with the data and the time of interest.
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
        """

        self.evt_data, self.hdr = evt_data, hdr
        self.data_map = make_nustar_map(evt_data, hdr)
        self.fit_fov()


    def fit_fov(self):
        """
        Fit a FOV around the data map.

        The FOV is determined by the data contained by data_map.
        The coordinates and side lengths of the FOV are then
        converted to the units used by input_map. This allows
        for a rect to be made for a macropixel map.
        """

        data_map = self.data_map

        # Get the bounding box around the data.
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
                dist = mtools.compute_distance(c[0], c[1], c_comp[0], c_comp[1])
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


    def fit_coordinate_center(self, region: SkyRegion) -> SkyRegion:
        """
        Moves the center of the provided region to the brightest pixel.
        """

        reg_data = mtools.get_region_data(region.to_pixel(self.data_map.wcs),
            self.data_map.data, b_full_size=True)

        # Ensure the array does not contain only zeros.
        if reg_data.any():
            y, x = np.unravel_index(reg_data.argmax(), reg_data.shape) # brightest pixel
            # x, y = photutils.centroids.centroid_com(reg_data) # center of mass
            com_sky = PixCoord(x, y).to_sky(self.data_map.wcs)
            new_region = CircleSkyRegion(com_sky, radius=region.radius)
        else:
            print('Submap contains only zeros. Not fitting region center.')
            new_region = region

        return new_region


    def check_region_outside_fov(self, region: SkyRegion):
        """
        Check whether the given region is within the other region.

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
        region : tuple
            The region that will be checked for overlap with the FOV.
            (center_x, center_y, radius) in arcseconds.

        Returns
        -------
        The amount by which the region extends beyond the FOV. 
        """

        reg_pix = region.to_pixel((self.data_map).wcs)
        fov_pix = (self.rect).to_pixel((self.data_map).wcs)

        intersection = fov_pix.intersection(reg_pix)
        intersection_mask = intersection.to_mask()
        fov_mask = fov_pix.to_mask()

        # Compute difference between the shapes in each direction.
        diff = [abs(v1 - v2) for v1, v2 in zip((fov_mask.data).shape, (intersection_mask.data).shape)]

        return np.max(diff)


    def fit_region_within_edges(self, region):
        """
        Fit the inital region within the FOV by reducing the radius
        if the region extends beyond the FOV edges.

        Parameters
        ----------
        region : tuple
            Contains the initial region coordinate in the format:
            (center_x, center_y, radius) in arcseconds.

        Returns
        -------
        A tuple containing the new coordinates in the format:
        (center_x, center_y, radius) in arcseconds.
        """
        
        # Determine how many pixels the region extends beyond the FOV edges.
        extended_pix = self.check_region_outside_fov(region)
        if extended_pix > 0:
            extended_pix += 5 # Reduce it by another ~2 arcseconds to account for pixel uncertainty
            dec_value = extended_pix*((self.data_map).scale[0].value) * u.arcsec
            region.radius = max(region.radius - dec_value, 10*u.arcsecond) # TODO: How do we want to handle this?
        
        return region


    def fit_region_within_chipgap(self, region):
        """
        Fit the inital region within the chip gap by reducing the radius
        if the region contains events from more than one detector.

        Parameters
        ----------
        region : tuple
            Contains the initial region coordinate in the format:
            (center_x, center_y, radius) in arcseconds.

        Returns
        -------
        A tuple containing the new coordinates in the format:
        (center_x, center_y, radius) in arcseconds.
        """

        det_map = make_det_map(self.evt_data, self.hdr)
        dets = mtools.find_dets_in_region(det_map, region.to_pixel(det_map.wcs))

        while len(dets) > 2 and region.radius.value > 25: # TODO: Make this min radius selection better
            # Decrement by 2.5 arcseconds per iteration since each detector pixel is about 2.5 arcseconds
            region.radius = (region.radius.value - 2.5)*u.arcsec
            dets = mtools.find_dets_in_region(det_map, region.to_pixel(det_map.wcs))
            if -1 in dets:
                dets.remove(-1)

        return region


    def fit_region(
        self,
        region: SkyRegion,
        fit_coordinate_center: bool = True,
        fit_within_detector_edges: bool = True,
        fit_within_chipgap: bool = True
    ) -> SkyRegion:
        """
        Fit the inital region within the chip gap by reducing the radius
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
        """

        reg = copy.deepcopy(region)
        if fit_coordinate_center:
            reg = self.fit_coordinate_center(reg)
        if fit_within_detector_edges:
            reg = self.fit_region_within_edges(reg)
        if fit_within_chipgap:
            reg = self.fit_region_within_chipgap(reg)

        return reg


    def get_fov_string(self):

        x, y = self.rect.center.Tx.arcsec, self.rect.center.Ty.arcsec
        width, height = self.rect.width.value/60, self.rect.height.value/60
        angle = self.rect.angle.value
        s = f'FOV center: ({x:0.1f}\", {y:0.1f}\")\n'\
            f'FOV side length: {width:0.1f}\' by {height:0.1f}\' \n'\
            f'FOV rotation: {angle:0.1f} deg\n'

        return s


    def plot(self, input_map, ax, b_draw_chip_gap=True, b_plot_corners=False, b_plot_center=False, **kwargs):
        """
        Plot the FOV on the provided map.

        Parameters
        ----------
        input_map : Sunpy map
            The map on which the FOV will be plotted.
        ax : matplotlib axes
            The axes on which the FOV will be plotted.
            Must be associated with input_map.
        b_draw_chip_gap : bool
            Specify whether to draw the chip gap.
        b_plot_corners : bool
            Specify whether to draw dots on the corners of the FOV .
        b_plot_center : bool
            Specify whether to draw a center point in the FOV.
        kwargs
            Keyword arguments for the cosmetic features of the circle.
        """

        default_kwargs = {'edgecolor': 'purple', 'linestyle': 'dashed', 'lw': 1}
        kwargs = {**default_kwargs, **kwargs}

        m_size = input_map.scale[0].value
        fov_pix = self.rect.to_pixel(input_map.wcs)

        # Plot the FOV.
        fov_pix.plot(ax=ax, **kwargs)
        if b_plot_center:
            mtools.plot_point(fov_pix.center.xy[0], fov_pix.center.xy[1], ax, radius=1/m_size)

        if b_plot_corners:
            for c in fov_pix.corners:
                mtools.plot_point(c[0], c[1], ax, radius=1/m_size)

        if b_draw_chip_gap:
            cg_rect1 = RectanglePixelRegion(PixCoord(*fov_pix.center.xy),
                10/m_size, fov_pix.height, angle=fov_pix.angle)
            cg_rect1.plot(ax=ax, facecolor='none', edgecolor=kwargs['edgecolor'], linestyle='dashed', lw=1)
            cg_rect2 = RectanglePixelRegion(PixCoord(*fov_pix.center.xy),
                fov_pix.width, 10/m_size, angle=fov_pix.angle)
            cg_rect2.plot(ax=ax, facecolor='none', edgecolor=kwargs['edgecolor'], linestyle='dashed', lw=1)