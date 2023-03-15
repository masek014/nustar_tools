import numpy as np
import astropy.units as u

from astropy.io import fits
from skimage import restoration
from scipy.ndimage import rotate


def deconvolve(nustar_map, psf_file, source_position, oa_position, it=10, clip=False):
    """
    Uses the nustar_map and point spread function (obtained from the psf_file)
    and deconvolves using the Richardson-Lucy method with a number of
    iterations (it).
    
    Parameters
    ----------
    nustar_map : Sunpy map
        The map of the data. Should be over the field of view. If "None" then the self.nustar_map class attribute is used.
    psf_file : file string or 2d array
        The FITS file containing the point-spread function.
    source_position : tuple of astropy quantity
        Specifies the location of the source.
    oa_position : float
        Specifies the position of the optical axis.
    it : int
        Number of iterations for the deconvolution.
    clip : bool
        Set values >1 and <-1 to 1 and -1 respectively after each iteration. Unless working with a 
        normalised image this should be "False" otherwise it's a mess.
        
    Returns
    -------
        A 2d numpy array of the deconvolved map.
    """

    source_x, source_y = source_position
    oa_x, oa_y = oa_position
    
    dist = np.sqrt( (source_x - oa_x)**2 + (source_y - oa_y)**2 ) << u.arcmin
    angle = np.arctan((source_y - oa_y).value / (source_x - oa_x).value) * u.rad << u.deg
    angle = np.abs(angle)

    # The index specifying the PSF array to use since there are 18 arrays for
    # different distances.
    index = int(dist.to(u.arcmin).value / 0.5) + 1
    
    with fits.open(psf_file) as hdu:
        hdr = hdu[index].header
        psf_array = hdu[index].data

    # Ensure the resolution of the map matches the PSF.
    for i in ['1', '2']:
        psf_res = hdr[f'CDELT{i}'] * u.Unit(hdr[f'CUNIT{i}'])
        map_res = nustar_map.meta[f'CDELT{i}'] * u.Unit(nustar_map.meta[f'CUNIT{i}'])
        assert psf_res == map_res, 'The resolution in the PSF and the provided map are different.'
        
    assert -90*u.deg <= angle <= 90*u.deg, 'Angle between optical axis and source not within -90 to 90 deg. Invalid source and/or optical axis position?'
    psf_array = rotate(psf_array, angle.to(u.deg), reshape=True)

    deconv = restoration.richardson_lucy(
        nustar_map.data,
        psf_array,
        num_iter=it,
        clip=clip,
        filter_epsilon=1
    )
    
    return deconv