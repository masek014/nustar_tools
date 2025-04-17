import astropy.units as u
import numpy as np
import shutil

from astropy.io import fits

from .response import ResponseHandler
from .broken_power_law import simulate_nonthermal_spectrum
from ..utils.histogram_tools import flux_conserving_rebin, minimum_bin_counts, filter_histogram_bins


PHA_FILE_FORMAT = '{spec_dir}/nu20801024001{fpm}06_cl_g0_sr.pha'
DIR_TIME_FORMAT = '%H%M%S'


def read_pha_spectrum(
    pha_file: str,
    energy_range: tuple[u.Quantity, u.Quantity] = None,
    minimum_counts_per_bin: u.Quantity = None,
    additional_error_factor: float = 0,
    distribution: bool = False
) -> tuple[u.Quantity, u.Quantity, u.Quantity]:
    '''Returns values, edges, errors.

    The additional error factor multiplies the count values and is then added
    in quadrature with the Poissonian count error.
    '''

    with fits.open(pha_file) as hdu:
        data = hdu[1].data
        exposure = hdu[1].header['EXPOSURE'] * \
            u.Unit(hdu[1].header['TIMEUNIT'])

    energies = (0.04 * data['CHANNEL'] + 1.6) * u.keV
    edges = np.append(energies, energies[-1]+0.04*u.keV)
    values = data['COUNTS'] * u.ct

    if energy_range is not None:
        values, edges = filter_histogram_bins(
            values, edges, energy_range)

    if minimum_counts_per_bin is not None:
        values, edges = minimum_bin_counts(
            values, edges, minimum_counts_per_bin)
    errors = np.sqrt(
        values.value + (additional_error_factor*values.value)**2) * u.ct

    if distribution:
        values = (values / np.diff(edges) / exposure)
        errors = (errors / np.diff(edges) / exposure)

    return values, edges, errors


def make_thermal_pha(
    in_pha: str,
    out_pha: str,
    rmf_file: str,
    arf_file: str,
    nonthermal_parameters: dict[str, u.Quantity],
    gain_slope: float,
    fpm_factor: float = 1
) -> dict[str, str]:
    '''Creates a new PHA file with the nonthermal component removed.
    The nonthermal parameters are: norm, break_energy, lower_index, upper_index.
    '''
    # Simulate nonthermal spectrum from XSPEC parameters.
    handler = ResponseHandler(rmf_file, arf_file)
    nonthermal_spectrum, nonthermal_edges = simulate_nonthermal_spectrum(
        handler.srm, handler.energy_edges, **nonthermal_parameters)

    # TODO: Figure out what to do with the factor!
    # if apply_fpm_scaling: # fpm == 'B':
    nonthermal_spectrum *= fpm_factor
    print('WARNING [make_thermal_pha]: The scaling factor between '
          'FPMs A and B needs to be accounted for. Fix this!!!')

    # Read PHA data from file.
    with fits.open(in_pha) as hdu:
        pha_hdr = hdu[1].header
        exposure = pha_hdr['EXPOSURE'] * u.Unit(pha_hdr['TIMEUNIT'])
    pha_counts, pha_edges = read_pha_spectrum(in_pha)

    # Applying a gain < 1 will reduce the max energy edge.
    # Temporarily trim the PHA edges for the purpose of rebinning,
    # then pad the rebinned spectrum with zeros to match original PHA shape.
    # This is fine to do since the max energy bin is ~160 keV,
    # which is well outside of the calibrated NuSTAR energy limit
    # and even further from the solar observational limit.
    max_edge = (nonthermal_edges * gain_slope)[-1]
    nonthermal_spectrum_phabins = flux_conserving_rebin(
        nonthermal_edges.value * gain_slope,
        nonthermal_spectrum.value / gain_slope,
        pha_edges[pha_edges <= max_edge].value
    ) * nonthermal_spectrum.unit
    nonthermal_spectrum_phabins = np.pad(
        nonthermal_spectrum_phabins,
        (0, len(pha_counts)-len(nonthermal_spectrum_phabins)),
        'constant'
    )

    # TODO: Should we round up or down?
    # I think we should round down.
    nonthermal_counts_phabins = nonthermal_spectrum_phabins * \
        np.diff(pha_edges) * exposure
    thermal_pha_counts = np.trunc(pha_counts - nonthermal_counts_phabins)
    thermal_pha_counts[thermal_pha_counts < 0] = 0

    thermal_pha_counts_error = np.trunc(
        np.sqrt(pha_counts.value + nonthermal_counts_phabins.value)) * u.ct

    shutil.copyfile(in_pha, out_pha)
    with fits.open(out_pha) as hdu:
        orig_header = hdu[1].header
        hdu[1].data['COUNTS'] = thermal_pha_counts
        new_col = fits.ColDefs(
            [fits.Column(name='STAT_ERR', format='J', array=thermal_pha_counts_error.value)])
        hdu[1] = fits.BinTableHDU.from_columns(hdu[1].columns + new_col)
        hdu[1].header = orig_header
        hdu[1].header.set('NAXIS1', hdu[1].header['NAXIS1']+4)
        hdu[1].header.set('STAT_ERR', 1, comment='statistical error specified')
        hdu[1].header.set(
            'POISSERR', 0, comment='Poissonian error not assumed')
        hdu[1].header.set('TTYPE3', 'STAT_ERR',
                          comment='Count error per channel', after='TUNIT2')
        hdu[1].header.set(
            'TFORM3', 'J', comment='data format of field: 4-byte INTEGER', after='TTYPE3')
        hdu[1].header.set('TUNIT3', 'count',
                          comment='physical unit of field', after='TFORM3')
        hdu.writeto(out_pha, overwrite=True)


def make_gain_accounted_pha(
    in_pha: str,
    out_pha: str,
    gain_slope: float
):
    '''Creates a new PHA file with gain-accounted channel bins.
    The new channel widths are still 0.04 keV.
    '''
    shutil.copyfile(in_pha, out_pha)
    with fits.open(out_pha) as hdu:

        energies = hdu[1].data['CHANNEL'] * 0.04 + 1.6
        ga_channels = (energies / gain_slope - 1.6) / 0.04

        # I was unable to update the CHANNEL column data type by using header.set,
        # so here we define a new CHANNEL column.
        orig_header = hdu[1].header
        new_col = fits.ColDefs(
            [fits.Column(name='CHANNEL', format='1D', array=ga_channels)])
        hdu[1] = fits.BinTableHDU.from_columns(new_col + hdu[1].columns[1])
        hdu[1].header = orig_header
        hdu[1].header.set(
            'TTYPE1', 'CHANNEL', comment='Gain-Accounted Pulse Invarient (PI) Channel')
        hdu[1].header.set(
            'TFORM1', '1D', comment='data format of field: 8-byte DOUBLE')
        hdu[1].header.set(
            'NAXIS1', hdu[1].header['NAXIS1']+4)
        hdu.writeto(out_pha, overwrite=True)


def make_pileup_corrected_pha(
    in_pha: str,
    out_pha: str,
    pileup_pha: str,
    correction_factor: float
):
    '''Creates a new PHA file with pileup subtracted from the channel bins.

    The correction is applied as:
    counts - correction_factor * pileup_counts
    '''
    shutil.copyfile(in_pha, out_pha)
    with fits.open(pileup_pha) as hdu:
        pileup_counts = hdu[1].data['COUNTS']

    with fits.open(out_pha) as hdu:

        # TODO: Do we want to take the floor or ceiling?
        # I think floor, but idk. Probably doesn't matter too much tbh.
        counts = hdu[1].data['COUNTS']
        corrected_counts = np.floor(counts - correction_factor * pileup_counts)
        corrected_counts[corrected_counts < 0] = 0

        # I was unable to update the COUNTS column data type by using header.set,
        # so here we define a new CHANNEL column.
        orig_header = hdu[1].header
        new_col = fits.ColDefs(
            [fits.Column(name='COUNTS', format='J', array=corrected_counts)])
        hdu[1] = fits.BinTableHDU.from_columns(hdu[1].columns[0] + new_col)
        hdu[1].header = orig_header

        for key in orig_header:
            if orig_header[key] == 'COUNTS':
                break

        hdu[1].header.set(
            key, 'COUNTS', comment='Pileup-Corrected Counts per channel')
        hdu[1].header.set(
            f'TFORM{key[-1]}', 'J', comment='data format of field: 4-byte INTEGER')

        hdu.writeto(out_pha, overwrite=True)
