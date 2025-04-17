import astropy.units as u
import numpy as np


def minimum_bin_counts(
    counts: u.Quantity,
    edges: u.Quantity,
    minimum_counts_per_bin: u.Quantity
) -> tuple[u.Quantity, u.Quantity]:
    '''Define new bin edges according to the specified minimum counts per bin.
    This defines the new bin edges as multiples of the bin widths, so no
    flux is created/lost due to interpolation.
    Returns values, edges.

    Equivalent to HEASOFT's grppha.

    There may be a way to take advantage of Numpy's indexing tricks
    to make this even better and faster.
    '''
    new_counts, new_edges = [], []
    i = 0
    while i < len(counts):
        j = i + 1
        count_sum = np.sum(counts[i:j])
        while count_sum < minimum_counts_per_bin and j < len(counts):
            j += 1
            count_sum = np.sum(counts[i:j])
        new_counts.append(count_sum)
        new_edges.append(edges[i])
        i = j
    new_edges.append(edges[i])
    if new_counts[-1] < minimum_counts_per_bin:
        new_counts = new_counts[:-1]
        new_edges = new_edges[:-1]
    new_edges = new_edges * edges.unit
    new_counts = new_counts * counts.unit

    return new_counts, new_edges


def filter_histogram_bins(
    values: np.ndarray,
    bins: np.ndarray,
    bin_range: tuple
) -> tuple[np.ndarray, np.ndarray]:
    '''Filters histogram bins to contain only bins within the specified range.
    Returns values, bins.
    '''
    inds = np.asarray((bins >= bin_range[0]) & (
        bins <= bin_range[1])).nonzero()[0]
    values = values[inds[0]:inds[-1]]
    bins = bins[inds]

    return values, bins


'''
Flux conserving rebin methods developed by William Setterberg.
'''


def flux_conserving_rebin(
    old_edges: np.typing.ArrayLike,
    old_values: np.typing.ArrayLike,
    new_edges: np.typing.ArrayLike,
) -> np.ndarray:
    '''Rebin a histogram by performing a flux-conserving rebinning.
    The total area of the histogram is conserved.
    Adjacent bins are proportionally interpolated for new edges that do not line up.
    Don't make the new edges too finely spaced;
    don't make a new bin fall inside of an old one completely.
    '''
    old_edges = np.array(np.sort(old_edges))
    new_edges = np.array(np.sort(new_edges))
    nd = np.diff(new_edges)
    od = np.diff(old_edges)
    if (new_edges[0] < old_edges[0]) or (new_edges[-1] > old_edges[-1]):
        raise ValueError('New edges cannot fall outside range of old edges.')
    if nd.shape == od.shape and np.all(nd == od):
        return old_values
    orig_flux = od * old_values
    ret = np.zeros(new_edges.size - 1)
    for i in range(ret.size):
        ret[i] = interpolate_new_bin(
            original_area=orig_flux,
            old_edges=old_edges,
            new_left=new_edges[i],
            new_right=new_edges[i+1])

    return ret


def proportional_interp_single_bin(
    left_edge: float,
    right_edge: float,
    interp: float
) -> tuple[float, float]:
    '''Say what portion of a histogram bin belongs on the left and right
    of an edge to interpolate.
    '''
    denom = right_edge - left_edge
    right_portion = (right_edge - interp) / denom
    left_portion = (interp - left_edge) / denom

    return left_portion, right_portion


def bounding_interpolate_indices(
    old_edges: np.ndarray,
    left: float,
    right: float
) -> tuple[int, int]:
    '''Find the indices of the old edges that bound the new left
    and right edges.
    '''
    indices = np.arange(old_edges.size)
    new_left = indices[old_edges <= left][-1]
    new_right = indices[old_edges >= right][0]

    return (new_left, new_right)


def interpolate_new_bin(
    original_area: np.array,
    old_edges: np.array,
    new_left: float,
    new_right: float
) -> float:
    '''Interpolate the new bin value given old edges, new edges,
    and the old flux (aka area).
    '''
    oa = original_area
    oe = old_edges

    old_start_idx, old_end_idx = bounding_interpolate_indices(
        oe, new_left, new_right
    )

    # portion of edge bins that get grouped with the new bin
    _, left_partial_prop = proportional_interp_single_bin(
        left_edge=oe[old_start_idx],
        right_edge=oe[old_start_idx+1],
        interp=new_left
    )
    left_partial_area = left_partial_prop * oa[old_start_idx]

    right_partial_prop, _ = proportional_interp_single_bin(
        left_edge=oe[old_end_idx-1],
        right_edge=oe[old_end_idx],
        interp=new_right
    )
    right_partial_area = right_partial_prop * oa[old_end_idx-1]

    partial_slice = slice(old_start_idx + 1, old_end_idx - 1)
    have_bad_slice = (partial_slice.start > partial_slice.stop)
    if have_bad_slice:
        raise ValueError('Your new bins are too fine. Use coarser bins.')

    between_area = oa[partial_slice].sum()

    delta = new_right - new_left
    new_bin_value = (
        left_partial_area + between_area + right_partial_area) / delta

    return new_bin_value
