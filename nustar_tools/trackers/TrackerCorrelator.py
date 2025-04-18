import datetime
import typing

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from astropy.time import Time

from .CoordinateTracker import CoordinateTracker
from ..mapping import tools as mtools
from ..plotting import tools as ptools


def find_continuities(arr: np.ndarray) -> list[tuple[int, int]]:
    '''Identifies intervals where the value is unchanged.'''
    ind = np.where(arr != 0)[0]
    continuities = []  # list of tuples defining the continuous ranges, inclusive
    start = end = ind[0]
    for i in ind[1:]:
        if i != end+1 or i == ind[-1]:
            if i == ind[-1]:
                end = i
            continuities.append((start, end))
            start = i
        end = i

    return continuities


class TrackerCorrelator():
    '''Computes correations between two CoordinateTracker objects.'''

    def __init__(self, tracker1: CoordinateTracker, tracker2: CoordinateTracker):
        self.tracker1 = tracker1
        self.tracker2 = tracker2

    @property
    def id_dir(self) -> str:
        '''Observation ID directory.'''
        return self.tracker1.id_dir

    @property
    def id_num(self) -> str:
        '''Observation ID number.'''
        return self.tracker1.id_num

    @property
    def date(self) -> str:
        '''Date of the observation.'''
        return self.tracker1.date

    @property
    def fpm(self) -> str:
        '''FPM of the trackers.'''
        fpm1 = self.tracker1.fpm
        fpm2 = self.tracker2.fpm
        if fpm1 is not None:
            return fpm1
        elif fpm2 is not None:
            return fpm2

        return None

    def _compute_threshold_old(
        self,
        bin_width: float = 0.001,
        num_std: float = 2
    ):
        '''Compute a threshold based on the amplitudes of the correlations.
        Uses continuities in the correlation.
        '''
        H, bins = self.compute_amplitude_hist(bin_width)
        cont = find_continuities(H)
        ind = (self.amplitude >= 0) & (self.amplitude <= bins[cont[0][1]+1])
        median = np.median(self.amplitude[ind])
        std_dev = np.std(self.amplitude[ind])
        self.threshold = median + num_std * std_dev

    def _compute_threshold(self, quantile: float = 99, num_std: float = 3):
        '''Compute a threshold based on the amplitudes of the correlations.'''
        p = np.percentile(self.amplitude, quantile)
        ind = self.amplitude <= p
        median = np.median(self.amplitude[ind])
        std_dev = np.std(self.amplitude[ind])
        self.threshold = median + num_std * std_dev

    def correlate_trackers(
        self,
        window_size: float = 60,
        time_step: int = 1,
        corr_func: typing.Callable = np.cov,
        normalize: bool = True
    ):
        '''Correlate the values associated with the two trackers.
        window_size is the width of the correlation interval, in seconds.
        time_step is the difference between windows, in seconds.
        '''
        # Ordered xx, xy, yx, yy.
        self.times = []
        self.matrices = {
            'xx': [], 'xy': [], 'yx': [], 'yy': []}  # Holds the covariance matrices
        self.covs = {
            'xx': [], 'xy': [], 'yx': [], 'yy': []}  # Holds the covariance values

        # Determine the skip amount for the array with higher cadence.
        diff1 = np.diff(self.tracker1.times)[0]
        diff2 = np.diff(self.tracker2.times)[0]
        if diff1 > diff2:
            by = int(diff1 / diff2)
            t1 = self.tracker1
            t2 = self.tracker2
        else:
            by = int(diff2 / diff1)
            t1 = self.tracker2
            t2 = self.tracker1

        # Determine start and end times.
        start_time = np.max([self.tracker1.times[0], self.tracker2.times[0]])
        end_time = np.min([self.tracker1.times[-1], self.tracker2.times[-1]])
        window_start = start_time
        window_end = window_start + datetime.timedelta(seconds=window_size)

        while window_end < end_time:

            t1_inds = (t1.times >= window_start) & (t1.times <= window_end)
            x1 = t1.x[t1_inds].value
            y1 = t1.y[t1_inds].value

            t2_inds = (t2.times >= window_start) & (t2.times <= window_end)
            x2 = t2.x[t2_inds].value[::by-1][:len(x1)]
            y2 = t2.y[t2_inds].value[::by-1][:len(y1)]

            if np.isnan(x1).any() or np.isnan(x2).any():
                for key in self.matrices.keys():
                    self.matrices[key].append(np.nan)
                    self.covs[key].append(np.nan)
            else:
                for (key, pair) in zip(
                        self.matrices.keys(), [(x1, x2), (x1, y2), (x2, y1), (y1, y2)]):
                    m = corr_func(*pair)
                    self.matrices[key].append(m)
                    self.covs[key].append(m[0, 1])

            # Center the time on the window midpoint.
            self.times.append(
                window_start+datetime.timedelta(seconds=window_size/2))
            window_start += datetime.timedelta(seconds=time_step)
            window_end += datetime.timedelta(seconds=time_step)

        for (key, item) in self.covs.items():
            self.covs[key] = np.nan_to_num(np.array(item))
            # self.covs[key] = np.ma.array(item, mask=np.isnan(item))
            if normalize:
                self.covs[key] /= np.linalg.norm(np.nan_to_num(self.covs[key]))

        self.amplitude = np.power(
            self.covs['xx']**2 + self.covs['xy']**2 +
            self.covs['yx']**2 + self.covs['yy']**2,
            1/2
        )

        self._compute_threshold()

        # # Convert to 2x2 numpy arrays.
        # self.matrices = np.empty((2,2), dtype=object)
        # for i in range(len(matrices)):
        #     self.matrices[i//2,i%2] = np.array(matrices[i])

        # self.covs = np.empty((2,2), dtype=object)
        # for i in range(len(covs)):
        #     self.covs[i//2,i%2] = np.array(covs[i])

    def compute_amplitude_hist(self, bin_width=0.001):

        num_bins = int(
            (mtools.np.nanmax(self.amplitude) - mtools.np.nanmin(self.amplitude))/bin_width)
        # num_bins = 100
        H, bins = np.histogram(self.amplitude, bins=num_bins)

        return H, bins

    def fill_time(self, ax: plt.Axes, where: np.ndarray):
        '''Shades the interval on the plot specified by where.
        where is a boolean array specifying intervals to shade
        '''
        ylim = ax.get_ylim()
        ax.fill_between(
            self.times, *ylim, where=where, color='gray', alpha=0.2)
        ax.set_ylim(ylim)

    def plot_tracker(
        self,
        which: str,
        xax: plt.Axes,
        yax: plt.Axes,
        where: np.ndarray = None,
        **set_kwargs
    ):
        '''which can either be  '1' or '2'
        where is a boolean array specifying intervals to shade.
        '''
        if str(which) == '1':
            tracker = self.tracker1
        elif str(which) == '2':
            tracker = self.tracker2
        else:
            raise ValueError(
                f'"which" must be either "1" or "2", not "{which}"')
        for ax, coord in zip([xax, yax], ['x', 'y']):
            tracker.coordinate_timeseries(ax, coord, set_kwargs=set_kwargs)
            if where is not None:
                self.fill_time(ax, where)

    def plot_cov(
        self,
        which: str,
        ax: plt.Axes,
        where: np.ndarray = None,
        **set_kwargs
    ):
        '''Plot the covariance arrays.
        which is a string: 'xx', 'xy', 'yx', or 'yy'.
        '''
        cov = self.covs[which]
        ax.plot(self.times, cov, color='black', lw=1)
        ax.tick_params(axis='x', which='both', direction='in', top=True)
        ax.tick_params(axis='y', which='both', direction='in', right=True)
        ax.set(ylabel=f'N. Cov(${which}$)', **set_kwargs)
        ptools.set_x_ticks(ax)
        ptools.set_y_ticks(ax)
        if where is not None:
            self.fill_time(ax, where)

    def plot_amplitude(self, ax: plt.Axes, where: np.ndarray = None, **set_kwargs):
        '''Plot the amplitude of the correlation.'''
        ax.plot(self.times, self.amplitude, color='black', lw=1)
        ax.axhline(
            self.threshold, color='blue', ls='dotted', lw=1, label='Threshold')
        ax.set(**set_kwargs)
        ax.tick_params(axis='x', which='both', direction='in', top=True)
        ax.tick_params(axis='y', which='both', direction='in', right=True)
        ptools.set_x_ticks(ax)
        ptools.set_y_ticks(ax)
        ax.legend(fontsize=10)
        if where is not None:
            self.fill_time(ax, where)

    def plot_amplitude_histogram(
        self,
        ax: plt.Axes,
        bin_width: float = 0.001,
        plot_threshold: bool = True,
        **set_kwargs
    ):
        '''Plot a histogram of the correlation amplitude values.'''
        default_kwargs = {
            'xlim': [0, ax.get_xlim()[1]],
            'xlabel': 'Amplitude',
            'ylabel': 'Counts',
            'yscale': 'log'
        }
        set_kwargs = {**default_kwargs, **set_kwargs}
        ax.stairs(
            *self.compute_amplitude_hist(bin_width),
            fill=True, color='lightblue')
        ax.set(**set_kwargs)
        ax.tick_params(axis='x', which='both', direction='in', top=True)
        ax.tick_params(axis='y', which='both', direction='in', right=True)

        if plot_threshold:
            ax.axvline(self.threshold, color='blue',
                       lw=1, ls='dotted', label='Threshold')
            # self._compute_threshold_old()
            # ax.axvline(self.threshold, color='gray',
            #     lw=1, ls='dashed', label='old threshold')
            p = np.percentile(self.amplitude, 99)
            ax.axvline(p, color='green', ls='dashed', label='99th percentile')
            ax.legend()

    def make_overview(
        self,
        time_range: tuple[str] | tuple[Time] | None = None
    ) -> tuple[plt.Figure, gridspec.GridSpec, gridspec.GridSpec]:

        fig = plt.figure(figsize=(12, 10), constrained_layout=True)
        spacing = dict(top=0.92, bottom=0.1, wspace=0.3)
        height_ratios = [1, 1, 1, 1, 2, 2]
        gs_top = gridspec.GridSpec(
            6, 2, height_ratios=height_ratios, **spacing)
        gs_bottom = gridspec.GridSpec(
            6, 2, hspace=0.5, height_ratios=height_ratios)
        where = self.amplitude > self.threshold
        if time_range is None:
            xlim = self.tracker1.observation_time
        else:
            xlim = [
                Time(time_range[0], scale='utc').datetime,
                Time(time_range[1], scale='utc').datetime,
                # utilities.convert_string_to_datetime(time_range[0]),
                # utilities.convert_string_to_datetime(time_range[1])
            ]

        # Plot tracker data.
        for i in range(2):
            xax = fig.add_subplot(gs_top[i, 0])
            yax = fig.add_subplot(gs_top[i, 1])
            self.plot_tracker(
                str(i+1), xax, yax, where, xlim=xlim, xticklabels=[])

        # Plot cov data.
        for i, key in enumerate(self.covs.keys(), start=2):
            r, c = i-i//2-i % 2+1, i % 2
            ax = fig.add_subplot(gs_top[r, c])
            self.plot_cov(key, ax, where, xlim=xlim)
            if key[0] != 'y':
                ax.set(xticklabels=[])

        # Plot amplitude
        ax = fig.add_subplot(gs_bottom[4, :])
        self.plot_amplitude(
            ax, where,
            xlim=xlim,
            ylim=[0, np.max(self.amplitude)],
            ylabel=r'$\sqrt{\Sigma (\mathrm{Cov}^2)}$'
        )

        # Histogram
        ax = fig.add_subplot(gs_bottom[5, :])
        self.plot_amplitude_histogram(ax, xlim=[0, 0.2])
        fig.align_ylabels()
        fig.suptitle(f'{self.date}, {self.id_num} - FPM{self.fpm}')
        # fig.patch.set_alpha(0)
        # plt.savefig(f'./{self.date}_{self.id_num}_fpm{self.fpm}_trackers.png', dpi=400, bbox_inches='tight')

        return fig, gs_top, gs_bottom
