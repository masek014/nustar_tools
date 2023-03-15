import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from array_tools import find_continuities
from ..utils import utilities
from ..plotting import tools as ptools
from ..mapping import tools as mtools


class TrackerCorrelator():

    
    def __init__(self, tracker1, tracker2):
        self.tracker1 = tracker1
        self.tracker2 = tracker2


    @property
    def id_dir(self):
        return self.tracker1.id_dir

    
    @property
    def id_num(self):
        return self.tracker1.id_num


    @property
    def date(self):
        return self.tracker1.date


    @property
    def fpm(self):
        
        fpm1 = self.tracker1.fpm
        fpm2 = self.tracker2.fpm
        if fpm1 is not None:
            return fpm1
        elif fpm2 is not None:
            return fpm2
        
        return None


    def _compute_threshold_old(self, bin_width=0.001, num_std=2):

        H, bins = self.compute_amplitude_hist(bin_width)
        cont = find_continuities(H)
        ind = (self.amplitude >= 0) & (self.amplitude <= bins[cont[0][1]+1])
        median = np.median(self.amplitude[ind])
        std_dev = np.std(self.amplitude[ind])
        
        self.threshold = median + num_std * std_dev


    def _compute_threshold(self, q=99, num_std=3):

        p = np.percentile(self.amplitude, q)
        ind = self.amplitude <= p
        median = np.median(self.amplitude[ind])
        std_dev = np.std(self.amplitude[ind])
        
        self.threshold = median + num_std * std_dev


    def correlate_trackers(self, window_size=60, time_step=1,
        corr_func=np.cov, b_normalize=True):
        
        # Ordered xx, xy, yx, yy.
        self.times = []
        self.matrices = {'xx':[], 'xy':[], 'yx':[], 'yy':[]} # Holds the covariance matrices
        self.covs = {'xx':[], 'xy':[], 'yx':[], 'yy':[]} # Holds the covariance values

        # Determine the skip amount for the array with higher cadence.
        diff1 = np.diff(self.tracker1.times)[0]
        diff2 = np.diff(self.tracker2.times)[0]
        if diff1 > diff2:
            by = int(diff1/diff2)
            t1 = self.tracker1
            t2 = self.tracker2
        else:
            by = int(diff2/diff1)
            t1 = self.tracker2
            t2 = self.tracker1

        # Determine start and end times.
        start_time = np.max([self.tracker1.times[0], self.tracker2.times[0]])
        end_time = np.min([self.tracker1.times[-1], self.tracker2.times[-1]])

        window_start = start_time
        window_end = window_start + utilities.timedelta(seconds=window_size)
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
                for (key, pair) in zip(self.matrices.keys(), [(x1, x2), (x1, y2), (x2, y1) ,(y1, y2)]):
                    m = corr_func(*pair)
                    self.matrices[key].append(m)
                    self.covs[key].append(m[0,1])

            # Center the time on the window midpoint.
            self.times.append(window_start+utilities.timedelta(seconds=window_size/2))
            window_start += utilities.timedelta(seconds=time_step)
            window_end += utilities.timedelta(seconds=time_step)

        for (key, item) in self.covs.items():
            self.covs[key] = np.nan_to_num(np.array(item))
            # self.covs[key] = np.ma.array(item, mask=np.isnan(item))
            if b_normalize:
                self.covs[key] /= np.linalg.norm(np.nan_to_num(self.covs[key]))

        self.amplitude = np.power(
            self.covs['xx']**2 + self.covs['xy']**2 + \
            self.covs['yx']**2 + self.covs['yy']**2 ,
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

        num_bins = int((mtools.np.nanmax(self.amplitude)-mtools.np.nanmin(self.amplitude))/bin_width)
        # num_bins = 100
        H, bins = np.histogram(self.amplitude, bins=num_bins)

        return H, bins


    def fill_time(self, ax, where):

        ylim = ax.get_ylim()
        ax.fill_between(self.times, *ylim, where=where, color='gray', alpha=0.2)
        ax.set_ylim(ylim)

    
    def plot_tracker(self, which, xax, yax, where=None, **set_kwargs):

        if which == '1':
            tracker = self.tracker1
        elif which == '2':
            tracker = self.tracker2

        for ax, coord in zip([xax, yax], ['x', 'y']):
            tracker.coordinate_timeseries(ax, coord,
                xlabel='', ylabel=f'T{which} {coord.upper()}',
                **set_kwargs
            )
            if where is not None:
                self.fill_time(ax, where)


    def plot_cov(self, which, ax, where=None, **set_kwargs):
        """
        which : str
            'xx', 'xy', 'yx', or 'yy'.
        """
        
        cov = self.covs[which]

        ax.plot(self.times, cov, color='black', lw=1)
        ax.tick_params(axis='x', which='both', direction='in', top=True)
        ax.tick_params(axis='y', which='both', direction='in', right=True)
        ax.set(ylabel=f'N. Cov(${which}$)', **set_kwargs)
        ptools.set_x_ticks(ax)
        ptools.set_y_ticks(ax)

        if where is not None:
            self.fill_time(ax, where)

    
    def plot_amplitude(self, ax, where=None, **set_kwargs):

        ax.plot(self.times, self.amplitude, color='black', lw=1)
        ax.axhline(self.threshold,
            color='blue', ls='dotted', lw=1,
            label='Threshold'
        )
        ax.set(**set_kwargs)
        ax.tick_params(axis='x', which='both', direction='in', top=True)
        ax.tick_params(axis='y', which='both', direction='in', right=True)
        ptools.set_x_ticks(ax)
        ptools.set_y_ticks(ax)
        ax.legend(fontsize=10)
        
        if where is not None:
            self.fill_time(ax, where)


    def plot_amplitude_histogram(self, ax, bin_width=0.001,
        b_plot_threshold=True, **set_kwargs):

        default_kwargs = dict(
            xlim=[0, ax.get_xlim()[1]],
            xlabel='Amplitude',
            ylabel='Counts',
            yscale='log'
        )
        set_kwargs = {**default_kwargs, **set_kwargs}

        ax.stairs(
            *self.compute_amplitude_hist(bin_width),
            fill=True, color='lightblue'
        )
        ax.set(**set_kwargs)
        ax.tick_params(axis='x', which='both', direction='in', top=True)
        ax.tick_params(axis='y', which='both', direction='in', right=True)

        if b_plot_threshold:
            ax.axvline(self.threshold, color='blue',
                lw=1, ls='dotted', label='Threshold')
            
            # self._compute_threshold_old()
            # ax.axvline(self.threshold, color='gray', 
            #     lw=1, ls='dashed', label='old threshold')
            
            p = np.percentile(self.amplitude, 99)
            ax.axvline(p, color='green', ls='dashed', label='99th percentile')

            ax.legend()

    
    def make_overview(self, time_range=None):

        fig = plt.figure(figsize=(12,10), constrained_layout=True)
        
        spacing = dict(top=0.92, bottom=0.1, wspace=0.3)
        height_ratios = [1, 1, 1, 1, 2, 2]
        gs_top = gridspec.GridSpec(6, 2,
            height_ratios=height_ratios, **spacing)
        gs_bottom = gridspec.GridSpec(6, 2,
            hspace=0.5, height_ratios=height_ratios)

        where = self.amplitude > self.threshold
        if time_range is None:
            xlim = self.tracker1.observation_time
        else:
            xlim = [
                utilities.convert_string_to_datetime(time_range[0]),
                utilities.convert_string_to_datetime(time_range[1])
            ]

        # Plot tracker data.
        for i in range(2):
            xax = fig.add_subplot(gs_top[i,0])
            yax = fig.add_subplot(gs_top[i,1])
            self.plot_tracker(str(i+1), xax, yax, where, xlim=xlim, xticklabels=[])

        # Plot cov data.
        for i, key in enumerate(self.covs.keys(), start=2):
            r,c = i-i//2-i%2+1,i%2
            ax = fig.add_subplot(gs_top[r,c])
            self.plot_cov(key, ax, where, xlim=xlim)
            if key[0] != 'y':
                ax.set(xticklabels=[])

        # Plot amplitude
        ax = fig.add_subplot(gs_bottom[4,:])
        self.plot_amplitude(ax, where,
            xlim=xlim,
            ylim=[0, np.max(self.amplitude)],
            ylabel=r'$\sqrt{\Sigma (\mathrm{Cov}^2)}$'
        )

        # Histogram
        ax = fig.add_subplot(gs_bottom[5,:])
        self.plot_amplitude_histogram(ax, xlim=[0,0.2])

        fig.align_ylabels()
        fig.suptitle(f'{self.date}, {self.id_num} - FPM{self.fpm}')
        # fig.patch.set_alpha(0)
        # plt.savefig(f'./{self.date}_{self.id_num}_fpm{self.fpm}_trackers.png', dpi=400, bbox_inches='tight')

        return fig, gs_top, gs_bottom