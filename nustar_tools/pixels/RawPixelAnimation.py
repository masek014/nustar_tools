import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import photutils

from matplotlib import animation

import nustar_pysolar as nustar

from ..utils import utilities
from .PixelArray import RawPixelArray


class RawPixelAnimation():


    def __init__(self,
        id_dir,
        fpm,
        time_step,
        frame_length,
        cmaps=['Blues', 'Greens', 'Oranges', 'Reds']
    ):
        
        self.id_dir = id_dir
        self.id_num = utilities.get_id_from_id_dir(id_dir)
        self.fpm = fpm
        self.time_step = time_step
        self.frame_length = frame_length

        self.evt_file = utilities.EVT_FILE_PATH_FORMAT.format(
            id_dir=self.id_dir,
            id_num=self.id_num,
            fpm=self.fpm
        )
        self.evt_data, self.hdr = utilities.get_event_data(self.evt_file)
        self.start, self.end, num_frames = utilities.characterize_frames(self.evt_data, time_step)
        self.pixel_array = RawPixelArray(self.evt_data, self.hdr)
        self.axs, self.mats = self.pixel_array.plot_det_counts(cmaps=cmaps)
        self.fig = plt.gcf()

        self.start_frame = 0
        self.end_frame = num_frames - 1 - frame_length//time_step


    def time_filter(self, time_range):

        return self.evt_data[nustar.filter.by_time(self.evt_data, self.hdr, time_range)]


    def animate(self, out_dir='./', fps=20, dpi=400):
        
        def update_mats(frame, self):

            start_time = self.start + self.time_step*(frame)
            frame_start = utilities.convert_nustar_time_to_astropy(start_time)
            frame_end = frame_start + self.frame_length*u.second
            det_arrs = self.pixel_array.make_det_counts((frame_start, frame_end))
            self.pixel_array.fig_text.set_text(f'{frame_start}-{frame_end}')

            for det in range(4):
                self.mats[det].set_data(np.fliplr(det_arrs[det]))
                self.mats[det].set_norm(None)

            return self.mats

        date = utilities.convert_nustar_time_to_datetime(self.start).strftime('%Y%m%d')
        file_name = f'{out_dir}/{date}_{self.id_num}_fpm{self.fpm}_{self.time_step}_{self.frame_length}_rawpixels.mp4'

        anim = animation.FuncAnimation(self.fig, update_mats,
            fargs=(self,), frames=self.end_frame, interval=1, blit=True)
        FFwriter = animation.FFMpegWriter(fps=fps, codec='mpeg4')
        anim.save(file_name, dpi=dpi, writer=FFwriter)


# Version with the centroid option. Determine if we want to keep this...
class RawPixelAnimationCentroid(RawPixelAnimation):

    def __init__(self,
        id_dir,
        fpm,
        time_step,
        frame_length,
        cmaps=['Blues', 'Greens', 'Oranges', 'Reds']
    ):
        super().__init__(id_dir, fpm, time_step, frame_length, cmaps)

        rows, cols = 2, 2
        ratios = [1, 1]
        figsize = (4,4)
        spacing = dict(left=0.03, bottom=0.01, right=0.97, top=0.95, wspace=0.02, hspace=0.02)
        rows = 3
        ratios.append(0.4)
        figsize = (4,4.8)
        spacing = dict(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.02, hspace=0.02)

        self.fig = plt.figure(figsize=figsize)
        self.spec = self.fig.add_gridspec(rows, cols, height_ratios=ratios, **spacing)
        self.mats = []


    def time_filter(self, time_range):

        return self.evt_data[nustar.filter.by_time(self.evt_data, self.hdr, time_range)]


    def animate(self, out_dir='./', fps=20, dpi=400):
        
        def init_animation():
            
            plt.clf() # init is called twice in FuncAnimation, this prevents double drawing the plots
            self.axs = []
            init_arrs = []
            for det in [0, 1, 2, 3]:
                
                row = int(det // 2)
                col = det % 2
                ax = self.fig.add_subplot(self.spec[row,col])
                arr = np.zeros(shape=(32,32))
                self.axs.append(ax)
                init_arrs.append(arr)

            axs = np.array(self.axs).reshape((2,2))
            _, self.mats = self.pixel_array.plot_det_counts(axs=axs)
            
            ax = self.fig.add_subplot(self.spec[2,:])
            line = ax.plot([],[], color='black', marker='o',
                linewidth=0.4, markersize=1)[0]
            ax.tick_params(which='major', length=2, width=0.3, direction='in')
            ax.tick_params(which='minor', length=1, width=0.3, direction='in')
            formatter = matplotlib.dates.DateFormatter('%H:%M:%S')
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
            self.axs.append(ax)
            self.mats.append(line)

            return self.mats


        def update_mats(frame, self):

            start_time = self.start + self.time_step*(frame)
            frame_start = utilities.convert_nustar_time_to_astropy(start_time)
            frame_end = frame_start + self.frame_length*u.second
            frame_data = self.time_filter((frame_start, frame_end))

            det_arrs = self.pixel_array.make_det_counts((frame_start, frame_end))
            self.fig.suptitle(f'{frame_start}-{frame_end}')
            for det in range(det_arrs.shape[0]):
                self.mats[det].set_data(np.fliplr(det_arrs[det]))
                self.mats[det].set_norm(None)
            
            line = self.mats[-1]
            dt = utilities.convert_nustar_time_to_datetime(start_time)
            x_centroid = photutils.centroids.centroid_com(np.fliplr(det_arrs[0]))[0]
            x = list(line.get_xdata()) + [dt]
            y = list(line.get_ydata()) + [x_centroid]
            xrange = (x[0], x[-1])
            line.set_data(x, y)
            self.axs[-1].set(xlim=xrange, ylim=(np.nanmin(y), np.nanmax(y)))

            return self.mats

        date = utilities.convert_nustar_time_to_datetime(self.start).strftime('%Y%m%d')
        file_name = f'{out_dir}/{date}_{self.id_num}_{self.time_step}_{self.frame_length}_rawpixels_centroid.mp4'
            
        anim = animation.FuncAnimation(self.fig, update_mats, init_func=init_animation,
            fargs=(self,), frames=self.end_frame, interval=1, blit=True)
        FFwriter = animation.FFMpegWriter(fps=fps)
        anim.save(file_name, dpi=dpi, writer=FFwriter)