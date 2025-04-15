#!/usr/bin/env python3

# Gif creation: https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
# On quality loss: https://stackoverflow.com/questions/52948735/quality-loss-in-imageio

import os
import imageio

from ..utils import utilities


CONF_FILE = utilities.CONF_FILE
MOVIE_CONFIG = {
    'FPS': 10,
    'EXTS': ['.gif', '.mp4']
}


# TODO: Remake this method so that any folder containing images can be given to it, it is not robust enough...
def make_movies(fig_dir, frame_folder_name):
    """
    Creates movies using the .png files in the specified directory.
    Currently **only** .png files can be used to make movies.

    # Either a folder name containing png files must be provided or
    # a list containing png file names.
    # If both are provided, the files in the png_files list will be used.

    Parameters
    ----------
    fig_dir : str
        The directory of the 'figures' folder.
    frame_folder_name : str
        The name of the folder which contains the png files.
        Note: The movies will be placed in this folder with
        the same name as the folder.

    Ex: make_movies('/Users/rbmasek/nustar/data/20150901/20110116001/figures/', 'frames_10s_A_bc_averaged')
    """

    utilities.apply_config_settings(MOVIE_CONFIG, 'MovieSettings', CONF_FILE)
    extensions = []
    if MOVIE_CONFIG['B_MAKE_GIF']:
        extensions.append('.gif')
    if MOVIE_CONFIG['B_MAKE_MP4']:
        extensions.append('.mp4')
    if not extensions:
        extensions.append('.gif')
        print('No movie extensions were specified; defaulting to .gif')

    # Get the contents of the folder containing the png files.
    in_dir = fig_dir + frame_folder_name + '/'  # + '/frame_images/'
    dir_contents = os.listdir(in_dir)
    png_files = []

    # Gather all of the .png files to be included.
    # Note: **ALL** png files in the folder will be included.
    for f in dir_contents:
        if f.endswith('.png'):
            png_files.append(f)

    # Sort the files so they are in chronological order.
    png_files.sort()

    # Make a movie file with each of the specified extensions.
    for ext in extensions:
        movie_name = fig_dir + frame_folder_name + '/' + frame_folder_name + ext

        # Write the png files to the movie file.
        # , codec='libx264', quality=10, pixelformat='yuv444p') # or yuvj444p
        writer = imageio.get_writer(movie_name, fps=MOVIE_CONFIG['FPS'])
        for f in png_files:
            writer.append_data(imageio.imread(in_dir+f))
        writer.close()

        print('Movie saved to ' + movie_name)


def make_cluster_movie(input_dir, min_frame, max_frame, output_dir='', movie_name_prefix='cluster_movie_'):
    """
    This generates a movie using cluster images stored at input_dir.
    """

    utilities.apply_config_settings(MOVIE_CONFIG, 'MovieSettings', CONF_FILE)
    extensions = []
    if MOVIE_CONFIG['B_MAKE_GIF']:
        extensions.append('.gif')
    if MOVIE_CONFIG['B_MAKE_MP4']:
        extensions.append('.mp4')
    if not extensions:
        extensions.append('.gif')
        print('No movie extensions were specified; defaulting to .gif')

    if output_dir == '':
        output_dir = input_dir

    utilities.verify_path(input_dir)
    utilities.verify_path(output_dir)

    base_str = 'clustermap_frame'
    png_files = []

    # Obtain the relevant png files.
    for i in range(min_frame, max_frame + 1):
        fig_str = base_str + str(i) + '.png'
        if os.path.exists(input_dir + fig_str):
            png_files.append(fig_str)

    # Repeat the last frame for 2 more seconds.
    if png_files:
        png_files += [png_files[-1]]*MOVIE_CONFIG['FPS']*2
        png_files.sort()  # Order chronologically

        # Make a movie file with each of the specified extensions.
        for ext in extensions:
            movie_name = output_dir + movie_name_prefix + str(min_frame) + \
                '-' + str(max_frame) + ext

            # Write the png files to the movie file.
            # , codec='libx264', quality=10, pixelformat='yuv444p') # or yuvj444p
            writer = imageio.get_writer(
                movie_name, fps=MOVIE_CONFIG['FPS'], mode='I')
            for f in png_files:
                writer.append_data(imageio.imread(input_dir + f))
            writer.close()

            print('Movie saved to ' + movie_name)

    else:
        print('No cluster maps found for the specified path and frame range.')
        print('Input directory:', input_dir)
        print('Frame range: ' + str(min_frame) + '-' + str(max_frame))


def make_event_movie(event, clusters_dir):

    utilities.apply_config_settings(MOVIE_CONFIG, 'MovieSettings', CONF_FILE)
    make_cluster_movie(clusters_dir, event.get_start_frame(), event.get_end_frame(),
                       output_dir=event.event_dir, movie_name_prefix='event_movie_')
