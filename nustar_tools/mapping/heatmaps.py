'''
Create plots of heatmaps.
'''

import matplotlib
import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
import numpy as np

from . import tools as mtools


def make_square(row_coords: list, col_coords: list) -> tuple[list, list, int]:
    '''Rebuild the provided coordinates lists so that the lists
    contain coordinates that make up a square,
    i.e. equal number of row coordinates as column coordinates.
    '''
    num_rows = max(row_coords) - min(row_coords) + 1
    num_cols = max(col_coords) - min(col_coords) + 1
    if num_rows != num_cols:
        diff = abs(num_rows - num_cols)
        left_pad = int(diff/2)
        right_pad = int(diff/2) + diff % 2
        if num_rows < num_cols:
            num_rows = num_cols
            for i in range(1, left_pad+1):
                row_coords.append(min(row_coords)-i)
            for i in range(1, right_pad+1):
                row_coords.append(max(row_coords)+i)
        else:
            num_cols = num_rows
            for i in range(1, left_pad+1):
                col_coords.append(min(col_coords)-i)
            for i in range(1, right_pad+1):
                col_coords.append(max(col_coords)+i)
    else:
        print('Number of rows and cols are equal, doing nothing.')

    return row_coords, col_coords, num_rows


def set_heatmap_ticks(ax: plt.Axes, x_labels: list[str], y_labels: list[str]):
    '''Set the tick placement and tick labels for a heatmap.

    Parameters
    ----------
    ax : matplotlib axis
        Axis to which the labels will be applied.
    x_labels : list of str
        Contains the labels for the x-axis.
    y_labels : list of str
        Contains the labels for the y-axis.
    '''
    # Show all ticks.
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)


def plot_heatmap(
    arr: np.ndarray,
    x_tick_labels: list[str] | None = None,
    y_tick_labels: list[str] | None = None,
    xlabel: str = 'X Coordinate',
    ylabel: str = 'Y Coordinate',
    title: str = '',
    count_labels: bool = True,
    fig_dir: str = './',
    file_name: str = 'heatmap',
    **cb_kwargs
) -> tuple[plt.Figure, plt.Axes]:
    '''
    Make a heatmap using the Macropixels in a given Event.
    The heatmap shows how many times a macropixel was included
    in the detection.

    Parameters
    ----------
    arr : np array
        The array containing the heatmap data.
    x_tick_labels : list
        The labels for the x-axis major ticks.
    y_tick_labels : list
        The labels for the x-axis major ticks.
    xlabel : str
        The label for the x-axis.
    ylabel : str
        The label for the y-axis.
    title : str
        The figure title.
    count_labels : bool
        Specifies whether the array values should
        be listed on the heatmap.
    fig_dir : str
        The path where the figure will be saved.
    file_name : str
        The name of the saved file.
    cb_kwargs : dict
        The keyword arguments for the colorbar.

    Returns
    -------
    fig : matplotlib figure
        The figure corresponding to the heatmap.
    ax : matplotlib axes
        The axes corresponding to the heatmap.
    '''

    default_kwargs = {
        'cmap': 'viridis',
        'norm': mplcolors.Normalize(np.min(arr), np.max(arr))
    }
    cb_kwargs = {**default_kwargs, **cb_kwargs}

    # Setup plot area.
    mtools.apply_style()
    fig, ax = plt.subplots()
    ax.tick_params(axis='both', which='major', pad=15)

    _ = ax.imshow(arr, **cb_kwargs)
    mtools.apply_colorbar(fig, ax, **cb_kwargs)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if x_tick_labels is None:
        x_tick_labels = []
    if y_tick_labels is None:
        y_tick_labels = []

    set_heatmap_ticks(ax, x_tick_labels, y_tick_labels)
    if count_labels:
        for i in range(len(x_tick_labels)):
            for j in range(len(y_tick_labels)):
                count_val = arr[i, j]
                c = 'black' if count_val > 0.9 * np.max(arr) else 'white'
                _ = ax.text(
                    j, i, count_val, fontsize=matplotlib.rcParams['font.size'],
                    ha='center', va='center', color=c)
    mtools.save_map(fig, fig_dir, file_name)

    return fig, ax
