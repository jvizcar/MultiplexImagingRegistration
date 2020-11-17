"""
Functions used for visualizing registration images.

Includes some interactive functions for overlaying register images with some control.
"""
from .utils import parse_tif_dir, normalize_image
from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact, Dropdown, IntSlider, fixed
import seaborn as sns


def interact_tif_dirs(tiff_dirs):
    """Interactor for applying visualization between two tiff images from different rounds but same channel.
    Allow quickly switching between register and non-register directories by providing a list of tiff_dirs."""
    _ = interact(
        visualize_tif_dir,
        tif_dir=Dropdown(options=tiff_dirs, value=tiff_dirs[0], description='Dir'),
        round1=IntSlider(min=0, max=11, value=1, continuous_update=False, description='Round 1'),
        round2=IntSlider(min=0, max=11, value=2, continuous_update=False, description='Round 2'),
        channel=IntSlider(min=1, max=5, value=1, continuous_update=False, description='Channel'),
        threshold1=fixed(5),
        threshold2=fixed(5),
        multi=fixed('Both')
    )


def visualize_tif_dir(tif_dir, round1, round2, channel, threshold1=5, threshold2=5, multi='Both'):
    """Visualize two images at a time from a tiff directory. Choose 2 rounds to visualize and the channel to visualize
    and provide some threshold controls if desired.

    """
    tiff_filepaths = parse_tif_dir(tif_dir)

    # read the two images -- normalize the images so the range pixel values are from 0 to 255
    im1 = normalize_image(imread(tiff_filepaths[round1][channel]))
    im2 = normalize_image(imread(tiff_filepaths[round2][channel]))

    # force the images to be the same size by clipping to the min dim for both
    h1, w1 = im1.shape
    h2, w2 = im2.shape
    h, w = min(h1, h2), min(w1, w2)
    im1 = im1[0:h, 0:w]
    im2 = im2[0:h, 0:w]

    # an RGB image will be colored based on the threshold of the images
    im_rgb = np.zeros((im1.shape[0], im1.shape[1], 3), dtype=np.uint8)

    # color the pixels based on if the original images pass certain value thresholds
    # use green for pixels that passed the threshold for both images
    # red for pixels that passed threshold only on image one and blue for only image 2
    # if multi is not both then it has to be either First (to display image 1 in original form) or Second
    if multi == 'Both':
        im_rgb[im1 > threshold1] = (255, 0, 0)
        im_rgb[im2 > threshold2] = (0, 0, 255)
        im_rgb[(im1 > threshold1) & (im2 > threshold2)] = (0, 255, 0)
    elif multi == 'First':
        im_rgb = im1
    elif multi == 'Second':
        im_rgb = im2
    else:
        raise ValueError('multi parameter must be Both, First, or Second')

    # plot the figures on top of each other, control the hue of the second image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(im_rgb)
    plt.show()

    if multi == 'Both':
        print('Green - both images     Red - first image only     Blue - second image only')


def plot_swarm_plots(df, groupcol, scorecol, order=None, figsize=(10, 10), hue=None):
    """Plot the swarm box plots from the hackathon results (errors).

    Parameters
    ----------
    df : pandas.DataFrame
        the dataframe with the data to plot
    groupcol : str
        the column to use for separating the rows / entries into groups
    scorecol : str
        the column used to plot the y-axis (contniuous variable)
    order : list (default: None)
        the groups to plot and in the order to plot them, otherwise all possible groups are plotted in the order they
        appear in the column
    figsize : tuple (default: (10, 10))
        the figure size
    hue : str (default: None)
        the column used to group categories by, the second grouping column, if None then a second grouping won't be used

    Return
    ------
    fig : matplotlib.fig
        the figure canvas of the plot
    ax : matplotlib.ax
        the axis of the figure

    """
    fig, ax = plt.subplots(figsize=figsize)

    if order is None:
        order = list(set(df[groupcol].tolist()))

    sns.swarmplot(x=groupcol, y=scorecol, data=df, ax=ax, order=order, hue=hue, size=7)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    return fig, ax


def pretty_swarm_plot(df, order=('Unregistered', 'Rigid Registration with DAPI', 'Nonrigid Registration'),
                      save_path=None, hue='Dataset', figsize=(10, 10), ylims=None):
    """A custom wrapper function around plot_swarm_plot custom to my project.

    """
    # plot for TRE
    fig, ax = plot_swarm_plots(df, 'Registration Method', 'TRE', order=order, hue=hue, figsize=figsize)
    ax.set_xlabel('')
    ax.set_ylabel('TRE', fontsize=18)
    ax.set_title('TRE for hackathon recreation', fontsize=20, fontweight='bold')
    ax.set_xticklabels(order, fontsize=16)

    if ylims is not None:
        ax.set_ylim(ylims[0])

    if hue is not None:
        ax.legend(fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path + '_TRE.png', dpi=300)
    plt.show()

    # plot for the error in micrometers
    fig, ax = plot_swarm_plots(df, 'Registration Method', 'Mean Error (um)', order=order, hue=hue, figsize=figsize)
    ax.set_xlabel('')
    ax.set_ylabel('Mean error distance (\u03bcm)', fontsize=18)
    ax.set_title('Mean error for hackathon recreation', fontsize=20, fontweight='bold')
    ax.set_xticklabels(order, fontsize=16)

    if ylims is not None:
        ax.set_ylim(ylims[1])

    if hue is not None:
        ax.legend(fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path + '_Error.png', dpi=300)
    plt.show()
