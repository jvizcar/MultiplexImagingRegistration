"""
Utility functions specific for this project.

List of functions:
* parse_tif_dir
* show_keypoints
* plot_with_keypoints
"""
import re
import numpy as np

from os.path import join
from os import listdir

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# set global matplotlib parameters
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.titlesize'] = 18


# global variables
FILE_REGEX = re.compile('R(?P<round>\d{1,2})_.*_c(?P<channel>\d)_ORG.*.tif')
FILE_REGEX_Q = re.compile('R(?P<round>\d{1,2}(Q|))_.*_c(?P<channel>\d)_ORG.*.tif')


def normalize_image(im):
    """Normalize an image into the range 0-255 as type np.uint8

    source: https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range

    Parameters
    ----------
    im : array-like
        this should be a 2D image from the ome.tiff (or similar) that have a wide range of values. Don't pass a regular
        RGB image or an image that is already in the range of 0 to 255

    Return
    ------
    normalized_im : array-like
        the normalized image

    """
    pixels = im.flatten()

    # scale pixels to range 0 to 1
    normalized_im = (pixels - np.min(pixels)) / (np.max(pixels) - np.min(pixels))

    # scale the pixels by 255
    normalized_im = (normalized_im.reshape(im.shape) * 255).astype(np.uint8)

    return normalized_im


def parse_tif_dir(data_dir, quench=False):
    """Read files in a ome tiff data directory (tif image per channel in round), specifically for the OHSU datasets.
    Function returns a dictionary with keys for rounds and channels (i.e. parsed_dir[round#][channel#]) and the values
    being the path to the file.

    Parameters
    ----------
    data_dir : str
        path to directory with the tiff images
    quench : bool (default: False)
        if True then quench rounds will be included in key R##Q

    Return
    ------
    parsed_dir : dict
        dictionary containing the image files in nested format, first level keys are the round numbers followed
        by the channel key

    """
    parsed_dict = {}

    file_regex = FILE_REGEX_Q if quench else FILE_REGEX

    for filename in listdir(data_dir):
        m = file_regex.search(filename)

        if m:
            m.groupdict()

            try:
                _round = int(m['round'])
            except ValueError:
                _round = m['round']

            channel = int(m['channel'])

            # seed the round dict if not created
            if _round not in parsed_dict:
                parsed_dict[_round] = {}

            # add the channel to round
            parsed_dict[_round][channel] = join(data_dir, filename)

    return parsed_dict


def show_keypoints(moving_im, target_im, moving_kpts, target_kpts, size=10):
    """Show the moving and taget images and corresponding images with keypoints shown as red dots. This function is
    purely for visualization purposes.

    Parameters
    ----------
    moving_im : numpy array (2D)
        moving image
    target_im : numpy array (2D)
        target image
    moving_kpts : numpy array
        shape of (N, 2) where N is the number of keypoints, should contain the x, y coordinates of keypoints to show for
        moving image
    target_kpts : numpy array
        shape of (N, 2) where N is the number of keypoints, should contain the x, y coordinates of keypoints to show for
        target image
    size : int
        size of keypoints circles when plotting, adjust depending on your image size

    """
    # show results
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(15, 15))
    ax = ax.ravel()

    ax[0].imshow(moving_im, cmap='gray')
    ax[0].set_title('Moving Image', fontsize=14)
    ax[1].imshow(target_im, cmap='gray')
    ax[1].set_title('Target Image', fontsize=14)
    ax[2].imshow(moving_im, cmap='gray')
    ax[2].set_title('Keypoints (Moving Image)', fontsize=14)
    ax[3].imshow(target_im, cmap='gray')
    ax[3].set_title('Keypoints (Target Image)', fontsize=14)

    for xy in moving_kpts:
        circ = Circle((xy.pt [0], xy.pt[1]), size, color=(1, 0, 0))
        ax[2].add_patch(circ)

    for xy in target_kpts:
        circ = Circle((xy.pt[0], xy.pt[1]), size, color=(1, 0, 0))
        ax[3].add_patch(circ)

    plt.show()


def plot_with_keypoints(im, kpts, figsize=(10, 10), title='', color=(1., 0, 0), size=10):
    """Plot n number multiplex channel image (1 channel) with keypoints drawn on top of it.

    Parameters
    ----------
    im : array-like
        image
    kpts : list
        list of opencv keypoints
    figsize : tuple (default: (10, 10))
        size of the figure in matplotlib terms
    title : str
        title of the figure
    color : tuple (defaultu (1., 0, 0))
        the RGB color to use for keypoints (range from 0. to 1.)
    size : int (default: 10)
        size of the keypoints in matplotlib terms

    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(im, cmap='gray')

    for xy in kpts:
        circ = Circle((xy.pt[0], xy.pt[1]), size, color=color)
        ax.add_patch(circ)

    ax.set_title(title)

    plt.show()


def plot_two_images(moving_im, target_im, moving_kpts=None, target_kpts=None, figsize=(15, 10), title='',
                    color=(1., 0, 0), kpt_size=10):
    """Plot the moving and target images next to each other (horizontally) with the option to show keypoints as well.

    Parameters
    ----------
    moving_im : array-like
        moving image
    target_im : array-like
        target image
    moving_kpts : list (default: None)
        moving keypoints, if None then this is ignored
    target_kpts : list (default: None)
        moving keypoints, if None then this is ignored
    figsize : tuple (default: (10, 10))
        size of the figure in matplotlib terms
    title : str
        title of the figure
    color : tuple (defaultu (1., 0, 0))
        the RGB color to use for keypoints (range from 0. to 1.)
    kpt_size : int (default: 10)
        size of the keypoints in matplotlib terms

    Return
    ------
    fig : matplotlib.figure
        the figure object, can be used to save the figure locally

    """
    fig, ax = plt.subplots(ncols=2, figsize=figsize)
    plt.subplots_adjust(top=1.15)  # adjust location of figure title in relationship wiht the subplots

    ax[0].imshow(moving_im, cmap='gray')
    ax[1].imshow(target_im, cmap='gray')
    ax[0].set_title('Moving Image')
    ax[1].set_title('Target Image')

    if moving_kpts is not None:
        for xy in moving_kpts:
            circ = Circle((xy.pt[0], xy.pt[1]), kpt_size, color=color)
            ax[0].add_patch(circ)

    if target_kpts is not None:
        for xy in target_kpts:
            circ = Circle((xy.pt[0], xy.pt[1]), kpt_size, color=color)
            ax[1].add_patch(circ)

    plt.suptitle(title)
    plt.show()
    return fig
