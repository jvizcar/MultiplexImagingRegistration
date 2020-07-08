"""
Utility functions specific for this project.
"""
import re

from os.path import join
from os import listdir

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# global variables
FILE_REGEX = re.compile('R(?P<round>\d{1,2})_.*_c(?P<channel>\d)_ORG.*.tif')


def parse_tif_dir(data_dir):
    """Read files in a ome tiff data directory (tif image per channel), especifically for the OHSU datasets.
    Function returns a dictionary with keys for rounds and channels parsed_dir[round#][channel#] and the values
    being the path to the file.

    Parameters
    ----------
    data_dir : str
        path to directory with the tiff images

    Return
    ------
    parsed_dir : dict
        dictionary containing the image files in nested format, first level keys are the round numbers followed
        by the channel key. Quenched rounds are ignored.

    """
    parsed_dict = {}

    for filename in listdir(data_dir):
        m = FILE_REGEX.search(filename)

        if m:
            m.groupdict()

            _round, channel = int(m['round']), int(m['channel'])

            # seed the round dict if not created
            if _round not in parsed_dict:
                parsed_dict[_round] = {}

            # add the channel to round
            parsed_dict[_round][channel] = join(data_dir, filename)

    return parsed_dict


def show_keypoints(moving_im, target_im, moving_kpts, target_kpts, size=50):
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
        circ = Circle((xy[0], xy[1]), size, color=(1, 0, 0))
        ax[2].add_patch(circ)

    for xy in target_kpts:
        circ = Circle((xy[0], xy[1]), size, color=(1, 0, 0))
        ax[3].add_patch(circ)

    plt.show()
