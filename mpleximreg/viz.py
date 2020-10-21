"""
Functions used for visualizing registration images.

Includes some interactive functions for overlaying register images with some control.
"""
from .utils import parse_tif_dir, normalize_image
from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact, Dropdown, IntSlider, fixed


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
