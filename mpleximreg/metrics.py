"""
Metrics used to estimate how close two images are registered to each other.

May include wrapper functions for SimpleITK implementations.
"""
import numpy as np


def target_registration_error(moving_pts, target_pts, im_shape):
    """Target Registration Error. Parameters must be matched keypoints between moving and target image.

    Parameters
    ----------
    moving_pts : array-like
        moving image x, y coordinates
    target_pts : array-like
        target image x, y coordinates
    im_shape : tuple
        height and width of moving image

    Return
    ------
    tre : float
        Target Registration Error

    """
    tre = np.mean(
        np.sqrt(np.sum((moving_pts - target_pts) ** 2, axis=1)) / np.sqrt(im_shape[0] ** 2 + im_shape[1] ** 2)
    )

    return tre
