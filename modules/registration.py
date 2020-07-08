"""
Functions for performing different registration approaches and calculating performance metrics.
"""
import numpy as np
import cv2
from skimage import transform

# KAZE features are used as the default keypoints
FD = cv2.KAZE_create(extended=True)
LINEAR_TRANSFORMER = transform.SimilarityTransform()


def register_images(moving_im, target_im, methods=None):
    """Register images to each other using various methods. For each method an error distance (TRE) is calculated
    between the images.

    Parameters
    ----------
    moving_im : numpy array
        image that will be registered to match target image
    target_im : numpy array
        target image that moving image will be registered to
    methods : list (default: None)
        list of methods used for registration. Default None value will lead to no registration being done. Options
        include 'linear' and 'none' (default).

    Return
    ------
    images : list
        list of images after registration (or original moving image if 'none' or default None passed). Last index is
        always the target image.
    kpts : list
        list of keypoints used on calculating the TRE distance for each method
    distances : list
        the TRE error between moving and target image, also referred to as distances, estimated using Kaze feature brute
        force matching

    """
    # seed return variables
    images, kpts, distances = [], [], []

    # default method is no registration
    if methods is None:
        methods = ['none']

    # matching keypoints should always be done
    try:
        moving_kpts, target_kpts = match_keypoints(moving_im, target_im, FD)
        h, w = moving_im.shape
    except:
        raise Exception('keypoint matching failed')

    for method in methods:
        if method == 'none':
            images.append(moving_im.copy())
            kpts.append(moving_kpts)
            distances.append(tre_distance(moving_kpts, target_kpts, h, w))
        elif method == 'linear':
            warped_im, warped_kpts = apply_transform(moving_im, target_im, moving_kpts, target_kpts, LINEAR_TRANSFORMER)
            images.append(warped_im)
            kpts.append(warped_kpts)
            distances.append(tre_distance(warped_kpts, target_kpts, h, w))
        else:
            raise Exception(f'method {method} not supported')

    return images, kpts, distances


def tre_distance(moving_pts, target_pts, img_h, img_w):
    """Calculate the Target Registration Error (TRE) given corresponding keypoints for each image. The keypoints
    should be matched between the images.

    Parameters
    ----------
    moving_pts : array
        coordinates of the keypoints for the moving image
    target_pts : array
        coordinates of the keypoints for the target image
    img_h : int
        height of the moving image
    img_w : int
        width of the moving image

    Return
    ------
    tre : float
        the Target Registration Error

    """
    dst = np.sqrt(np.sum((moving_pts - target_pts) ** 2, axis=1)) / np.sqrt(img_h ** 2 + img_w ** 2)
    tre = np.mean(dst)
    return tre


def match_keypoints(moving, target, feature_detector):
    """Calculate image keypoints that are simimlar between two images.

    Parameters
    ----------
    moving : numpy array
        image that is to be warped to align with target image
    target : numpy array
        image to which the moving image will be aligned
    feature_detector : opencv object
        a feature detector from opencv that has detectAndCompute method

    Returns
    -------
    filtered_src_points : numpy array
        x, y locations of matched keypoints for moving image
    filtered_dst_points : numpy array
        x, y locations of matched keypoints for target image

    """
    # get keypoints for each image
    kp1, desc1 = feature_detector.detectAndCompute(moving, None)
    kp2, desc2 = feature_detector.detectAndCompute(target, None)

    # brute force match descriptors between the images
    matcher = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(desc1, desc2)

    src_match_idx = [m.queryIdx for m in matches]
    dst_match_idx = [m.trainIdx for m in matches]

    # use the matcher to keep keypoints that matched between images - organize the keypoints so they align
    src_points = np.float32([kp1[i].pt for i in src_match_idx])
    dst_points = np.float32([kp2[i].pt for i in dst_match_idx])

    # find homography mask and filter keypoints (unsure how this works yet)
    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, ransacReprojThreshold=10)

    good = [matches[i] for i in np.arange(0, len(mask)) if mask[i] == [1]]

    filtered_src_match_idx = [m.queryIdx for m in good]
    filtered_dst_match_idx = [m.trainIdx for m in good]

    filtered_src_points = np.float32([kp1[i].pt for i in filtered_src_match_idx])
    filtered_dst_points = np.float32([kp2[i].pt for i in filtered_dst_match_idx])

    return filtered_src_points, filtered_dst_points


def apply_transform(moving, target, moving_pts, target_pts, transformer, output_shape_rc=None):
    """Apply a transform to an image given moving and target image keypoints and transformer parameter form the
    scikit-image package.

    Parameters
    ----------
    moving : numpy array
        image to transform
    target : numpy array
        image that moving image will transform into
    moving_pts : numpy array
        x, y locations of matched keypoints for moving image
    target_pts : numpy array
        x, y locations of matched keypoints for target image
    transformer : skimage tranform object
        See https://scikit-image.org/docs/dev/api/skimage.transform.html for different transformations
    output_shape_rc : tuple
        (shape of warped image (row, col). If None, use shape of target image

    Returns
    -------
    warped_im : numpy array
        the moving image warped to match the target image
    warped_pts : numpy array
        the x, y location of the warped keypoints

    """
    if output_shape_rc is None:
        output_shape_rc = target.shape[:2]

    if str(transformer.__class__) == "<class 'skimage.transform._geometric.PolynomialTransform'>":
        transformer.estimate(target_pts, moving_pts)
        warped_img = transform.warp(moving, transformer, output_shape=output_shape_rc)

        ### Restimate to warp points
        transformer.estimate(moving_pts, target_pts)
        warped_pts = transformer(moving_pts)
    else:
        transformer.estimate(moving_pts, target_pts)
        warped_img = transform.warp(moving, transformer.inverse, output_shape=output_shape_rc)
        warped_pts = transformer(moving_pts)

    return warped_img, warped_pts
