"""
Functions for performing different registration approaches and calculating performance metrics.

By convention all functions that return both moving and target variables will return moving first followed by the
target variable. Pay attention to the docstring to avoid unwanted errors.

List of functions:
* register_images
* tre_distance
* match_keypoints
* apply_transform
* register_images_adv
"""
import numpy as np
import cv2
import SimpleITK as sitk
from shutil import rmtree
from imageio import imwrite, imread
from pandas import DataFrame
from skimage import transform, util
from copy import deepcopy
from skimage import img_as_uint

from os import makedirs
from os.path import join

from .utils import parse_tif_dir, normalize_image


def nonrigid_params(moving_filepath, target_filepath, txt_file_dir, spacing=.2645833333):
    """Obtain the transformation required to register two images by providing the moving and target image. Note that
    target and fixed are analogous terms in the context of the function.

    Parameters
    ----------
    moving_filepath : str
        filepath to the moving image
    target_filepath : str
        filepath to the target image
    txt_file_dir : str
        path to directory with txt files for doing non-rigid registration - this is specific to the hackathon and should
        contain the rigid_largesteps.txt and nl.txt files
    spacing : float
        the spacing to be applied to the x and y directions of the images

    Return
    ------
    SimpleElastix parameter object for registering the moving image to the target image

    """
    rigid_textfile = join(txt_file_dir, 'rigid_largersteps.txt')
    nonrigid_textfile = join(txt_file_dir, 'nl.txt')

    moving_im = sitk.ReadImage(moving_filepath)
    target_im = sitk.ReadImage(target_filepath)

    # set the spacing between the pixels in both x and y dimensions
    moving_im.SetSpacing((spacing, spacing))
    target_im.SetSpacing((spacing, spacing))

    # initiate the object to apply filters to the image
    selx = sitk.ElastixImageFilter()

    # set the fixed (target) and moving image to the object
    selx.SetMovingImage(moving_im)
    selx.SetFixedImage(target_im)

    # add the registration to do in sequential order by reading directly from the text file that contains all the
    # instructions -- this approach uses first an affine transformation followed by a non-rigid transformation
    pmap = sitk.ReadParameterFile(rigid_textfile)
    selx.SetParameterMap(pmap)
    pmap = sitk.ReadParameterFile(nonrigid_textfile)
    selx.AddParameterMap(pmap)

    # create a temporary location to store the files execute creates
    tempfilepath = '.temp'
    makedirs(tempfilepath, exist_ok=True)
    selx.SetOutputDirectory(tempfilepath)

    # execture the filters
    selx.Execute()
    rmtree(tempfilepath)

    # return the transformation parameters
    return selx.GetTransformParameterMap()


def nonrigid_transform(im_filepath, transform_params, spacing=.2645833333, save_filepath=None):
    """Applying a SimpleElastix transform to an image.

    Parameters
    ----------
    im_filepath : str
        filepath to the image to transform
    transform_params : SimpleElastix object
        SimpleElastix parameter object for registering the moving image to the target image
    spacing : float
        the spacing to be applied to the x and y directions of the images
    save_filepath : str (default: None)
        if not None then the image will be saved to this location

    Return
    ------
    transformed_im : array-like
        the transformed image

    """
    # get the dtype of the image before transform
    datatype = imread(im_filepath).dtype

    # initiate object to apply transformation and set the parameters to the given param object
    selx = sitk.TransformixImageFilter()
    selx.SetTransformParameterMap(transform_params)

    # set the moving image and transform it
    im = sitk.ReadImage(im_filepath)
    im.SetSpacing((spacing, spacing))

    selx.SetMovingImage(im)
    selx.Execute()

    # convert the image to a numpy array for saving
    transformed_im = selx.GetResultImage()
    transformed_im = sitk.GetArrayFromImage(transformed_im)

    if save_filepath is not None:
        # save the transformed image but make sure it is of the same datatype as the original image was
        imwrite(save_filepath, transformed_im.astype(datatype))

    return transformed_im


def nonrigid_transform_dir(im_dir, save_dir, txt_file_dir, target_round=1, reg_channel=2):
    """Wrapper around the nonrigid registration functions -- apply nonrigid registration on an entire directory of
    tiff image files in the OHSU format of file naming.

    Parameters
    ----------
    im_dir : str
        directory with the images to use, the filenaming should be in the convention used by OHSU
    save_dir : str
        location to save the registered images - the naming will be kept from the original images
    txt_file_dir : str
        path to directory with txt files for doing non-rigid registration - this is specific to the hackathon and should
        contain the rigid_largesteps.txt and nl.txt files
    target_round : int (default: 1)
        the round to use for the target
    reg_channel : int (default: 2)
        the channel to use in each round to get the transformation parameters

    """
    # create the save location
    makedirs(save_dir, exist_ok=True)

    im_filepaths = parse_tif_dir(im_dir)

    target_filepath = im_filepaths[target_round][reg_channel]

    for _round, channels in im_filepaths.items():
        moving_filepath = im_filepaths[_round][reg_channel]

        if _round not in (0, target_round):  # round 0 has no signal, safe to ignore
            print(f'Registering round {_round}')
            # get the transform params
            transform_params = nonrigid_params(moving_filepath, target_filepath, txt_file_dir)

            # apply the transform to all images in this round
            for filepath in channels.values():
                filename = filepath.split('/')[-1]
                _ = nonrigid_transform(filepath, transform_params, save_filepath=join(save_dir, filename))
        else:
            # save images from the target round without modification
            for filepath in channels.values():
                filename = filepath.split('/')[-1]
                imwrite(join(save_dir, filename), normalize_image(imread(filepath)))
                imwrite(join(save_dir, filename), normalize_image(imread(filepath)))


def get_kpts(im, method='AKAZE'):
    """Get the image keytpoints with the standard (AKAZE) method."""
    if isinstance(im, str):
        im = imread(im)

    if method == 'AKAZE':
        kpts, descs = cv2.AKAZE_create().detectAndCompute(im, None)
    elif method == 'KAZE':
        kpts, descs = cv2.KAZE_create(extended=True).detectAndCompute(im, None)
    else:
        raise Exception(f'Method {method} is not a valid keytpoint detection method.')
    return kpts, descs


def kpts_to_array(kpts):
    """Convert opencv keypoint list to an array for fast computations."""
    return np.float32([kpt for kpt in kpts])


def target_registration_error(shape, moving_kpts, target_kpts):
    """Calcualte the TRE between a moving and target image

    Shape (height, width) must be explicitly passed - it will not be inferred.
    """
    if isinstance(moving_kpts, list):
        moving_kpts = kpts_to_array(moving_kpts)

    if isinstance(target_kpts, list):
        target_kpts = kpts_to_array(target_kpts)

    # calculate the average distance between all martched points
    mean_distance = np.mean(
        np.sqrt(np.sum((moving_kpts - target_kpts) ** 2, axis=1))
    )

    tre = np.mean(
        np.sqrt(np.sum((moving_kpts - target_kpts) ** 2, axis=1)) / np.sqrt(shape[0] ** 2 + shape[1] ** 2)
    )

    return mean_distance, tre


def match_kpts(moving_im=None, target_im=None, target_kpts=None, target_desc=None, moving_kpts=None, moving_desc=None,
               method='AKAZE'):
    """Brute force match opencv keypoints

    This returns the matched keypoints in array form not in opencv list form.
    """
    # get the all keytpoints and description if not already passed
    if target_desc is None or target_kpts is None:
        if target_im is None:
            raise Exception("Pass the target image or its keypoints / descriptors")
        target_kpts, target_desc = get_kpts(target_im, method=method)

    if moving_kpts is None or moving_desc is None:
        if moving_im is None:
            raise Exception("Pass the moving image or its keypoints / descriptors")
        moving_kpts, moving_desc = get_kpts(moving_im, method=method)

    # match the descriptors
    matcher = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(moving_desc, target_desc)

    # subset only the matched points, as x, y coordinate points
    moving_match_idx = [m.queryIdx for m in matches]
    target_match_idx = [m.trainIdx for m in matches]

    moving_points = np.float32([moving_kpts[i].pt for i in moving_match_idx])
    target_points = np.float32([target_kpts[i].pt for i in target_match_idx])

    # filter the matched points using ransac mask
    mask = cv2.findHomography(moving_points, target_points, cv2.RANSAC, ransacReprojThreshold=10)[1]

    filtered_moving_points = np.float32([moving_points[i] for i in np.arange(0, len(mask)) if mask[i] == [1]])
    filtered_target_points = np.float32([target_points[i] for i in np.arange(0, len(mask)) if mask[i] == [1]])

    return filtered_moving_points, filtered_target_points


def tre_dir(im_dir, registration_channel, method='n/a', dataset='n/a', target_round=1, channel=1, save_path=None):
    """Calculate the TRE between a channel in each round as the moving images to a single target round channel.

    Parameters
    ----------
    im_dir : str
        directory with tif images
    registration_channel : int or str
        the channel used to register the directories
    method : str (default: 'n/a')
        name of registration approach
    dataset : str (default: 'n/a')
        name of the dataset being used
    target_round : int (default: 1)
        the target to use for registering
    channel : int (default: 1)
        channel to use for estimated the error, recommended to use the DAPI channel
    save_path : str (default: None)
        if not None then the DataFrame is saved to csv file

    Return
    ------
    df : pandas.DataFrame
        the dataframe with the results of calculating the registration error between the images in the directory

    """
    # get the target keypoints / descriptors - this is only done once
    im_filepaths = parse_tif_dir(im_dir)
    target_im = imread(im_filepaths[target_round][channel])
    target_kpts, target_desc = get_kpts(target_im)

    data = {'Registration Method': method, 'Target Round': target_round, 'Moving Round': [],
            'Registration Channel': registration_channel, 'Dataset': dataset, 'TRE': [], 'Mean Error (um)': []}

    #  loop through the rest of the images
    for _round, c in im_filepaths.items():
        if _round not in (0, target_round):
            print(f'Calculating for round {_round}')
            data['Moving Round'].append(f'R{_round}')

            moving_im = imread(im_filepaths[_round][channel])

            # get the matching keypoints between the target and moving image
            moving_pts, target_pts = match_kpts(moving_im=moving_im, target_kpts=target_kpts, target_desc=target_desc)

            # calculate the TRE
            mean_distance, error = target_registration_error(moving_im.shape[:2], moving_pts, target_pts)

            data['Mean Error (um)'].append(mean_distance)
            data['TRE'].append(error)

    df = DataFrame(data)

    if save_path is not None:
        df.to_csv(save_path, index=False)

    return df


def rigid_transformer(moving_filepath, target_filepath, target_kpts=None, target_desc=None, moving_kpts=None,
                      moving_desc=None):
    """Get the rigid transformation between a moving image and a target image using AKAZE keypoint brute force matching.

    Parameters
    ----------
    moving_filepath : str
        filepath to moving image
    target_filepath : str
        filepath to target image
    target_kpts : list
        keypoints for target image, if None then they will be extracted by the default methods
    target_desc : list
        descriptors to go along with target_kpts parameters
    moving_kpts : list
        keypoints for moving image, if None then they will be extracted by the default methods
    moving_desc : list
        descriptors to go along with moving_kpts parameters

    Returns
    -------
    transformer : skimage.SimilirityTransform()
        a scikit-image similarity transform that has been fit to transform the moving image to target image

    target_shape : tuple
        height, width of target image

    """
    moving_im = imread(moving_filepath)
    target_im = imread(target_filepath)

    # check need to extract features from images
    if target_desc is None or target_kpts is None:
        target_kpts, target_desc = cv2.AKAZE_create().detectAndCompute(target_im, None)

    if moving_desc is None or moving_kpts is None:
        moving_kpts, moving_desc = cv2.AKAZE_create().detectAndCompute(moving_im, None)

    # match keypoints
    matched_moving_kpts, matched_target_kpts = match_kpts(
        moving_kpts=moving_kpts, moving_desc=moving_desc, target_desc=target_desc, target_kpts=target_kpts
    )

    transformer = transform.SimilarityTransform()
    transformer.estimate(matched_moving_kpts, matched_target_kpts)
    target_shape = target_im.shape

    return transformer, target_shape


def rigid_transform(im_filepath, transformer, target_shape, save_filepath=None):
    """Applying a SimpleElastix transform to an image.

    Parameters
    ----------
    im_filepath : str
        filepath to the image to transform
    target_shape : tuple
        height and width of target image used to get the rigid transformer
    transformer : skimage.SimilirityTransform()
        a scikit-image similarity transform that has been fit to transform the moving image to target image
    save_filepath : str (default: None)
        if not None then the image will be saved to this location

    Return
    ------
    transformed_im : array-like
        the transformed image

    """
    im = imread(im_filepath)
    transformed_im = transform.warp(im, transformer.inverse, output_shape=target_shape)

    # save the image to file
    if save_filepath is not None:
        # convert image to uint before saving
        imwrite(save_filepath, img_as_uint(transformed_im))

    return transformed_im


def rigid_transform_dir(im_dir, save_dir, target_round=1, reg_channel=1):
    """Wrapper around the rigid registration function -- apply rigid registration on an entire directory of
    tiff image files in the OHSU format of file naming.

    Parameters
    ----------
    im_dir : str
        directory with the images to use, the filenaming should be in the convention used by OHSU
    save_dir : str
        location to save the registered images - the naming will be kept from the original images
    target_round : int (default: 1)
        the round to use for the target
    reg_channel : int (default: 1)
        the channel to use in each round to get the transformation parameters

    """
    # create the save location
    makedirs(save_dir, exist_ok=True)

    im_filepaths = parse_tif_dir(im_dir)

    target_filepath = im_filepaths[target_round][reg_channel]

    for _round, channels in im_filepaths.items():
        moving_filepath = im_filepaths[_round][reg_channel]

        if _round not in (0, target_round):  # round 0 has no signal, safe to ignore
            print(f'Registering round {_round}')
            # get the transformer
            transformer, target_shape = rigid_transformer(moving_filepath, target_filepath)

            # apply the transform to all images in this round
            for filepath in channels.values():
                filename = filepath.split('/')[-1]
                _ = rigid_transform(filepath, transformer, target_shape, save_filepath=join(save_dir, filename))
        else:
            # save images from the target round without modification
            for filepath in channels.values():
                filename = filepath.split('/')[-1]
                imwrite(join(save_dir, filename), imread(filepath))
                imwrite(join(save_dir, filename), imread(filepath))


def match_keypoints(moving_kpts, target_kpts, moving_desc, target_desc, matcher):
    """Calculate image keypoints that are simimlar between two images.

    Parameters
    ----------
    moving_kpts : list
        list of image keypoints for moving image
    target_kpts : list
        list of image keypoints for target image
    moving_desc : list
        list of image descriptors for moving image
    target_desc : list
        list of image descriptors for target image
    matcher : opencv object
        opencv object used to matched keypoints, such as the brute force matcher. Must have the match method available

    Returns
    -------
    filtered_moving_points : numpy array
        x, y locations of matched keypoints for moving image
    filtered_target_points : numpy array
        x, y locations of matched keypoints for target image

    """
    # match descriptors
    matches = matcher.match(moving_desc, target_desc)

    # subset only the matched points, as x, y coordinate points
    moving_match_idx = [m.queryIdx for m in matches]
    target_match_idx = [m.trainIdx for m in matches]

    moving_points = np.float32([moving_kpts[i].pt for i in moving_match_idx])
    target_points = np.float32([target_kpts[i].pt for i in target_match_idx])

    # filter the matched points using ransac mask
    mask = cv2.findHomography(moving_points, target_points, cv2.RANSAC, ransacReprojThreshold=10)[1]

    filtered_moving_points = np.float32([moving_points[i] for i in np.arange(0, len(mask)) if mask[i] == [1]])
    filtered_target_points = np.float32([target_points[i] for i in np.arange(0, len(mask)) if mask[i] == [1]])

    return filtered_moving_points, filtered_target_points
