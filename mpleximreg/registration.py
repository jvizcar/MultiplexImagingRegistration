"""
Functions for performing different registration approaches and calculating performance metrics.

List of functions:
* register_images
* tre_distance
* match_keypoints
* apply_transform
* register_images_adv
"""
import numpy as np
import cv2


def draw_kpts(im, keyp, col, th):
    for curKey in keyp:
        x = np.int(curKey.pt[0])
        y = np.int(curKey.pt[1])
        size = np.int(curKey.size)
        cv2.circle(im, (x, y), size, col, thickness=th, lineType=8, shift=0)
    return im


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


# def register_images_adv(moving_im, target_im, feature_detector, matcher, visuals=False, savepath=None):
#     """Register a moving image to a target image using OpenCV based image features and keypoint matching strategies.
#     The registration done with matched keypoints is rigid, applying both translation and rotation but no warping
#     transformations.
#
#     Paramters
#     ---------
#     moving_im : array-like
#         the image to transform when registering to target image
#     target_im : array-like
#         target image
#     feature_detector : cv2 object (feature detector)
#         the feature detector to use, must have the detectAndCompute method
#     matcher : opencv object
#         a feature matcher from opencv
#     visuals : bool (default: False)
#         if True then visuals will be displayed for each step
#     savepath : str (default: None)
#         file dir + filename to save image to
#
#     Return
#     ------
#     results : dict
#         information about the workflow
#
#     """
#     results = {}
#
#     # detect keypoints for both images
#     moving_kpts, moving_desc = feature_detector.detectAndCompute(moving_im, None)
#     target_kpts, target_desc = feature_detector.detectAndCompute(target_im, None)
#
#     # add number of keypoints detected in each image
#     results['moving-n_kpts'] = len(moving_kpts)
#     results['target-n_kpts'] = len(target_kpts)
#
#     # match keypoints between moving and target keypoints
#     matches = matcher.match(moving_desc, target_desc)
#
#     # add the number of keypoints that matched
#     results['n_matched_kpts'] = len(matches)
#
#     # subset to only the matched keypoints for each image
#     moving_matched_kpts = [moving_kpts[match.queryIdx] for match in matches]
#     target_matched_kpts = [target_kpts[match.trainIdx] for match in matches]
#
#     # convert the keypoints to a numpy array
#     moving_matched_pts = np.float32([kpt.pt for kpt in moving_matched_kpts])  # (x,y) coords
#     target_matched_pts = np.float32([kpt.pt for kpt in target_matched_kpts])  # (x,y) coords
#
#     # find homography matrix and mask to filter bad points with Ransac
#     mask = cv2.findHomography(moving_matched_pts, target_matched_pts, cv2.RANSAC, ransacReprojThreshold=10)[1]
#
#     # use the mask to keep only "good" keypoints
#     moving_filtered_kpts = [moving_matched_kpts[i] for i in np.arange(0, len(mask)) if mask[i] == [1]]
#     target_filtered_kpts = [target_matched_kpts[i] for i in np.arange(0, len(mask)) if mask[i] == [1]]
#
#     # convert the filtered keypoints to arrays
#     moving_filtered_pts = np.float32([kpt.pt for kpt in moving_filtered_kpts])
#     target_filtered_pts = np.float32([kpt.pt for kpt in target_filtered_kpts])
#
#     # add the number of filtered points
#     results['n_filtered_kpts'] = len(moving_matched_kpts)
#
#     # use the filtered pts to transform the moving image
#     transformed_im, transformed_pts = apply_transform(
#         moving_im, target_im, moving_filtered_pts, target_filtered_pts, LINEAR_TRANSFORMER
#     )
#
#     # calculate the error between the matched keypoints
#     h, w = moving_im.shape[:2]
#     results['error (raw)'] = tre_distance(moving_filtered_pts, target_filtered_pts, h, w)
#     results['error (registered)'] = tre_distance(transformed_pts, target_filtered_pts, h, w)
#
#     if visuals or savepath is not None:
#         # create the grid
#         fig = plt.figure(figsize=(20, 15))
#         gs = GridSpec(2, 3, figure=fig)
#         ax1 = fig.add_subplot(gs[0, 0])
#         ax2 = fig.add_subplot(gs[0, 1])
#         ax3 = fig.add_subplot(gs[1, 0])
#         ax4 = fig.add_subplot(gs[1, 1])
#         ax5 = fig.add_subplot(gs[1, 2])
#
#         # draw the unfiltered keypoints as red and filtered keypoitns over it as green
#         moving_im_rgb = np.stack((moving_im,) * 3, axis=-1)
#         target_im_rgb = np.stack((target_im,) * 3, axis=-1)
#
#         moving_im_kpts = drawKeyPts(moving_im_rgb, moving_kpts, (255, 0, 0), 15)
#         target_im_kpts = drawKeyPts(target_im_rgb, target_kpts, (255, 0, 0), 15)
#         moving_im_kpts = drawKeyPts(moving_im_kpts, moving_filtered_kpts, (0, 255, 0), 15)
#         target_im_kpts = drawKeyPts(target_im_kpts, target_filtered_kpts, (0, 255, 0), 15)
#
#         ax1.imshow(moving_im_kpts)
#         ax1.set_title('Moving Image with Keypoints')
#         ax2.imshow(target_im_kpts)
#         ax2.set_title('Target Image with Keypoints')
#
#         # plot the raw images and the warped image in the center
#         ax3.imshow(moving_im, cmap='gray')
#         ax3.set_title('Moving Image (Raw)')
#         ax4.imshow(transformed_im, cmap='gray')
#         ax4.set_title('Moving Image (Registered)')
#         ax5.imshow(target_im, cmap='gray')
#         ax5.set_title('Target Image (Raw)')
#
#         # adjust plots
#         ax1.set_aspect('equal')
#         ax1.set_xticklabels([])
#         ax1.set_yticklabels([])
#         ax2.set_aspect('equal')
#         ax2.set_xticklabels([])
#         ax2.set_yticklabels([])
#         ax3.set_aspect('equal')
#         ax3.set_xticklabels([])
#         ax3.set_yticklabels([])
#         ax4.set_aspect('equal')
#         ax4.set_xticklabels([])
#         ax4.set_yticklabels([])
#         ax5.set_aspect('equal')
#         ax5.set_xticklabels([])
#         ax5.set_yticklabels([])
#         fig.subplots_adjust(wspace=0.1)
#
#         plt.suptitle('Raw Error: %.5f, Registered Error: %.5f' % (results['error (raw)'],
#                                                                   results['error (registered)']))
#
#         if savepath is not None:
#             plt.savefig(savepath, bbox_inches='tight')
#
#         if visuals:
#             plt.show()
#
#         # make sure to close figure
#         plt.close(fig)
#
#     return results


# def rigid_registration(moving_im, target_im, target_kpts=None, target_desc=None):
#     """Register images to each other using various methods. For each method an error distance (TRE) is calculated
#     between the images.
#
#     Parameters
#     ----------
#     moving_im : numpy array
#         image that will be registered to match target image
#     target_im : numpy array
#         target image that moving image will be registered to
#     target_kpts : list
#         keypoints for target image, if None then they will be extracted by the default methods
#     target_desc : list
#         descriptors to go along with target_kpts parameters
#
#     Return
#     ------
#     images : list
#         list of images after registration (or original moving image if 'none' or default None passed). Last index is
#         always the target image.
#     kpts : list
#         list of keypoints used on calculating the TRE distance for each method
#     distances : list
#         the TRE error between moving and target image, also referred to as distances, estimated using Kaze feature
#         brute force matching
#
#     """
#     # check need to extract features from target image
#     if target_desc is None and target_kpts is None:
#         target_kpts, target_desc = feature_detector.detectAndCompute(moving, None)
#
#     # default method is no registration
#     if methods is None:
#         methods = ['none']
#
#     # matching keypoints should always be done
#     try:
#         moving_kpts, target_kpts = match_keypoints(moving_im, target_im, FD)
#         h, w = moving_im.shape
#     except:
#         raise Exception('keypoint matching failed')
#
#     for method in methods:
#         if method == 'none':
#             images.append(moving_im.copy())
#             kpts.append(moving_kpts)
#             distances.append(tre_distance(moving_kpts, target_kpts, h, w))
#         elif method == 'linear':
#             warped_im, warped_kpts = apply_transform(
#               moving_im, target_im, moving_kpts, target_kpts, LINEAR_TRANSFORMER)
#             images.append(warped_im)
#             kpts.append(warped_kpts)
#             distances.append(tre_distance(warped_kpts, target_kpts, h, w))
#         else:
#             raise Exception(f'method {method} not supported')
#
#     return images, kpts, distances


# def apply_transform(moving, target, moving_pts, target_pts, transformer, output_shape_rc=None):
#     """Apply a transform to an image given moving and target image keypoints and transformer parameter form the
#     scikit-image package.
#
#     Parameters
#     ----------
#     moving : numpy array
#         image to transform
#     target : numpy array
#         image that moving image will transform into
#     moving_pts : numpy array
#         x, y locations of matched keypoints for moving image
#     target_pts : numpy array
#         x, y locations of matched keypoints for target image
#     transformer : skimage tranform object
#         See https://scikit-image.org/docs/dev/api/skimage.transform.html for different transformations
#     output_shape_rc : tuple
#         (shape of warped image (row, col). If None, use shape of target image
#
#     Returns
#     -------
#     warped_im : numpy array
#         the moving image warped to match the target image
#     warped_pts : numpy array
#         the x, y location of the warped keypoints
#
#     """
#     if output_shape_rc is None:
#         output_shape_rc = target.shape[:2]
#
#     if str(transformer.__class__) == "<class 'skimage.transform._geometric.PolynomialTransform'>":
#         transformer.estimate(target_pts, moving_pts)
#         warped_img = transform.warp(moving, transformer, output_shape=output_shape_rc)
#
#         # restimate to warp points
#         transformer.estimate(moving_pts, target_pts)
#         warped_pts = transformer(moving_pts)
#     else:
#         transformer.estimate(moving_pts, target_pts)
#         warped_img = transform.warp(moving, transformer.inverse, output_shape=output_shape_rc)
#         warped_pts = transformer(moving_pts)
#
#     return warped_img, warped_pts
