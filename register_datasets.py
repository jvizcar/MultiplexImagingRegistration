"""
Run registration methods used in the 2020 hackathon on the breast cancer, breast, and tonsil datasets.

For each dataset the goodness of registration is measured by using the target registration error (TRE) on matched
keypoints assessed in the DAPI channel. The TRE is reported for the unregistered images, images after registration using
linear transforms, and images after registration using SimpleITK non-rigid transforms.
"""
from mpleximreg import utils, registration, metrics
from os.path import join
from imageio import imread
import cv2
from skimage import transform
from pandas import DataFrame
from tqdm import tqdm

# variable determining datasets to run registrations on
data_dir = '/media/jc/NVME_SSD/SageBionetworks/Registration/Datasets'
dataset_dirs = ['normalBreast', 'breastCancer', 'tonsils']
reg_dataset_dirs = ['normalBreast-OHSU_registered', 'normalBreast-OHSU_registered', 'tonsils-OHSU_registered']

# static variables, might be modified if testing effects not analyzed in Hackathon
target_round = 1
feature_extractor = cv2.AKAZE_create()  # hackathon used KAZE, AKAZE is faster and with similar results
kpt_matcher = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
transformer = transform.SimilarityTransform()

# loop through each dataset
for dataset_dir, reg_dataset_dir in zip(dataset_dirs, reg_dataset_dirs):
    print(f'Registering dataset {dataset_dir}...')
    data = {'Strategy': [], 'TRE': [], 'Moving round': [], 'Target round': target_round}

    dataset_paths = utils.parse_tif_dir(join(data_dir, dataset_dir))

    # target image is channel 1 (DAPI) of target round
    target_im = utils.normalize_image(imread(dataset_paths[target_round][1]))

    # extract features from target image
    target_kpts, target_desc = feature_extractor.detectAndCompute(target_im, None)

    for r in tqdm(dataset_paths):
        if r not in [target_round, 0]:
            moving_im = utils.normalize_image(imread(dataset_paths[r][1]))

            # extract features from moving image
            moving_kpts, moving_desc = feature_extractor.detectAndCompute(moving_im, None)

            # get a list of keypoints that matched between the two using default methods
            matched_moving_pts, matched_target_pts = registration.match_keypoints(
                moving_kpts, target_kpts, moving_desc, target_desc, kpt_matcher)

            # calculate the TRE for the unregistered images
            error = metrics.target_registration_error(matched_moving_pts, matched_target_pts, moving_im.shape[:2])

            data['Strategy'].append('Unregistered')
            data['TRE'].append(error)
            data['Moving round'].append(r)

            # transform the image using the matched points
            transformer.estimate(matched_moving_pts, matched_target_pts)
            warped_img = transform.warp(moving_im.copy(), transformer.inverse, output_shape=target_im.shape[:2])
            warped_img = utils.normalize_image(warped_img)

            # using the warped image as the new moving image, matched keypoints
            moving_kpts, moving_desc = feature_extractor.detectAndCompute(warped_img, None)
            matched_moving_pts, matched_target_pts = registration.match_keypoints(
                moving_kpts, target_kpts, moving_desc, target_desc, kpt_matcher)
            reg_error = metrics.target_registration_error(matched_moving_pts, matched_target_pts, warped_img.shape[:2])

            data['Strategy'].append('Linear/Keypoint (DAPI)')
            data['TRE'].append(reg_error)
            data['Moving round'].append(r)

            # to come is the non-rigid registration, which might be done on another script so all we would do here is
            # calculating the error for already warped images

    # repeat the process with the already registered OHSU images, don't do any further registration
    print(f'Registering dataset {reg_dataset_dir}...')
    dataset_paths = utils.parse_tif_dir(join(data_dir, reg_dataset_dir))

    # target image is channel 1 (DAPI) of target round
    target_im = utils.normalize_image(imread(dataset_paths[target_round][1]))

    # extract features from target image
    target_kpts, target_desc = feature_extractor.detectAndCompute(target_im, None)

    for r in tqdm(dataset_paths):
        if r not in [target_round, 0]:
            moving_im = utils.normalize_image(imread(dataset_paths[r][1]))

            # extract features from moving image
            moving_kpts, moving_desc = feature_extractor.detectAndCompute(moving_im, None)

            # get a list of keypoints that matched between the two using default methods
            matched_moving_pts, matched_target_pts = registration.match_keypoints(
                moving_kpts, target_kpts, moving_desc, target_desc, kpt_matcher)

            # calculate the TRE for the unregistered images
            error = metrics.target_registration_error(matched_moving_pts, matched_target_pts, moving_im.shape[:2])

            data['Strategy'].append('OHSU (DAPI)')
            data['TRE'].append(error)
            data['Moving round'].append(r)

    # save the data as csv file for plotting
    df = DataFrame(data)
    df.to_csv(f'Data_Files/{dataset_dir}_hackathon_registration.csv', index=False)
