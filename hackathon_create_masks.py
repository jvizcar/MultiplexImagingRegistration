"""
Elastix uses signal masks in non-rigid registration.

This script creates the signal mask for the breast TMA breast cancer images.

This example was used in the hackathon. The signal mask for each round is created by taking all the signal in every
channel in the round (minus the DAPI channel). Image processing is done to clean up the mask before saving the final
version of the mask.

More info: the signal from each channel image is added to the final mask by first removing any noisy signal, estimated
via wavelet Gaussian analysis. High outlier values are capped to the 95th percentile
"""
from modules import nonrigid_registration as nr
from skimage.io import imsave

# src_dir - location with images, dst_dir - location to save signal masks
src_dir = '/media/jc/NVME_SSD/SageBionetworks/normalBreast/'
dst_dir = '/media/jc/NVME_SSD/SageBionetworks/normalBreastMasks/'

# create signal mask for each round and save to file
for i in range(1, 12):
    # get filepaths to all channel images in round, except channel 1
    round_file_list = nr.get_imgs_in_round(src_dir, i)

    # create the masks for this round
    signal_mask = nr.get_tissue_mask(round_file_list)[0]

    # save the mask to file
    imsave(''.join([dst_dir, str(i).zfill(2), '_tissue_mask.png']), signal_mask)
