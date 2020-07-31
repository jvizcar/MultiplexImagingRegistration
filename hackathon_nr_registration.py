"""
Peform non-rigid registration for the hackathon. This can be done on any two images, not just the signal rounds.
"""
from os.path import join
from os import makedirs

from modules.utils import parse_tif_dir
from modules import nonrigid_registration as nr

import SimpleITK as sitk

# set variables
out_dir = '/media/jc/NVME_SSD/SageBionetworks/normalBreast'  # location to save registration logs and images
img_dir = '/media/jc/NVME_SSD/SageBionetworks/normalBreast'
target_round = 1
register_channel = 2  # channel to use for registration
transform_channel = 1  # channel you want to transform
avoid_channels = [0]  # add any channel you want to not register

# create directory to save registered images
reg_im_dir = join(out_dir, 'registered_images')
makedirs(reg_im_dir, exist_ok=True)

# get dict of image filepaths
filepath_dict = parse_tif_dir(img_dir)

# registration models - start with simple rigid registration followed by none-linear registration
reg_models = [
    'registration/elx_reg_params/rigid_largersteps.txt',
    'registration/elx_reg_params/nl.txt'
]

target_filepath = filepath_dict[target_round][register_channel]

for _round, channels_dict in filepath_dict.items():
    # don't register the target round
    if _round in [target_round] + avoid_channels:
        continue

    print(f'Registering round {_round}')

    # create directory to save registration logs for round
    out_subdir = join(out_dir, f'R{_round}_targetRound{target_round}_registerChannel{register_channel}')
    makedirs(out_subdir, exist_ok=True)

    _ = nr.register_images(filepath_dict[_round][register_channel], target_filepath, reg_models, out_subdir)

    # grab the transformations
    rig_tform = join(out_subdir, 'TransformParameters.0.txt')
    nl_tform = join(out_subdir, 'TransformParameters.1.txt')
    tform_list = [rig_tform, nl_tform]

    # transform channel image (could be different than channel used to obtain registration transforms)
    im = nr.transform_2D_image(
        filepath_dict[_round][transform_channel], tform_list, im_output_fp='', write_image=False
    )

    # save image to file
    sitk.WriteImage(im, join(reg_im_dir, f'R{_round}_targetRound{target_round}_registerChannel{register_channel}' +
                             f'_c{transform_channel}_ORG.tif'), True)

# for later purposes save the target round / channel image, without modification
target_im = sitk.ReadImage(filepath_dict[target_round][transform_channel])
sitk.WriteImage(target_im, join(reg_im_dir, f'R{target_round}_targetRound{target_round}_registerChannel' +
                                f'{register_channel}_c{transform_channel}_ORG.tif'), True)
