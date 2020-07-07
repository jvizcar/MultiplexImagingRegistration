"""
Code provided Aidan Daly that was used in the hackathon to do the non-rigid registration with SimpleElastix. This code
does not show how the signal masks were created, the rest of the code is similar (if not the same) as implemented by me
in this repo.
"""
import os
import glob
import SimpleITK as sitk
import cv2

from skimage import img_as_float
from skimage.util import img_as_ubyte

from registration import Reg2D as r2d
from keypoint_registration import match_keypoints, keypoint_distance


mask_dir = "../normal_breast_af_masks/"
out_dir = os.path.join(mask_dir, "elastix_output")
if not os.path.isdir(out_dir):
	os.mkdir(out_dir)

rounds = ["R%d" % i for i in range(1, 11)]
imfiles = [glob.glob(os.path.join(mask_dir, "%s.png" % r))[0] for r in rounds]

# Calculate transforms
for i, im in enumerate(imfiles):

	if i==0:
		target = im
	else:
		moving = im

		reg_models = [
		    'registration/elx_reg_params/rigid_largersteps.txt',
		    'registration/elx_reg_params/nl.txt'
		]

		base = rounds[i]
		out_folder = os.path.join(out_dir, base)

		if not os.path.isdir(out_folder):
			os.mkdir(out_folder)
		r2d.register_2D_images(moving, 0.2645833333, target, 0.2645833333,
			reg_models, out_folder)

# Apply transforms
dapi_dir = "../NormalBreast/"
imfiles = [glob.glob(os.path.join(dapi_dir, "%s_*_c1_*.tif" % r))[0] for r in rounds]

for i, im in enumerate(imfiles):

	# Don't need to align R1
	if i==0:
		dapi_target = sitk.GetArrayFromImage(sitk.ReadImage(im))
		dapi_target = img_as_ubyte(img_as_float(dapi_target))
	else:
		base = rounds[i]
		out_folder = os.path.join(out_dir, base)

		rig_xform = glob.glob(os.path.join(out_folder, '*.0.txt'))[0]
		nl_xform = glob.glob(os.path.join(out_folder, '*.1.txt'))[0]
		xforms = [rig_xform, nl_xform]

		warped = r2d.transform_2D_image(im, 0.2645833333, xforms,
			im_output_fp='', write_image=False)
		dapi_moving = sitk.GetArrayFromImage(warped)
		dapi_moving = img_as_ubyte(img_as_float(dapi_moving))

		fd = cv2.KAZE_create(extended=True)
		moving_pts, target_pts = match_keypoints(dapi_moving, dapi_target, feature_detector=fd)
		dst = keypoint_distance(moving_pts, target_pts, dapi_target.shape[0], dapi_target.shape[1])
		print('%d\t%d\t%f' % (i+1, 1, dst))

