from skimage import io, exposure, filters, restoration
import numpy as np
import os
import re
import SimpleITK as sitk
from pathlib import Path
# import pickle


def get_imgs_in_round(src_dir, round_id):
    """Get filepaths for all images in a round, excluding channel 1.

    Parameters
    ----------
    src_dir : str
        directory with images
    round_id : int
        round of images to get

    Return
    ------
    f_list : list
        list of filepaths to images

    """
    f_list = [
        "".join([src_dir, f]) for f in os.listdir(src_dir)
        if f.startswith("".join(["R", str(round_id), "_"]))
        if re.search("_c1_", f) is None
    ]
    return f_list


def get_tissue_mask(f_list, sigma_thresh=1.0):
    """Combine the channel images together using varioius cleanup approaches to remove noise and peform the combination.
    The goal is to create a signal mask that includes all the signal in the channels.

    Parameters
    ----------
    f_list : list
        list of filepaths to channel images in round
    sigma_thresh: float
        values greater than this sigma threshold indicate channel image is noisy, and excluded from mask

    Return
    ------
    eq_tissue_mask
        mask of combined signal between the channels
    keep_idx: list
        index of images used to create mask

    EXAMPLE:
        src_dir = "./images/"
        f_list = ["".join([src_dir, f]) for f in os.listdir(src_dir) if not f.startswith(".")]
        get_tissue_mask, keep_idx = get_tissue_mask(f_list)

    """
    keep_idx = []
    tissue_mask = None

    for i, f in enumerate(f_list):
        # read image and rescale the range between 0 and 255
        img = io.imread(f, True)
        img = exposure.rescale_intensity(img, out_range=(0, 255))

        if tissue_mask is None:
            # seed mask with zeros
            tissue_mask = np.zeros(img.shape, dtype=np.uint8)

        # exclude noisy image using wavelet-based estimator of Gaussian noise standard deviation
        # note that the np.mean is unnecessary since estimage_sigma returns a single value for a 1 channel image
        sigma_est = np.mean(restoration.estimate_sigma(img, multichannel=False))

        # if the Gaussian standard deviation noise is higher than threshold than skip image as too noisy
        if sigma_est > sigma_thresh:
            print("skipping because too noisy", f)
            continue

        #  this image will be used to create final mask
        keep_idx.append(i)

        # cap outlier values (> 95th percentile) to 95th percentile value
        max_val = np.percentile(img, 95)  # cap extreme values
        img[img > max_val] = max_val

        # perform Otsu's thresholding - add thresholded values to the pixels for each channel
        # note that this is a strange way to perform Otsu's thresholding. They are performing it on only the pixels
        # that are greater than 0 so not including the zero background. Thresholding will thus then separate pixels
        # that are low intensity versus those of high intensity in the best Gaussian possible way
        t = filters.threshold_otsu(img[img > 0])
        tissue_mask[img > t] += img[img > t]

        # note that the tissue mask is the combined signals from each channel

    # rescale pixels that summed up beyond 255 to 255
    tissue_mask = exposure.rescale_intensity(tissue_mask, out_range=(0, 255))

    # Perform Otsu's threshold on the final mask and return as binary mask (value either 0 or 255)
    t = filters.threshold_otsu(tissue_mask[tissue_mask > 0])
    final_mask = np.zeros(tissue_mask.shape, dtype=np.uint8)
    final_mask[tissue_mask >= t] = 255

    # histogram equalization to turn the binary final mask to the same distrubtion before thresholding
    eq_tissue_mask = exposure.equalize_hist(tissue_mask, mask=final_mask)
    eq_tissue_mask[final_mask == 0] = 0

    return eq_tissue_mask, keep_idx


def register_images(source_image, source_image_res, target_image, target_image_res, reg_models, reg_output_fp):
    """ Register image with multiple models and return a list of elastix transformation maps.

    Parameters
    ----------
    source_image : str
        file path to the image that will be aligned
    source_image_res : float
        pixel resolution of the source image(e.g., 0.25 um /px)
    target_image : str
        file path to the image to which source_image will be aligned
    source_image_res : float
        pixel resolution of the target image(e.g., 0.25 um /px)
    reg_models : list
        python list of file paths to elastix paramter files
    reg_output_fp : type
        where to place elastix registration data: transforms and iteration info
    Returns
    -------
    list
        list of elastix transforms for aligning subsequent images
    """

    source = sitk.ReadImage(source_image)
    target = sitk.ReadImage(target_image)

    # set the image resolution for the images
    source.SetSpacing((source_image_res, source_image_res))
    target.SetSpacing((target_image_res, target_image_res))

    try:
        selx = sitk.SimpleElastix()
    except AttributeError:
        selx = sitk.ElastixImageFilter()

    # setup simple elastics object output directory and logs
    selx.LogToConsoleOn()
    selx.SetOutputDirectory(reg_output_fp)

    # set up the moving and target images
    selx.SetMovingImage(source)
    selx.SetFixedImage(target)

    for idx, model in enumerate(reg_models):
        if idx == 0:
            pmap = sitk.ReadParameterFile(model)
            pmap["WriteResultImage"] = ("false", )
            selx.SetParameterMap(pmap)
        else:
            pmap = sitk.ReadParameterFile(model)
            pmap["WriteResultImage"] = ("false", )
            selx.AddParameterMap(pmap)

    selx.LogToFileOn()

    # execute registration:
    selx.Execute()

    return list(selx.GetTransformParameterMap())


def transform_2D_image(source_image,
                       source_image_res,
                       transformation_maps,
                       im_output_fp,
                       write_image=False):
    """Transform 2D images with multiple models and return the transformed image
        or write the transformed image to disk as a .tif file.
    Parameters
    ----------
    source_image : str
        file path to the image that will be transformed
    source_image_res : float
        pixel resolution of the source image(e.g., 0.25 um /px)
    transformation_maps : list
        python list of file paths to elastix parameter files
    im_output_fp : str
        output file path
    write_image : bool
        whether to write image or return it as python object
    Returns
    -------
    type
        Description of returned object.
    """

    try:
        transformix = sitk.SimpleTransformix()
    except AttributeError:
        transformix = sitk.TransformixImageFilter()

    if isinstance(source_image, sitk.Image):
        image = source_image
    else:
        print("reading image from file")
        image = sitk.ReadImage(source_image)
        image.SetSpacing((source_image_res, source_image_res))

    for idx, tmap in enumerate(transformation_maps):

        if idx == 0:
            if isinstance(tmap, sitk.ParameterMap) is False:
                tmap = sitk.ReadParameterFile(tmap)
            tmap["InitialTransformParametersFileName"] = (
                "NoInitialTransform", )
            transformix.SetTransformParameterMap(tmap)
            tmap["ResampleInterpolator"] = (
                "FinalNearestNeighborInterpolator", )
        else:
            if isinstance(tmap, sitk.ParameterMap) is False:
                tmap = sitk.ReadParameterFile(tmap)
            tmap["InitialTransformParametersFileName"] = (
                "NoInitialTransform", )
            tmap["ResampleInterpolator"] = (
                "FinalNearestNeighborInterpolator", )

            transformix.AddTransformParameterMap(tmap)

    #take care for RGB images
    pixelID = image.GetPixelID()

    transformix.LogToConsoleOn()
    transformix.LogToFileOn()
    transformix.SetOutputDirectory(str(Path(im_output_fp).parent))

    if pixelID in list(range(1, 13)) and image.GetDepth() == 0:
        transformix.SetMovingImage(image)
        image = transformix.Execute()
        image = sitk.Cast(image, pixelID)

    elif pixelID in list(range(1, 13)) and image.GetDepth() > 0:
        images = []
        for chan in range(image.GetDepth()):
            transformix.SetMovingImage(image[:, :, chan])
            images.append(sitk.Cast(transformix.Execute(), pixelID))
        image = sitk.JoinSeries(images)
        image = sitk.Cast(image, pixelID)

    elif pixelID > 12:
        images = []
        for idx in range(image.GetNumberOfComponentsPerPixel()):
            im = sitk.VectorIndexSelectionCast(image, idx)
            pixelID_nonvec = im.GetPixelID()
            transformix.SetMovingImage(im)
            images.append(sitk.Cast(transformix.Execute(), pixelID_nonvec))
            del im

        image = sitk.Compose(images)
        image = sitk.Cast(image, pixelID)

    if write_image is True:
        sitk.WriteImage(image, im_output_fp + "_registered.tif", True)
        return
    else:
        return image
