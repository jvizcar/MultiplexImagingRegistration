# MultiplexImagingRegistration
Updated on 2020-07-06

A Docker image has been created that can run the code inside this repo. Below is a following example on how to run this:

```angular2
$ docker run --rm -it -p<localport>:8888 -v <localDirToMount>:/mnt/<folderNameToMountTo> jvizcar/sage_mp_registration:latest
```

Make sure to mount any directories you wish to use, mount them to /mnt/<chooseFolderName> and select the port you wish
to expose for running a jupyter notebook. By default this Docker container starts a jupyter service upon run. Password
for the jupyter notebook is "w00fw00f".

Modules contains Python modules that can be imported with useful functions used in image registration of multiplex 
imaging. The code here was developed using TMA images for a single scan, where each round's channel images were saved
locally as tif files. To test this on the source images, such as ome.tiff or .czi those mutliplex images should first be
saved into a directory as tif files per channel in round. For code to work make sure the filenames contain this format:
R#_....._c#_ORG.tif where R# is the round number and c# is the channel number.

## Hackathon Scripts

The 2020 hackathon performed rigid registration on keypoints and also did non-rigid registration using intensity signals
using the SimpleITK Python library. The registration strategies can be recreated by running the register_datasets.py 
script which will save the results to three csv files in the Data_Files dir (the three csv files correspond to the three
datasets available: breast TMAs, breast cancer TMAs, and tonsil TMAs).

A jupyter notebook (hackathon_results.ipynb) is provided for visualizing the results by plotting the data in the csv 
files.

## Non-rigid registration
### Signal masks
Non-rigid registration used in the hackathon, and code found here, uses signal masks to get the transform between two 
rounds. The signal mask is obtained by combining the channel images of a round together and peforming some processing 
steps to avoid including noisy signal. The signal masks between two rounds is then used by SimpleElastix to find the 
non-rigid transform. Note that the souce images themselves are not used but rather the mask for the round. This is 
important to keep in mind.

Example:
```angular2
from modules import nonrigid_registration as nr

...

round_file_list = nr.get_imgs_in_round(src_dir, i)  # i the round
signal_mask = nr.get_tissue_mask(round_file_list)[0]
```

The function get_imgs_in_round returns a list of filepaths to the channel images for the specified round (i).

The function get_tissue_mask uses the channel images specified in round_file_list to create the signal mask. This two
step code snippet can be used to create a signal mask for a single round given any number of channels (not sure if this
will work for only one channel).

Note that channels are excluded if too noisy, look at source code for details, so the code could fail to generate a 
signal mask. In this case you could reduce the noisy threshold or include other channels.

### Elastix NR Registration

Registration is done using the SimpleElastix module, part of the SimpleITK Python library which wraps Elastix for 
Python. 

Registering two images is simple: 

```angular2
from modules import nonrigid_registration as nr
from os.path import join
...
reg_models = [
    'registration/elx_reg_params/rigid_largersteps.txt',
    'registration/elx_reg_params/nl.txt'
]

# transformation files are saved to file in out_subdir
_ = nr.register_images(moving_filepath, target_filepath, reg_models, out_subdir)

# create list of files with transformations
rig_tform = join(out_subdir, 'TransformParameters.0.txt')
nl_tform = join(out_subdir, 'TransformParameters.1.txt')
tform_list = [rig_tform, nl_tform]

# transform channel image (could be different than channel used to obtain registration transforms)
im = nr.transform_2D_image(
    filepath_to_transform, tform_list, im_output_fp='', write_image=False
)
```

Custom code must be written depending on your need. Look at hackathon_nr_registration.py for an example.

If you have a directory with your channel images in format R#_...._c#_ORG.tif then you can modify the variables in 
nr_registration.py to register a specific channel in every round.

### Coming soon
Ability to use multiple channels to do the registration. This can be done by creating a signal mask from multiple
channels (done in the hackathon but maybe not used) and using the signal mask in the non-rigid registration.

More complex methods of registration and easier ability to select forms of registrations to use. Also - ability to do
registration without the need to create intermediate files.




