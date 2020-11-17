# Multiplex Imaging Registration
Updated 5 Nov. 2020

### Summary

---
**Data_Files/**

Results from registration (errors) are saved in csv files that can be used for plotting. These csv files have columns
for the method used in registration, the round used as the target and moving round, channel used to register, dataset 
used, TRE, and the error distance in micro meters.
* *distance-results-tbl.tsv*: results from hackathon registration challenge ([link](https://github.com/IAWG-CSBC-PSON/registration-challenge/blob/master/distance-results-tbl.tsv))
* *hackathon_challenge_results.csv*: formatted distance-results-tbl.tsv file for plotting

---
### mpleximreg

Python module containing functions for running various registrations on the data in the hackathon.

---
### notebooks

Most of the relevant examples are provided in Jupyter notebooks.

**Hackathon_recreation.ipynb**
* Code used to recreate the results of the hackathon by running rigid registration and non-rigid registration using 
channel 2 on all three datasets and plotting (as well saving figures)
    * Datasets - BRCA, Breast tissue, and Tonsils from 

**Registration_with_background.ipynb**
* shows an example of running non-rigid registration on image tif dirs
* allows setting target round and channel to use in registration
* shows an example of calculating the TRE on an image tif dir - also returns the error distance in micro-meters
* provides some visualization of the results

**Extra.ipynb**
* extra code snippets used in this project, kept as potential future reference
    * there is an example of how to create a csv file for the tif data directories so the functions can be run
---
### Docker image

A [Docker image](https://hub.docker.com/repository/docker/jvizcar/sage_mp_registration) has been created that can run 
the code inside this repo. Below is a following example on how to run this:

```angular2
$ docker run --rm -it -p<localport>:8888 -v <root_repo>:/code -v <data_dir>:/data jvizcar/sage_mp_registration:latest
```

By default this Docker container starts a jupyter service upon run. The terminal will display the token (password) for
running accessing the Jupyter notebook on the port used (see localport in the command). 

The examples (Jupyter notebooks) are run on tif dirs for multiplex data which can be obtained from Synapse at this
location: [https://www.synapse.org/#!Synapse:syn21615167](https://www.synapse.org/#!Synapse:syn21615167). 

**Mounting**: you should mount the root of this repo to /code in the Docker container and your directory with your tif
image dirs to /data in the Docker container as shown above. This will allow seamless running of the code examples with
minimal modification of the directory paths.
