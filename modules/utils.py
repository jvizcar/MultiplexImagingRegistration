"""
Utility functions specific for this project.
"""
import re

from os.path import join
from os import listdir


# global variables
FILE_REGEX = re.compile('R(?P<round>\d{1,2})_.*_c(?P<channel>\d)_ORG.*.tif')


def parse_tif_dir(data_dir):
    """Read files in a ome tiff data directory (tif image per channel), especifically for the OHSU datasets.
    Function returns a dictionary with keys for rounds and channels parsed_dir[round#][channel#] and the values
    being the path to the file.

    Parameters
    ----------
    data_dir : str
        path to directory with the tiff images

    Return
    ------
    parsed_dir : dict
        dictionary containing the image files in nested format, first level keys are the round numbers followed
        by the channel key. Quenched rounds are ignored.

    """
    parsed_dict = {}

    for filename in listdir(data_dir):
        m = FILE_REGEX.search(filename)

        if m:
            m.groupdict()

            _round, channel = int(m['round']), int(m['channel'])

            # seed the round dict if not created
            if _round not in parsed_dict:
                parsed_dict[_round] = {}

            # add the channel to round
            parsed_dict[_round][channel] = join(data_dir, filename)

    return parsed_dict
