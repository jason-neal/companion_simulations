import fnmatch
import os
from typing import List

import pandas as pd


def save_pd_cvs(name, data):
    # Take dict of data to save to csv called name
    df = pd.DataFrame(data=data)
    df.to_csv(name + ".csv", sep=',', index=False)
    return 0


def get_filenames(path, regexp, regexp2=None):
    # type: (str, str, str) -> List[str]
    """Regexp must be a regular expression as a string.

    eg '*.ms.*', '*_2.*', '*.ms.norm.fits*'

    regexp2 is if want to match two expressions such as
    '*_1*' and '*.ms.fits*'
    """
    filelist = []
    for file in os.listdir(path):
        if regexp2 is not None:  # Match two regular expressions
            if fnmatch.fnmatch(file, regexp) and fnmatch.fnmatch(file, regexp2):
                filelist.append(file)
        else:
            if fnmatch.fnmatch(file, regexp):
                filelist.append(file)
    filelist.sort()
    return filelist
