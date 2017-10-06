

"""Co-add_chi2_values.py.

Create Table of minimum Chi_2 values and save to a table.
"""
import argparse
import os

import corner
import matplotlib.pyplot as plt
#import pandas as pd
import sqlalchemy as sa
#from joblib import Parallel, delayed
#from pandas.plotting import scatter_matrix

import simulators
from bin.analysis_iam_chi2 import decompose_database_name
from utilities.param_file import get_host_params
from utilities.phoenix_utils import closest_model_params
#from utilities.scatter_corner import scatter_corner


def _parser():
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Co-added Chi-squared db.')
    parser.add_argument('-s', '--stars', help='Star names', nargs="+", default=None)
    parser.add_argument("-o", "--obsnum", help="Observation number", nargs="+", default=None)
    parser.add_argument('--suffix', help='Suffix to add to the file names.', default="")
    return parser.parse_args()


def main(star, obs_num, suffix):
    databases = ["", "", "", ""]
    for chip in range(1, 5):
        databases[chip-1] = os.path.join(simulators.paths["output_dir"], star,
           "{0}-{1}_{2}_iam_chisqr_results{3}.db".format(star, obs_num, chip, suffix))

    print("databases", databases)



if __name__ == "__main__":
    args = _parser()
    stars = args.stars
    obsnums = args.obsnum
  #  if stars is None:
   #     stars = ["HD30501", "HD211847", "HD4747"]

    assert len(stars) == len(obsnums), "Number of stars and obsnums need to be the same number."
    for star, obs in zip(stars, obsnums):
        main(star, obs, args.suffix)
