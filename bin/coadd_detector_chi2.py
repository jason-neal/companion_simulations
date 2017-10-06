

"""Co-add_chi2_values.py.

Create Table of minimum Chi_2 values and save to a table.
"""
import argparse
import glob
import os

import corner
import matplotlib.pyplot as plt
import pandas as pd
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

    #sql_join and add to databaseand
    patterns = [os.path.join(
            simulators.paths["output_dir"], star,
            "{0}-{1}_{2}_iam_chisqr_results{3}*.csv".format(star, obs_num, chip, suffix))
        for chip in range(1,5)]
    # print("first Patterns", patterns)
    # print((sum(1 for _ in glob.iglob(pattern)) for pattern in patterns))
    if (sum(1 for _ in glob.iglob(patterns[0]))) == 0:
        patterns = [os.path.join(
                simulators.paths["output_dir"], star, "processed_csv",
                "{0}-{1}_{2}_iam_chisqr_results{3}*.csv".format(star, obs_num, chip, suffix))
            for chip in range(1,5)]

    print("new Patterns", patterns)
    if (sum(1 for _ in glob.iglob(pattern)) for pattern in patterns) == 0:
        print("pattern lengths", [sum(1 for _ in glob.iglob(pattern)) for pattern in patterns])
        raise ValueError("Issue with patterns finding files")

    #load all 4 detectors cvs at same timeself.
    #print("end Patterns", patterns)
    # Try from csv files

    print("pattern lengths", [sum(1 for _ in glob.iglob(pattern)) for pattern in patterns])


    # get list of patterns. and sort in order for loading in.
    detector_files = [sorted(glob.glob(pattern)) for pattern in patterns]
    print(detector_files, len(detector_files))


    for i, files  in enumerate(zip(*detector_files)):
        print(i)
        print(files)

        pd_iter1 = pd.read_csv(files[0], iterator=True, chunksize=10)
        pd_iter2 = pd.read_csv(files[1], iterator=True, chunksize=10)
        pd_iter3 = pd.read_csv(files[2], iterator=True, chunksize=10)
        pd_iter4 = pd.read_csv(files[3], iterator=True, chunksize=10)

        print(pd_iter4)

        df_1 = pd_iter1.get_chunk()
        df_2 = pd_iter2.get_chunk()
        df_3 = pd_iter3.get_chunk()
        df_4 = pd_iter4.get_chunk()

        print("df_2 head", df_1.head())
        print("df_2 head", df_2.head())
        print("df_3 head", df_3.head())
        print("df_4 head", df_4.head())

if __name__ == "__main__":
    args = _parser()
    stars = args.stars
    obsnums = args.obsnum
  #  if stars is None:
   #     stars = ["HD30501", "HD211847", "HD4747"]

    assert len(stars) == len(obsnums), "Number of stars and obsnums need to be the same number."
    for star, obs in zip(stars, obsnums):
        main(star, obs, args.suffix)
