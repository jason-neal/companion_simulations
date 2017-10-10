

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

import simulators
from bin.analysis_iam_chi2 import decompose_database_name
from utilities.param_file import get_host_params
from utilities.phoenix_utils import closest_model_params


def _parser():
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Create Co-added Chi-squared db.')
    parser.add_argument('-s', '--stars', nargs="+", default=None,
                        help='Star names')
    parser.add_argument("-o", "--obsnum", nargs="+", default=None,
                        help="Observation number")
    parser.add_argument('--suffix', default="",
                        help='Suffix to add to the file names.')
    parser.add_argument('-v', '--verbose', action="store_true",
                        help='Enable verbose.')
    parser.add_argument('-r', '--replace', action="store_true",
                        help='Overwrite the databse if already exists.')
    parser.add_argument('-c', '--chunksize', default=1000, type=int,
                        help='Chinksize for reading in csv files.')
    return parser.parse_args()


def main(star, obs_num, suffix, replace=False, verbose=True, chunksize=1000):
    patterns = [os.path.join(
            simulators.paths["output_dir"], star,
            "{0}-{1}_{2}_iam_chisqr_results{3}*.csv".format(star, obs_num, chip, suffix))
        for chip in range(1,5)]

    if (sum(1 for _ in glob.iglob(patterns[0]))) == 0:
        patterns = [os.path.join(
                simulators.paths["output_dir"], star, "processed_csv",
                "{0}-{1}_{2}_iam_chisqr_results{3}*.csv".format(star, obs_num, chip, suffix))
            for chip in range(1,5)]

    print("new Patterns", patterns)
    if sum(sum(1 for _ in glob.iglob(pattern)) for pattern in patterns) == 0:
        raise ValueError("Issue with patterns finding for {0} obs {1}".format(star, obs_num))

    # Start up database
    coadd_database = os.path.join(
            simulators.paths["output_dir"], star,
            "{0}-{1}_coadd_iam_chisqr_results{2}.db".format(star, obs_num, suffix))

    print("Replace", replace)
    print("os.path.isfile(coadd_database)", os.path.isfile(coadd_database))
    if os.path.isfile(coadd_database):
       if replace:
           os.remove(coadd_database)
       else:
           raise IOError("The database file {0} already exists."
                " Add the switch -r to replace the old database file.".format(
                coadd_database))

    database_name = 'sqlite:///{0}'.format(coadd_database)
    engine = sa.create_engine(database_name)
    if verbose:
        print("csv_database =", engine, type(engine))

        print("pattern lengths", [sum(1 for _ in glob.iglob(pattern)) for pattern in patterns])

    # get list of patterns. and sort in order for loading in.
    detector_files = [sorted(glob.glob(pattern)) for pattern in patterns]

    i = 0
    j = 1
    for num, files  in enumerate(zip(*detector_files)):
        assert len(files) == 4
        f_0 = files[0]

        if "[" in f_0:
            n = f_0.split("[")[-1]
            n = n.split("]")[0]
            assert all(n in f for f in files)   # All have this same host
            teff, logg, feh = n.split("_")
            if verbose:
                print("host params", teff, logg, feh)
            host_flag = True
        else:
            host_flag = False

        # Initalize iterators:
        iterators =  [pd.read_csv(f, iterator=True, chunksize=chunksize) for f in files]

        while True:
            try:
                chunks = [pd_iter.get_chunk() for pd_iter in iterators]
                assert all([len(chunks[k]) == len(chunks[l]) for k, l in ((0,1), (1,2), (2,3))])
            except StopIteration:
                break

            joint_12 = pd.merge(chunks[0], chunks[1], how="outer", on=['teff_2', 'logg_2', 'feh_2', 'rv', 'gamma'], suffixes=["_1", "_2"])
            joint_34 = pd.merge(chunks[2], chunks[3],  how="outer", on=['teff_2', 'logg_2', 'feh_2', 'rv', 'gamma'], suffixes=["_3", "_4"])
            pd_joint = pd.merge(joint_12, joint_34,  how="outer", on=['teff_2', 'logg_2', 'feh_2', 'rv', 'gamma'])

            # co-adding chisquare values across detectors
            pd_joint["coadd_chi2"] = pd_joint["chi2_1"] + pd_joint["chi2_2"] + pd_joint["chi2_3"] + pd_joint["chi2_4"]

            assert not pd_joint.isnull().values.any(), "There are nans in the joint dataframe!!!"

            # Adding host parameters
            pd_joint["teff_1"] = teff
            pd_joint["logg_1"] = logg
            pd_joint["feh_1"] = feh
            pd_joint = pd_joint.rename(columns={c: c.replace(' ', '').lower() for c in pd_joint.columns})
            pd_joint.index += j

            i += 1
            pd_joint.to_sql('chi2_table', engine, if_exists='append')
            j = pd_joint.index[-1] + 1
            if verbose:
                print("indicies = ", i, j)

        if verbose:
            print("Reached end of part =", num)

    if verbose:
        print("Completed coadd db creation")


if __name__ == "__main__":
    args = vars(_parser())
    stars = args.pop('stars')
    obsnums = args.pop('obsnum')
    opts = {k: args[k] for k in args}

    assert len(stars) == len(obsnums), "Number of stars and obsnums need to be the same number."
    for star, obs in zip(stars, obsnums):
        main(star, obs, **opts)
