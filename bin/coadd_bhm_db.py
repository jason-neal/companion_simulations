#!/usr/bin/env python
"""Co-add_chi2_values.py.

Create Table of minimum Chi_2 values and save to a table.
"""
import argparse
import glob
import os
import subprocess

import pandas as pd
import sqlalchemy as sa

import simulators


def _parser():
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Create Co-added Chi-squared db.')
    parser.add_argument('star', help='Star name')
    parser.add_argument("obsnum", help="Observation number")
    parser.add_argument('--suffix', default="",
                        help='Suffix to add to the file names.')
    parser.add_argument('-v', '--verbose', action="store_true",
                        help='Enable verbose.')
    parser.add_argument('-r', '--replace', action="store_true",
                        help='Overwrite the database if already exists.')
    parser.add_argument('-c', '--chunksize', default=1000, type=int,
                        help='Chinksize for reading in csv files.')
    parser.add_argument("-m", '--move', action="store_true",
                        help='Move original files after joining (default=False).')

    return parser.parse_args()


def main(star, obsnum, suffix, replace=False, verbose=True, chunksize=1000, move=False):
    """"""
    star = star.upper()
    if suffix is None:
        suffix = ""

    patterns = [os.path.join(
        simulators.paths["output_dir"], star, "bhm",
        "{0}-{1}_{2}_bhm_chisqr_results{3}*.csv".format(star, obsnum, chip, suffix))
        for chip in range(1, 5)]

    if (sum(1 for _ in glob.iglob(patterns[0]))) == 0:
        patterns = [os.path.join(
            simulators.paths["output_dir"], star, "bhm", "processed_csv",
            "{0}-{1}_{2}_bhm_chisqr_results{3}*.csv".format(star, obsnum, chip, suffix))
            for chip in range(1, 5)]

    print("new Patterns", patterns)
    if sum(sum(1 for _ in glob.iglob(pattern)) for pattern in patterns) == 0:
        raise ValueError("Issue with patterns finding for {0} obs {1}".format(star, obsnum))

    # Start up database
    coadd_database = os.path.join(
        simulators.paths["output_dir"], star, "bhm",
        "{0}-{1}_coadd_bhm_chisqr_results{2}.db".format(star, obsnum, suffix))

    print("Replace", replace)
    print("os.path.isfile(coadd_database)", os.path.isfile(coadd_database))
    if os.path.isfile(coadd_database):
        if replace:
            os.remove(coadd_database)
        else:
            raise IOError("The database file {0} already exists. Add the switch"
                          " -r to replace the old database file.".format(coadd_database))

    database_name = 'sqlite:///{0}'.format(coadd_database)
    engine = sa.create_engine(database_name)
    if verbose:
        print("csv_database =", engine, type(engine))

        print("pattern lengths", [sum(1 for _ in glob.iglob(pattern)) for pattern in patterns])

    # get list of patterns. and sort in order for loading in.
    detector_files = [sorted(glob.glob(pattern)) for pattern in patterns]

    i, j = 0, 1
    for num, files in enumerate(zip(*detector_files)):
        assert len(files) == 4
        f_0 = files[0]

        # Initalize iterators:
        iterators = [pd.read_csv(f, iterator=True, chunksize=chunksize) for f in files]

        while True:
            try:
                chunks = [pd_iter.get_chunk() for pd_iter in iterators]
                assert all([len(chunks[k]) == len(chunks[l])
                            for k, l in ((0, 1), (1, 2), (2, 3))])
            except StopIteration:
                break

            joint_12 = pd.merge(chunks[0], chunks[1], how="outer", suffixes=["_1", "_2"],
                                on=['teff_1', 'logg_1', 'feh_1', 'gamma'])
            joint_34 = pd.merge(chunks[2], chunks[3], how="outer", suffixes=["_3", "_4"],
                                on=['teff_1', 'logg_1', 'feh_1', 'gamma'])
            pd_joint = pd.merge(joint_12, joint_34, how="outer",
                                on=['teff_1', 'logg_1', 'feh_1', 'gamma'])

            # co-adding chisquare values across detectors
            pd_joint["coadd_chi2"] = pd_joint["chi2_1"] + pd_joint["chi2_2"] + pd_joint["chi2_3"] + pd_joint["chi2_4"]
            pd_joint["coadd_npix"] = pd_joint["npix_1"] + pd_joint["npix_2"] + pd_joint["npix_3"] + pd_joint["npix_4"]

            assert not pd_joint.isnull().values.any(), "There are nans in the joint DataFrame!!!"

            # Adding host parameters
            pd_joint = pd_joint.rename(columns={c: c.replace(' ', '').lower() for c in pd_joint.columns})
            pd_joint.index += j

            i += 1
            pd_joint.to_sql('chi2_table', engine, if_exists='append')
            j = pd_joint.index[-1] + 1
            if verbose:
                print("Indicies = ", i, j)

        if move:
            for f in files:
                f_split = os.path.split(f)  # ["head", "tail"]
                new_f = os.path.join(f_split[0], "processed_csv", f_split[1])
                os.makedirs(os.path.dirname(new_f), exist_ok=True)
                subprocess.call("mv {} {}".format(f, new_f), shell=True)

        if verbose:
            print("Reached end of part =", num)

    if verbose:
        print("Completed bhm coadd db creation")


if __name__ == "__main__":
    args = vars(_parser())
    opts = {k: args[k] for k in args}
    main(**opts)

    print("\nNow use coadd_bhm_analysis.py next")
