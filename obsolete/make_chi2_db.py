#!/usr/bin/env python
"""Script to join all the chisquared part files into a sql database."""
import argparse
import glob as glob
import logging
import os
import subprocess
import sys

import pandas as pd
import sqlalchemy as sa

logging.warning("This is obsolete you may be looking for coadd_chi2.py")


def parse_args(args):
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(
        description='Join all the chisquared part files into database.')
    parser.add_argument('pattern', help='Pattern')
    parser.add_argument('-s', '--suffix', help='Suffix to add to database name.', default=None)
    parser.add_argument('-v', '--verbose', help='Turn on Verbose.', action="store_true")
    parser.add_argument("-m", '--move', action="store_true",
                        help='Move original files after joining (default=False).')
    parser.add_argument("-r", '--remove', action="store_true",
                        help='Delete original files after joining (default=False).')
    return parser.parse_args(args)


def sql_join(pattern, suffix=None, verbose=False, move=False, remove=False):
    """Join data files with names that match a given pattern.

    Inputs
    ------
    pattern: str
        Regex to match with glob. Part before "_part" is prefix to database.
    suffix: str
        Identifier to add before .db.
    verbose: bool (default True)
        Verbose level

    Returns
    -------
    None
    Saves file to sql database

    http://pythondata.com/working-large-csv-files-python/
    """
    number_of_files = sum(1 for _ in glob.iglob(pattern))
    if verbose:
        print("Concatenating {} files.".format(number_of_files))

    if suffix is None:
        suffix = ""
    # Get first part of name
    prefix = next(glob.iglob(pattern)).split("_part")[0]
    if verbose:
        print(prefix, suffix)
    database_name = 'sqlite:///{0}{1}.db'.format(prefix, suffix)
    engine = sa.create_engine(database_name)
    if verbose:
        print("csv_database =", engine, type(engine))

    chunksize = 100000
    i = 0
    j = 1
    for f in glob.iglob(pattern):

        if "[" in f:
            n = f.split("[")[-1]
            n = n.split("]")[0]
            teff, logg, feh = n.split("_")
            print("host params", teff, logg, feh)
            host_flag = True
        else:
            host_flag = False

        for df in pd.read_csv(f, chunksize=chunksize, iterator=True):
            # print("chunk number = {}".format(i))
            if host_flag:
                df["teff_1"] = teff
                df["logg_1"] = logg
                df["feh_1"] = feh
                df = df.rename(columns={c: c.replace(' ', '').lower() for c in df.columns})
                df.index += j
                i += 1
                df.to_sql('chi2_table', engine, if_exists='append')
                j = df.index[-1] + 1
                if verbose:
                    print("indicies = ", i, j)

        if move:
            f_split = os.path.split(f)
            new_f = os.path.join(f_split[0], "processed_csv", f_split[1])
            os.makedirs(os.path.dirname(new_f), exist_ok=True)
            subprocess.call("mv {} {}".format(f, new_f), shell=True)

    if verbose:
        print("Saved results to {}.".format(database_name))

    if remove:
        print("Removing original files.")
        raise NotImplementedError

    return 0


if __name__ == '__main__':
    args = vars(parse_args(sys.argv[1:]))
    opts = {k: args[k] for k in args}

    sys.exit(sql_join(**opts))
