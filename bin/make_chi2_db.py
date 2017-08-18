"""Script to join all the chisquared part files into a sql database."""
import argparse
import glob as glob
import sys

import pandas as pd
from sqlalchemy import create_engine

def _parser():
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Wavelength Calibrate CRIRES \
                                    Spectra')
    parser.add_argument('pattern', help='Pattern')
    parser.add_argument('-s', '--suffix', help='Suffix to add to database name.')
    parser.add_argument('-v', '--verbose', help='Turn on Verbose.', action="store_true")
    # parser.add_argument('-s', '--sort', help='Sort by column.', default='chi2')
    # parser.add_argument('--direction', help='Sort direction.',
    #                     default='ascending', choices=['ascending','decending'])
    parser.add_argument("-r", '--remove', action="store_true",
                        help='Remove original files after joining (default=False).')
    return parser.parse_args()


def sql_join(pattern, suffix="", verbose=True, remove=False):
    """Join data files with names that match a given pattern.

    Inputs
    ------
    pattern: str
        Regex to match with glob. Part before "_part" is prefix to databse.
    suffix: str
        Identifier to add before .db.
    verbose: bool (default True)
        Verbose level

    Returns
    -------
    None
    Saves file to sql database

    """
    number_of_files = sum(1 for _ in glob.iglob(pattern))
    if verbose:
        print("Concatinating {} files.".format(number_of_files))

    # Get frist part of name
    prefix = next(glob.iglob(pattern)).split("_part")[0]
    print(prefix)
    csv_database = create_engine('sqlite:///{}.db'.format(prefix, suffix))
    print(type(csv_database))
    print((csv_database))
    for f in glob.iglob(pattern):
        df = pd.read_csv(f)

        if "[" in f:
            n = f.split("[")[-1]
            n = n.split("]")[0]
            teff, logg, feh = n.split("_")
            print("host params", teff, logg, feh)
            df["teff_1"] = teff
            df["logg_1"] = logg
            df["feh_1"] = feh
        df = df.rename(columns={c: c.replace(' ', '') for c in df.columns})
        df.to_sql('table', csv_database, if_exists='append')

    if verbose:
            print("Saved results to {}.".format(csv_database))

    if remove:
        print("Removing original files.")
        raise NotImplementedError
        # (subprocess.call("rm ") for f in glob.iglob(pattern))

    return 0


if __name__ == '__main__':
    args = vars(_parser())
    opts = {k: args[k] for k in args}

    sys.exit(sql_join(**opts))
