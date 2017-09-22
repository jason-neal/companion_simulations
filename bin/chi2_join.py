"""Script to join all the chisquared part files."""
import argparse
import glob as glob
import sys

import pandas as pd

raise Exception("This is only code, use the database one.")


def _parser():
    """Take care of all the argparse stuff.

    :returns: the args
    """
    # parser = GooeyParser(description='Wavelength Calibrate CRIRES Spectra')
    parser = argparse.ArgumentParser(description='Wavelength Calibrate CRIRES \
                                    Spectra')
    parser.add_argument('pattern', help='Pattern')
    parser.add_argument('output', help='Output name')
    parser.add_argument('-v', '--verbose', help='Turn on Verbose.', action="store_true")
    parser.add_argument('-s', '--sort', help='Sort by column.', default='chi2')
    # parser.add_argument('--direction', help='Sort direction.',
    #                     default='ascending', choices=['ascending','descending'])
    parser.add_argument("-r", '--remove', action="store_true",
                        help='Remove original files after joining (default=False).')
    return parser.parse_args()


def pandas_join(pattern, output, sort='chi2', verbose=True, remove=False):
    """Join data files with names that match a given pattern.

    Inputs
    ------
    pattern
    output: str
        Name of file to write the result to.
    verbose: bool (default True)
        Verbose level

    Returns
    -------
    None
    Saves file to

    """
    number_of_files = sum(1 for _ in glob.iglob(pattern))
    if verbose:
        print("Concatenating {} files.".format(number_of_files))

    dfs = (pd.read_csv(f) for f in glob.iglob(pattern))
    joint_df = pd.concat(dfs, ignore_index=True)

    # Sort by minimum chi2
    if sort:
        column_names = joint_df.columns.values
        if sort in column_names:
            joint_df.sort_values(sort, inplace=True)   # ascending/descending

    if verbose:
        print("Saving result to {}.".format(output))

    result = joint_df.to_csv(output, columns=joint_df.columns, index=False)
    if result is None:
        if remove:
            print("Removing original files.")
            raise NotImplementedError
        # (subprocess.call("rm ") for f in glob.iglob(pattern))

    return 0


if __name__ == '__main__':
    args = vars(_parser())
    opts = {k: args[k] for k in args}

    sys.exit(pandas_join(**opts))
