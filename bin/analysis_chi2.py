"""tcm_chi_squared_analysis."""

import argparse
import sys

import matplotlib.pyplot as plt
import pandas as pd


def _parser():
    """Take care of all the argparse stuff.

    :returns: the args
    """
    # parser = GooeyParser(description='Wavelength Calibrate CRIRES Spectra')
    parser = argparse.ArgumentParser(description='Wavelength Calibrate CRIRES \
                                    Spectra')
    parser.add_argument('filename', help='Filename of chi2 data.')
    # parser.add_argument('output', help='Output name')
    # parser.add_argument('-v', '--verbose', help='Turn on Verbose.', action="store_true")
    # parser.add_argument('-s', '--sort', help='Sort by column.', default='chi2')
    # parser.add_argument('--direction', help='Sort direction.',
    #                     default='ascending', choices=['ascending','decending'])
    # parser.add_argument("-r", '--remove', action="store_true",
    #                    help='Remove original files after joining (default=False).')
    return parser.parse_args()


def quicklook(df, x, y):
    plt.plot(df[x], df[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title("Quicklook, {} vs {}".format(x, y))
    plt.show()


def main(filename):
    df = pd.read_csv(filename)

    print(df.head())

    df.sort_values("chi2", inplace=True)

    df_small = df[:][:50]

    for x in ["teff1", "feh1", "logg1", "rvs", "gammas", "alpha"]:
        quicklook(df_small, x, "chi2")


if __name__ == "__main__":
    args = vars(_parser())
    opts = {k: args[k] for k in args}

    sys.exit(main(**opts))
