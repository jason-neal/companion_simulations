#!/usr/bin/env python

import argparse
import sys

import numpy as np

import simulators
from simulators.bhm_module import bhm_analysis


def parse_args(args):
    """Take care of all the argparse stuff.

    :returns: the args
    """
    # parser = GooeyParser(description='Wavelength Calibrate CRIRES Spectra')
    parser = argparse.ArgumentParser(description='Perform bhm computations.')
    parser.add_argument('star', help='Star name file')
    parser.add_argument('-n', '--obs_num', help='Observation number')
    parser.add_argument('-d', '--detector', default=None,
                        help='detector number, All if not provided.')
    parser.add_argument('-o', '--output', default=False, help='Output Filename')
    parser.add_argument('-s', '--model', choices=["tcm", "bhm"],
                        help='Choose spectral model to evaluate, ["tcm"(default), "bhm"]')
    parser.add_argument('-m', '--mode', choices=["chi2", "plot"],
                        help='Calculate chi2 or plot results.')

    return parser.parse_args(args)


alphas = np.arange(*simulators.sim_grid["alphas"])
gammas = np.arange(*simulators.sim_grid["gammas"])
rvs = np.arange(*simulators.sim_grid["rvs"])


def main(star, obs_num, detector, output=None, model="tcm", mode="plot"):
    if output is None:
        output = "Analysis-{0}-{1}_{2}-{}_chisqr_results.dat".format(
            star, obs_num, detector, model)

    if mode == "plot":
        # Load chi2 and dot he plotting
        if model == "bhm":

            pass
        elif model == "tcm":

            pass
    elif mode == "chi2":
        # Do the chi2 calculations and save to a file.
        if model == "bhm":

            bhm_analysis(obs_spec, model_pars, gammas=gammas, verbose=False, norm=False)
            pass

        elif model == "tcm":
            # use gammas, rvs, alphas ...
            # tcm_analysis()
            pass
        elif model == "iam":
            # use gammas, rvs, alphas ...
            # iam_analysis()
            pass


if __name__ == '__main__':
    args = vars(parse_args(sys.argv[1:]))
    opts = {k: args[k] for k in args}

    sys.exit(main(**opts))
