import sys
import json
import argparse

import numpy as np
from bhm_HD211847 import bhm_analysis


def _parser():
    """Take care of all the argparse stuff.

    :returns: the args
    """
    # parser = GooeyParser(description='Wavelength Calibrate CRIRES Spectra')
    parser = argparse.ArgumentParser(description='Wavelength Calibrate CRIRES \
                                    Spectra')
    parser.add_argument('star', help='Star name file')
    parser.add_argument('-n', '--obs_num', help='Obervation number')
    parser.add_argument('-d', '--detector', default=None, help='detector number, All if not provided.')  # if False/nune the [1,2,3,4]
    parser.add_argument('-o', '--output', default=False, help='Ouput Filename')
    parser.add_argument('-s', '--model', choices=["tcm", "bhm"],
                        help='Choose spectral model to evaulate, ["tcm"(default), "bhm"]')
    parser.add_argument('-m', '--mode', choices=["chi2", "plot"],
                        help='Calculate chi2 or plot results.')

    args = parser.parse_args()
    return args


def grids_from_config(name=None):
    """Load chi2 analysis params from json file."""
    if name is None:
        name = "chi2_config.json"
        with open(name, "r") as f:
            config_values = json.load(f)

        for param in ["alphas", "gammas", "rvs"]:
            if param not in values:
                raise ValueError("Chi2 config file is invalid.")

        alphas = np.arange(*config_values["alphas"])
        gammas = np.arange(*config_values["gammas"])
        rvs = np.arange(*config_values["rvs"])

        return gammas, rvs, alphas


def main(star, obs_num, detector, output=None, model="tcm", mode="plot"):
    if output is None:
        output = "Analysis-{0}-{1}_{2}-{}_chisqr_results.dat".format(star, obs_num, detector, model)

    if mode == "plot":
        # Load chi2 and dot he plotting
        if model == "bhm":

            pass
        elif model == "tcm":

            pass
    elif mode == "chi2":
        # Do the chi2 calcualtions and save to a file.
        if model == "bhm":
            gammas, __, __ = grids_from_config()

            bhm_analysis(obs_spec, model_pars, gammas=None, verbose=False, norm=False)
            pass

        elif model == "tcm":
            gammas, rvs, alphas = grids_from_config()
            pass


if __name__ == '__main__':
    args = vars(_parser())
    opts = {k: args[k] for k in args}

    sys.exit(main(**opts))
