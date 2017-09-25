"""inherint_alpha_model.py.

Jason Neal
24 August 2017

Using the flux ratio of the spectra themselves.
"""
from __future__ import division, print_function

import argparse
import logging
import os
import sys

import numpy as np
from astropy.io import fits
from joblib import Parallel, delayed

import simulators
from simulators.iam_module import (iam_analysis, iam_helper_function,
                                   parallel_iam_analysis)
# from utilities.chisqr import chi_squared
from utilities.crires_utilities import barycorr_crires_spectrum
from utilities.phoenix_utils import closest_model_params, generate_close_params
from utilities.spectrum_utils import load_spectrum


logging.basicConfig(level=logging.WARNING,
                    format='%(levelname)s %(message)s')
debug = logging.debug

wav_dir = simulators.starfish_grid["raw_path"]

wav_model = fits.getdata(os.path.join(wav_dir, "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"))
wav_model /= 10   # turn into nm

gammas = np.arange(*simulators.sim_grid["gammas"])
rvs = np.arange(*simulators.sim_grid["rvs"])
# alphas = np.arange(*simulators.sim_grid["alphas"])
# alphas = np.arange(0.01, 0.2, 0.02)


def _parser():
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Inherint alpha modelling.')
    parser.add_argument("star", help='Star name.', type=str)
    parser.add_argument("obs_num", help='Star observation number.', type=str)
    parser.add_argument('-c', '--chip', help='Chip Number.', default=None)
    parser.add_argument('-p', '--parallel', help='Use parallelization.', action="store_true")
    parser.add_argument('-s', '--small', help='Use smaller subset of parameters.', action="store_true")
    parser.add_argument('-m', '--more_id', help='Extra name identifier.', type=str)

    return parser.parse_args()


def main(star, obs_num, chip=None, parallel=True, small=True, verbose=False, more_id=None):
    """Main function."""

    if chip is None:
        chip = 4

    obs_name, params, output_prefix = iam_helper_function(star, obs_num, chip)
    if more_id is not None:
        output_prefix = output_prefix + str(more_id)

    print("The observation used is ", obs_name, "\n")

    host_params = [params["temp"], params["logg"], params["fe_h"]]
    comp_params = [params["comp_temp"], params["logg"], params["fe_h"]]

    closest_host_model = closest_model_params(*host_params)  # unpack temp, logg, fe_h with *
    closest_comp_model = closest_model_params(*comp_params)

    # Function to find the good models I need from parameters
    model1_pars = list(generate_close_params(closest_host_model, small="host"))
    model2_pars = list(generate_close_params(closest_comp_model, small=small))

    # Load observation
    obs_spec = load_spectrum(obs_name)
    obs_spec = barycorr_crires_spectrum(obs_spec)
    # TODO
    # Mask out bad portion of observed spectra ## HACK
    if chip == 4:
        # Ignore first 40 pixels
        obs_spec.wav_select(obs_spec.xaxis[40], obs_spec.xaxis[-1])

    param_iter = len(rvs) * len(gammas) * len(model2_pars) * len(model1_pars)
    print("STARTING iam_analysis\nWith {} parameter iterations".format(param_iter))
    # print("model1_pars", len(model1_pars), "model2_pars", len(model2_pars))

    ####
    if parallel:
        chi2_grids = parallel_iam_analysis(obs_spec, model1_pars, model2_pars,
                                           rvs, gammas, verbose=verbose,
                                           norm=True, prefix=output_prefix,
                                           save_only=True)
    else:
        chi2_grids = iam_analysis(obs_spec, model1_pars, model2_pars, rvs,
                                  gammas, verbose=verbose, norm=True,
                                  prefix=output_prefix)

    ####
    # Print TODO
    print("TODO: Add joining of sql table here")

    # subprocess.call(make_chi2_bd.py)


if __name__ == "__main__":
    args = vars(_parser())
    opts = {k: args[k] for k in args}

    def parallelized_main(opts, chip):
        opts["chip"] = chip
        return main(**opts)

    # Iterate over chips
    if opts["chip"] is None:
         res = Parallel(n_jobs=-1)(delayed(parallelized_main)(opts, chip)
                                   for chip in range(1, 5))
         sys.exit(sum(res))
    else:
        sys.exit(main(**opts))
