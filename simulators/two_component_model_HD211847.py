"""two_compoonent_model.py.

Jason Neal
2nd Janurary 2017

Compare observed spectra to many different phoenix-aces spectral models to find which matches one is best fit by itself.

Need to determine the best RV offset to apply to the spectra.
Possibly bu sampling a few and then taking average value (they should be all
the same I would think unless the lines changed dramatically).

"""
from __future__ import division, print_function

# import itertools
import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from utilities.spectrum_utils import load_spectrum  # , select_observation
from astropy.io import fits

from simulators.tcm_module import tcm_analysis, parallel_tcm_analysis, tcm_helper_function
from utilities.crires_utilities import barycorr_crires_spectrum
from utilities.phoenix_utils import (closest_model_params,
                                     generate_close_params)

logging.basicConfig(level=logging.WARNING,
                    format='%(levelname)s %(message)s')
debug = logging.debug

model_base_dir = ("/home/jneal/Phd/data/PHOENIX-ALL/PHOENIX/")
wav_dir = "/home/jneal/Phd/data/PHOENIX-ALL/PHOENIX/"

wav_model = fits.getdata(os.path.join(wav_dir, "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"))
wav_model /= 10   # turn into nm


import simulators
gammas = np.arange(*simulators.sim_grid["gammas"])
rvs = np.arange(*simulators.sim_grid["rvs"])
alphas = np.arange(*simulators.sim_grid["alphas"])


def _parser():
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='tcm')
    parser.add_argument('--chip', help='Chip Number.', default=None)
    parser.add_argument('-p', '--parallel', help='Use parallelization.', action="store_true")
    parser.add_argument('-s', '--small', help='Use smaller subset of parameters.', action="store_true")

    return parser.parse_args()


def main(chip=None, parallel=True, small=True, verbose=False):
    """Main function."""

    star = "HD211847"
    obs_num = 2

    if chip is None:
        chip = 4

    obs_name, params, output_prefix = tcm_helper_function(star, obs_num, chip)

    print("The observation used is ", obs_name, "\n")

    host_params = [params["temp"], params["logg"], params["fe_h"]]
    comp_params = [params["comp_temp"], params["logg"], params["fe_h"]]

    closest_host_model = closest_model_params(*host_params)  # unpack temp, logg, fe_h with *
    closest_comp_model = closest_model_params(*comp_params)

    # Function to find the good models I need
    # models = find_phoenix_model_names(model_base_dir, original_model)
    # Function to find the good models I need from parameters
    model1_pars = list(generate_close_params(closest_host_model, small=small))
    model2_pars = list(generate_close_params(closest_comp_model, small=small))

    # Load observation
    obs_spec = load_spectrum(obs_name)
    obs_spec = barycorr_crires_spectrum(obs_spec)
    # TODO
    # Mask out bad portion of observed spectra ## HACK
    if chip == 4:
        # Ignore first 40 pixels
        obs_spec.wav_select(obs_spec.xaxis[40], obs_spec.xaxis[-1])

    param_iter = len(alphas) * len(rvs) * len(gammas) * len(model2_pars) * len(model1_pars)
    print("STARTING tcm_analysis\nWith {} parameter iterations".format(param_iter))
    print("model1_pars", len(model1_pars), "model2_pars", len(model2_pars))

    ####
    if parallel:
        chi2_grids = parallel_tcm_analysis(obs_spec, model1_pars, model2_pars, alphas, rvs, gammas, verbose=verbose, norm=True, prefix=output_prefix, save_only=True)
    else:
        chi2_grids = tcm_analysis(obs_spec, model1_pars, model2_pars, alphas, rvs, gammas, verbose=verbose, norm=True, prefix=output_prefix)

    # Print TODO
    print("TODO: Add joining of sql table here")

    # subprocess.call(make_chi2_bd.py)


def check_inputs(var):
    if var is None:
        var = np.array([0])
    elif isinstance(rvs, (float, int)):
        var = np.asarray(var, dtype=np.float32)
    return var


if __name__ == "__main__":
    args = vars(_parser())
    opts = {k: args[k] for k in args}

    # Iterate over chips
    if opts["chip"] is None:
        for chip in range(1, 5):
            opts["chip"] = chip
            res = main(**opts)
        sys.exit(res)
    else:
        sys.exit(main(**opts))
