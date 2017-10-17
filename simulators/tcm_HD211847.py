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

import numpy as np

import simulators
from astropy.io import fits
from simulators.tcm_module import (parallel_tcm_analysis, tcm_analysis,
                                   tcm_helper_function)
from utilities.crires_utilities import barycorr_crires_spectrum
from utilities.phoenix_utils import closest_model_params, generate_close_params
from utilities.spectrum_utils import load_spectrum  # , select_observation

logging.basicConfig(level=logging.WARNING,
                    format='%(levelname)s %(message)s')
debug = logging.debug

wav_dir = simulators.starfish_grid["raw_path"]

wav_model = fits.getdata(os.path.join(wav_dir, "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"))
wav_model /= 10   # turn into nm


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

    # Function to find the good models I need from parameters
    model1_pars = list(generate_close_params(closest_host_model, small=small))
    model2_pars = list(generate_close_params(closest_comp_model, small=small))

    # Load observation
    obs_spec = load_spectrum(obs_name)
    obs_spec = barycorr_crires_spectrum(obs_spec)

    # Mask out bad portion of observed spectra ## HACK
    chip_masks = get_maskinfo(star, obs_num, chip)
    if chip == 4:
        # Ignore first 50 pixels of detector 4
        obs_spec.wav_select(obs_spec.xaxis[50], obs_spec.xaxis[-1])
    for mask_limits in chip_masks:
        if len(mask_limits) is not 2:
            raise ValueError("Mask limits in mask file is incorrect for {0}-{1}_{2}".format(star, obs_num, chip))
        obs_spec.wav_select(*mask_limits)  # Wavelengths to include


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
