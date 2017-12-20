#!/usr/bin/env python
"""two_compoonent_model.py.

Jason Neal
2nd Janurary 2017

Compare observed spectra to many different phoenix-aces spectral models to find which matches one is best fit by itself.

Need to determine the best RV offset to apply to the spectra.
Possibly bu sampling a few and then taking average value (they should be all
the same I would think unless the lines changed dramatically).

"""

import argparse
import logging
import os
import sys

import numpy as np
from astropy.io import fits

import simulators
from mingle.utilities.crires_utilities import barycorr_crires_spectrum
from mingle.utilities.errors import spectrum_error
from mingle.utilities.masking import spectrum_masking
from mingle.utilities.phoenix_utils import closest_model_params, generate_close_params
from mingle.utilities.spectrum_utils import load_spectrum  # , select_observation
from simulators.tcm_module import (tcm_analysis, tcm_helper_function, setup_tcm_dirs)

logging.basicConfig(level=logging.WARNING,
                    format='%(levelname)s %(message)s')

wav_dir = simulators.starfish_grid["raw_path"]

wav_model = fits.getdata(os.path.join(wav_dir, "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"))
wav_model /= 10  # turn into nm

gammas = np.arange(*simulators.sim_grid["gammas"])
rvs = np.arange(*simulators.sim_grid["rvs"])
alphas = np.arange(*simulators.sim_grid["alphas"])


def parse_args(args):
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='tcm')
    parser.add_argument('--chip', help='Chip Number.', default=None)
    parser.add_argument('-s', '--small', help='Use smaller subset of parameters.', action="store_true")
    parser.add_argument("--error_off", help="Turn snr value errors off.", action="store_true")
    parser.add_argument('--disable_wav_scale', action="store_true",
                        help='Disable scaling by wavelength.')
    return parser.parse_args(args)


def main(chip=None, small=True, verbose=False, error_off=False, disable_wav_scale=False):
    """Main function."""
    wav_scale = not disable_wav_scale

    star = "HD211847"
    obsnum = 2
    star = star.upper()
    setup_tcm_dirs(star)
    if chip is None:
        chip = 4

    obs_name, params, output_prefix = tcm_helper_function(star, obsnum, chip)

    logging.debug("The observation used is ", obs_name, "\n")

    host_params = [params["temp"], params["logg"], params["fe_h"]]
    comp_params = [params["comp_temp"], params["logg"], params["fe_h"]]

    closest_host_model = closest_model_params(*host_params)  # unpack temp, logg, fe_h with *
    closest_comp_model = closest_model_params(*comp_params)

    # Function to find the good models I need from parameters
    model1_pars = list(generate_close_params(closest_host_model, small=small))
    model2_pars = list(generate_close_params(closest_comp_model, small=small))

    # Load observation
    obs_spec = load_spectrum(obs_name)
    # Mask out bad portion of observed spectra
    obs_spec = spectrum_masking(obs_spec, star, obsnum, chip)
    # Barycentric correct spectrum
    _obs_spec = barycorr_crires_spectrum(obs_spec, extra_offset=None)
    # Determine Spectrum Errors
    try:
        errors = spectrum_error(star, obsnum, chip, error_off=error_off)
    except KeyError as e:
        errors = None

    param_iter = len(alphas) * len(rvs) * len(gammas) * len(model2_pars) * len(model1_pars)
    logging.info("STARTING tcm_analysis\nWith {} parameter iterations".format(param_iter))
    logging.debug("model1_pars", len(model1_pars), "model2_pars", len(model2_pars))

    chi2_grids = tcm_analysis(obs_spec, model1_pars, model2_pars, alphas, rvs, gammas, errors=errors,
                              verbose=verbose, norm=True, prefix=output_prefix, wav_scale=wav_scale)

    return 0


if __name__ == "__main__":
    args = vars(parse_args(sys.argv[1:]))
    opts = {k: args[k] for k in args}

    # Iterate over chips
    if opts["chip"] is None:
        for chip in range(1, 5):
            opts["chip"] = chip
            res = main(**opts)
        sys.exit(res)
    else:
        sys.exit(main(**opts))
