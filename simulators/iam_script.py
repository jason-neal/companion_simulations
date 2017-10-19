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

import simulators
from astropy.io import fits
from joblib import Parallel, delayed
from simulators.iam_module import (iam_analysis, iam_helper_function,
                                   parallel_iam_analysis)
from utilities.crires_utilities import barycorr_crires_spectrum
from utilities.masking import spectrum_masking
from utilities.phoenix_utils import (closest_model_params,
                                     generate_close_params,
                                     generate_close_params_with_simulator)
# from utilities.chisqr import chi_squared
from utilities.simulation_utilities import check_inputs
from utilities.spectrum_utils import load_spectrum

logging.basicConfig(level=logging.WARNING,
                    format='%(levelname)s %(message)s')
debug = logging.debug

wav_dir = simulators.starfish_grid["raw_path"]

wav_model = fits.getdata(os.path.join(wav_dir, "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"))
wav_model /= 10   # turn into nm

gammas = np.arange(*simulators.sim_grid["gammas"])
rvs = np.arange(*simulators.sim_grid["rvs"])
# Pre-check_rv_vals(rvs)
check_inputs(rvs)
check_inputs(gammas)
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
    parser.add_argument('-p', '--parallel', help='Use parallelization.',
                        action="store_true")
    parser.add_argument("-n", "--n_jobs", help="Number of parallel Jobs",
                        default=1, type=int)

    parser.add_argument('-s', '--small', action="store_true",
                        help='Use smaller subset of parameters.')
    parser.add_argument('--suffix', help='Suffix for file.', type=str)

    return parser.parse_args()


def main(star, obs_num, chip=None, parallel=True, small=True, verbose=False, suffix=None):
    """Main function."""
    if chip is None:
        chip = 4

    obs_name, params, output_prefix = iam_helper_function(star, obs_num, chip)
    if suffix is not None:
        output_prefix = output_prefix + str(suffix)

    print("The observation used is ", obs_name, "\n")

    host_params = [params["temp"], params["logg"], params["fe_h"]]
    comp_params = [params["comp_temp"], params["logg"], params["fe_h"]]

    closest_host_model = closest_model_params(*host_params)  # unpack temp, logg, fe_h with *
    closest_comp_model = closest_model_params(*comp_params)

    # Function to find the good models I need from parameters
    model1_pars = list(generate_close_params_with_simulator(
        closest_host_model, "host", small="host", limits="phoenix"))
    model2_pars = list(generate_close_params_with_simulator(
        closest_comp_model, "companion", small=small, limits="phoenix"))
    # model1_pars = list(generate_close_params(closest_host_model, small="host"))
    # model2_pars = list(generate_close_params(closest_comp_model, small=small))

    # Load observation
    obs_spec = load_spectrum(obs_name)
    # Mask out bad portion of observed spectra
    obs_spec = spectrum_masking(obs_spec, star, obs_num, chip)
    # Barycentric correct spectrum
    obs_spec = barycorr_crires_spectrum(obs_spec)

    rv_iter = len(rvs) * len(gammas)
    model_iter = len(model2_pars) * len(model1_pars)
    print(("STARTING iam_analysis\nWith {0} parameter iterations.\n{1} rv iterations,"
          " {2} model iterations").format(rv_iter * model_iter, rv_iter, model_iter))

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
    return 0


if __name__ == "__main__":
    args = vars(_parser())
    opts = {k: args[k] for k in args}
    n_jobs = opts.pop("n_jobs", 1)

    def parallelized_main(main_opts, chip):
        main_opts["chip"] = chip
        return main(**main_opts)

    # Iterate over chips
    if opts["chip"] is None:
        res = Parallel(n_jobs=n_jobs)(delayed(parallelized_main)(opts, chip)
                                      for chip in range(1, 5))
        sys.exit(sum(res))
    else:
        sys.exit(main(**opts))
