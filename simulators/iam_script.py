#!/usr/bin/env python
"""inherent_alpha_model.py.

Jason Neal
24 August 2017

Using the flux ratio of the spectra themselves.
"""

import argparse
import logging
import os
import sys

import numpy as np
from astropy.io import fits
from joblib import Parallel, delayed

import simulators
from bin.coadd_analysis_script import main as coadd_analysis
from bin.coadd_chi2_db import main as coadd_db
from mingle.utilities.crires_utilities import barycorr_crires_spectrum
from mingle.utilities.errors import spectrum_error
from mingle.utilities.masking import spectrum_masking
from mingle.utilities.phoenix_utils import (closest_model_params,
                                            generate_close_params_with_simulator)
from mingle.utilities.simulation_utilities import check_inputs
from mingle.utilities.spectrum_utils import load_spectrum
from simulators.iam_module import (iam_analysis, iam_helper_function,
                                   setup_iam_dirs)

logging.basicConfig(level=logging.WARNING,
                    format='%(levelname)s %(message)s')

# wav_dir = simulators.starfish_grid["raw_path"]
# wav_model = fits.getdata(os.path.join(wav_dir, "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"))
# wav_model /= 10  # turn into nm

gammas = np.arange(*simulators.sim_grid["gammas"])
rvs = np.arange(*simulators.sim_grid["rvs"])
# Pre-check_rv_vals(rvs)
check_inputs(rvs)
check_inputs(gammas)


def parse_args(args):
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Inherint alpha modelling.')
    parser.add_argument("star", help='Star name.', type=str)
    parser.add_argument("obsnum", help='Star observation number.', type=str)
    parser.add_argument('-c', '--chip', help='Chip Number.', default=None)
    parser.add_argument("-j", "--n_jobs", help="Number of parallel Jobs",
                        default=1, type=int)
    parser.add_argument("-n", "--renormalize", help="Scalar re-normalize flux to models. Default=False",
                        action="store_true")
    parser.add_argument("-m", "--norm_method", help="Re-normalization method flux to models. Default=scalar",
                        choices=["scalar", "linear"], default="scalar")
    parser.add_argument("--error_off", help="Turn snr value errors off.",
                        action="store_true")
    parser.add_argument('-s', '--small', action="store_true",
                        help='Use smaller subset of parameters.')
    parser.add_argument('-a', '--area_scale', action="store_false",
                        help='Scaling by stellar area. (raise to disable)')
    parser.add_argument('--disable_wav_scale', action="store_true",
                        help='Disable scaling by wavelength.')
    parser.add_argument('--suffix', help='Suffix for file.', type=str)

    return parser.parse_args(args)


def main(star, obsnum, chip=None, parallel=False, small=True, verbose=False,
         suffix=None, error_off=False, area_scale=True, disable_wav_scale=False, renormalize=False, norm_method="scalar"):
    """Main function."""
    wav_scale = not disable_wav_scale
    if chip is None:
        chip = 4

    star = star.upper()
    setup_iam_dirs(star)
    obs_name, params, output_prefix = iam_helper_function(star, obsnum, chip)
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

    rv_iter = len(rvs) * len(gammas)
    model_iter = len(model2_pars) * len(model1_pars)
    print(("\nSTARTING iam_analysis\n{0} parameter iterations\n{1} rv iterations\n"
           "{2} model iterations\n\n").format(rv_iter * model_iter, rv_iter, model_iter))

    # IAM Analysis
    chi2_grids = iam_analysis(obs_spec, model1_pars, model2_pars, rvs,
                                  gammas, verbose=verbose, norm=renormalize,
                                  prefix=output_prefix, errors=errors,
                                  area_scale=area_scale, wav_scale=wav_scale,
                                  norm_method=norm_method)

    print("\nNow use bin/coadd_chi2_db.py")
    return 0


if __name__ == "__main__":
    args = vars(parse_args(sys.argv[1:]))
    opts = {k: args[k] for k in args}
    n_jobs = opts.pop("n_jobs", 1)


    def parallelized_main(main_opts, chip):
        main_opts["chip"] = chip
        return main(**main_opts)

    # Iterate over chips
    if opts["chip"] is None:
        res = Parallel(n_jobs=n_jobs)(delayed(parallelized_main)(opts, chip)
                                      for chip in range(1, 5))
        print("Finished parallel loops")
        if not sum(res):

            print("\nDoing analysis after simulations!\n")
            coadd_db(opts["star"], opts["obsnum"], opts["suffix"], replace=True,
                     verbose=True, move=True)

            coadd_analysis(opts["star"], opts["obsnum"], suffix=opts["suffix"],
                           echo=False, mode="all", verbose=False, npars=3)

            print("\nFinished the db analysis after iam_script simulations!\n")
            sys.exit(0)
        else:
            sys.exit(sum(res))
    else:
        sys.exit(main(**opts))
