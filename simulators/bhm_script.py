#!/usr/bin/env python
"""Run bhm analysis for HD211847."""
import argparse
import sys

import numpy as np
import logging
import simulators
from joblib import Parallel, delayed
from logutils import BraceMessage as __
from mingle.utilities.crires_utilities import barycorr_crires_spectrum
from mingle.utilities.errors import spectrum_error, betasigma_error
from mingle.utilities.masking import spectrum_masking
from mingle.utilities.spectrum_utils import load_spectrum
from simulators.bhm_module import bhm_analysis, bhm_helper_function, get_bhm_model_pars
from simulators.bhm_module import setup_bhm_dirs

from bin.coadd_bhm_db import main as coadd_db
from bin.coadd_bhm_analysis import main as coadd_analysis

from argparse import Namespace
from typing import List


def parse_args(args: List[str]) -> Namespace:
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Best host modelling.')
    parser.add_argument("star", help='Star name.', type=str)
    parser.add_argument("obsnum", help='Star observation number.')
    parser.add_argument('-c', '--chip', help='Chip Number.', default=None)
    parser.add_argument("-j", "--n_jobs", help="Number of parallel Jobs",
                        default=1, type=int)
    parser.add_argument('-s', '--suffix', type=str, default="",
                        help='Extra name identifier.')
    parser.add_argument("-n", "--renormalize", help="Scalar re-normalize flux to models. Default=False",
                        action="store_true")
    parser.add_argument("-m", "--norm_method", help="Re-normalization method flux to models. Default=scalar",
                        choices=["scalar", "linear"], default="scalar")
    parser.add_argument("--error_off", action="store_true",
                        help="Turn snr value errors off.")
    parser.add_argument('--disable_wav_scale', action="store_true",
                        help='Disable scaling by wavelength.')
    parser.add_argument("-b", '--betasigma', help='Use BetaSigma std estimator.',
                        action="store_true")
    parser.add_argument('-v', '--verbose', action="store_true",
                        help='Turn on Verbose.')
    parser.add_argument('-x', '--strict_mask', action="store_true",
                        help='Apply extra strict masking.')
    return parser.parse_args(args)


def main(star, obsnum, chip=None, suffix=None, error_off=False, disable_wav_scale=False, renormalize=False,
         norm_method="scalar", betasigma=False, strict_mask=False):
    """Best Host modelling main function."""
    wav_scale = not disable_wav_scale
    star = star.upper()
    setup_bhm_dirs(star)
    # Define the broadcasted gamma grid
    gammas = np.arange(*simulators.sim_grid["gammas"])
    # print("bhm gammas", gammas)

    obs_name, params, output_prefix = bhm_helper_function(star, obsnum, chip)

    if suffix is not None:
        output_prefix = output_prefix + str(suffix)
    print("The observation used is ", obs_name, "\n")

    # Host Model parameters to iterate over
    # model_pars = get_bhm_model_pars(params, method="close")
    model_pars = get_bhm_model_pars(params, method="config")  # Use config file

    # Load observation
    obs_spec = load_spectrum(obs_name)
    from spectrum_overload import Spectrum
    assert isinstance(obs_spec, Spectrum)
    # Mask out bad portion of observed spectra
    obs_spec = spectrum_masking(obs_spec, star, obsnum, chip, stricter=strict_mask)

    # Barycentric correct spectrum
    _obs_spec = barycorr_crires_spectrum(obs_spec, extra_offset=None)

    # Determine Spectrum Errors
    try:
        if betasigma:
            print("DOING BETASIGMA ERRORS")
            N = simulators.betasigma.get("N", 5)
            j = simulators.betasigma.get("j", 2)
            errors, derrors = betasigma_error(obs_spec, N=N, j=j)
            print("Beta-Sigma error value = {:6.5f}+/-{:6.5f}".format(errors, derrors))
            logging.info(__("Beta-Sigma error value = {:6.5f}+/-{:6.5f}", errors, derrors))
        else:
            print("NOT DOING BETASIGMA ERRORS")
            errors = spectrum_error(star, obsnum, chip, error_off=error_off)
            logging.info(__("File obtained error value = {0}", errors))
    except KeyError as e:
        print("ERRORS Failed so set to None")
        errors = None

    bhm_analysis(obs_spec, model_pars, gammas, errors=errors, verbose=False, norm=renormalize,
                 wav_scale=wav_scale, prefix=output_prefix, norm_method=norm_method)
    print("after bhm_analysis")

    print("\nNow use bin/coadd_bhm_db.py")
    return 0


if __name__ == "__main__":
    args = vars(parse_args(sys.argv[1:]))
    opts = {k: args[k] for k in args}
    n_jobs = opts.pop("n_jobs", 1)
    verbose = opts.pop("verbose", False)


    def parallelized_main(main_opts, chip):
        main_opts["chip"] = chip
        return main(**main_opts)


    if opts["chip"] is None:
        chip_nums = 3 if opts.get("strict_mask", False) else 4
        res = Parallel(n_jobs=n_jobs)(delayed(parallelized_main)(opts, chip)
                                      for chip in range(1, chip_nums + 1))
        if not sum(res):
            try:
                print("\nDoing analysis after simulations!\n")
                coadd_db(opts["star"], opts["obsnum"], opts["suffix"], replace=True,
                         verbose=verbose, move=True)

                coadd_analysis(opts["star"], opts["obsnum"], suffix=opts["suffix"],
                               echo=False, mode="all", verbose=verbose, npars=1)

                print("\nFinished the db analysis after bhm_script simulations!\n")
                print("Initial bhm_script parameters = {}".format(args))
                sys.exit(0)
            except Exception as e:
                print("Unable to correctly do chi2 analysis after bhm_script")
                print(e)
        else:
            sys.exit(sum(res))
    else:
        sys.exit(main(**opts))
