#!/usr/bin/env python
"""Run bhm analysis for HD211847."""
import argparse
import sys

import numpy as np

import simulators
from mingle.utilities.crires_utilities import barycorr_crires_spectrum
from mingle.utilities.errors import spectrum_error
from mingle.utilities.masking import spectrum_masking
from mingle.utilities.spectrum_utils import load_spectrum
from simulators.bhm_module import bhm_analysis, bhm_helper_function, get_model_pars
from simulators.bhm_module import setup_bhm_dirs

from bin.coadd_bhm_db import main as coadd_db
from bin.coadd_bhm_analysis import main as coadd_analysis


def parse_args(args):
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Best host modelling.')
    parser.add_argument("star", help='Star name.', type=str)
    parser.add_argument("obsnum", help='Star observation number.')
    parser.add_argument('-c', '--chip', help='Chip Number.', default=None)
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
    return parser.parse_args(args)


def main(star, obsnum, chip=None, suffix=None, error_off=False, disable_wav_scale=False, renormalize=False,
         norm_method="scalar"):
    """Best Host modelling main function."""
    wav_scale = not disable_wav_scale
    star = star.upper()
    setup_bhm_dirs(star)
    # Define the broadcasted gamma grid
    gammas = np.arange(*simulators.sim_grid["gammas"])
    print("bhm gammas", gammas)

    obs_name, params, output_prefix = bhm_helper_function(star, obsnum, chip)

    if suffix is not None:
        output_prefix = output_prefix + str(suffix)
    print("The observation used is ", obs_name, "\n")

    # Host Model parameters to iterate over
    model_pars = get_model_pars(params, method="close")

    # Load observation
    obs_spec = load_spectrum(obs_name)
    from spectrum_overload import Spectrum
    assert isinstance(obs_spec, Spectrum)
    # Mask out bad portion of observed spectra
    obs_spec = spectrum_masking(obs_spec, star, obsnum, chip)

    # Barycentric correct spectrum
    _obs_spec = barycorr_crires_spectrum(obs_spec, extra_offset=None)

    # Determine Spectrum Errors
    try:
        errors = spectrum_error(star, obsnum, chip, error_off=error_off)
    except KeyError as e:
        errors = None

    chi2_grids = bhm_analysis(obs_spec, model_pars, gammas, errors=errors, verbose=False, norm=renormalize,
                              wav_scale=wav_scale, prefix=output_prefix, norm_method=norm_method)
    print("after bhm_analysis")

    (model_chisqr_vals, model_xcorr_vals, model_xcorr_rv_vals,
     broadcast_chisqr_vals, broadcast_gamma, broadcast_chi2_gamma) = chi2_grids

    # TEFF = np.array([par[0] for par in model_pars])
    # LOGG = np.array([par[1] for par in model_pars])
    # FEH = np.array([par[2] for par in model_pars])

    # Testing shapes
    print("Finished chi square generation")
    print("\nNow use bin/coadd_bhm_db.py")


if __name__ == "__main__":
    args = vars(parse_args(sys.argv[1:]))
    opts = {k: args[k] for k in args}
    star = opts.pop("star")

    chips = opts.pop("chip")

    if chips is None:
        chips = range(1, 5)
        do_after = True
    else:
        do_after = False

    for chip in chips:
        main(star, chip=chip, **opts)

    if do_after:
        print("\nDoing analysis after simulations!\n")
        coadd_db(star, opts["obsnum"], opts["suffix"], replace=True,
                 verbose=True, move=True)

        coadd_analysis(star, opts["obsnum"], suffix=opts["suffix"],
                       echo=False, mode="all", verbose=False, npars=3)

        print("\nFinished the db analysis after iam_script simulations!\n")
