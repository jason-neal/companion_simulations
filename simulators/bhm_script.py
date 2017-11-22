#!/usr/bin/env python
"""Run bhm analysis for HD211847."""
import argparse

import numpy as np

import simulators
from mingle.utilities.crires_utilities import barycorr_crires_spectrum
from mingle.utilities.errors import spectrum_error
from mingle.utilities.masking import spectrum_masking
from mingle.utilities.spectrum_utils import load_spectrum
from simulators.bhm_module import bhm_analysis, bhm_helper_function, get_model_pars
from simulators.bhm_module import setup_bhm_dirs


def _parser():
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Best host modelling.')
    parser.add_argument("star", help='Star name.', type=str)
    parser.add_argument("obs_nums", help='Star observation number.', nargs="+")
    parser.add_argument('-c', '--chips', help='Chip Number.', default=None, nargs="+")
    parser.add_argument('-m', '--mask', action="store_true",
                        help='Apply wavelength mask.')
    parser.add_argument('-s', '--suffix', type=str, default="",
                        help='Extra name identifier.')
    parser.add_argument("--error_off", action="store_true",
                        help="Turn snr value errors off.")
    parser.add_argument('--disable_wav_scale', action="store_true",
                        help='Disable scaling by wavelength.')
    return parser.parse_args()


def main(star, obs_num, chip=None, verbose=False, suffix=None, mask=False, error_off=False, disable_wav_scale=False):
    """Best Host modelling main function."""
    wav_scale = not disable_wav_scale
    star = star.upper()
    setup_bhm_dirs(star)
    # Define the broadcasted gamma grid
    gammas = np.arange(*simulators.sim_grid["gammas"])
    print("bhm gammas", gammas)

    obs_name, params, output_prefix = bhm_helper_function(star, obs_num, chip)

    if suffix is not None:
        output_prefix = output_prefix + str(suffix)
    print("The observation used is ", obs_name, "\n")

    # Host Model parameters to iterate over
    model_pars = get_model_pars(params, method="close")

    # Load observation
    obs_spec = load_spectrum(obs_name)
    # Mask out bad portion of observed spectra
    obs_spec = spectrum_masking(obs_spec, star, obs_num, chip)
    # Barycentric correct spectrum
    obs_spec = barycorr_crires_spectrum(obs_spec, extra_offset=None)

    # Determine Spectrum Errors
    try:
        errors = spectrum_error(star, obs_num, chip, error_off=error_off)
    except KeyError as e:
        errors = None

    chi2_grids = bhm_analysis(obs_spec, model_pars, gammas, errors=errors, verbose=False, norm=False,
                              wav_scale=wav_scale, prefix=output_prefix)
    print("after bhm_analysis")

    (model_chisqr_vals, model_xcorr_vals, model_xcorr_rv_vals,
     broadcast_chisqr_vals, broadcast_gamma, broadcast_chi2_gamma) = chi2_grids

    TEFF = np.array([par[0] for par in model_pars])
    LOGG = np.array([par[1] for par in model_pars])
    FEH = np.array([par[2] for par in model_pars])

    # Testing shapes
    print("model_chisqr_vals", model_chisqr_vals.shape)
    print("model_xcorr_vals", model_xcorr_vals.shape)
    print("model_xcorr_rv_vals", model_xcorr_rv_vals.shape)
    print("broadcast_chisqr_vals", broadcast_chisqr_vals.shape)
    print("broadcast_chisqr_vals", broadcast_chisqr_vals[:20])
    print("broadcast_gamma", broadcast_gamma.shape)
    print("broadcast_gamma", broadcast_gamma[:20])
    print("broadcast_chi2_gamma", broadcast_chi2_gamma.shape)
    print("broadcast_chi2_gamma", broadcast_chi2_gamma[:20])

    print("Finished chi square generation")


if __name__ == "__main__":
    args = vars(_parser())
    opts = {k: args[k] for k in args}
    star = opts.pop("star")
    obs_nums = opts.pop("obs_nums")
    chips = opts.pop("chips")

    if chips is None:
        chips = range(1, 5)

    for obs in obs_nums:
        for chip in chips:
            main(star, obs, chip, **opts)
