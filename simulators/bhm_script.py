#!/usr/bin/env python
"""Run bhm analysis for HD211847."""
import argparse
import os

import numpy as np
import pandas as pd

import simulators
from mingle.utilities.crires_utilities import barycorr_crires_spectrum
from mingle.utilities.errors import spectrum_error
from mingle.utilities.masking import spectrum_masking
from mingle.utilities.param_file import parse_paramfile
from mingle.utilities.phoenix_utils import closest_model_params, generate_close_params
from mingle.utilities.spectrum_utils import load_spectrum
from simulators.bhm_module import bhm_analysis
from simulators.iam_module import setup_dirs

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


def bhm_helper_function(star, obs_num, chip):
    param_file = os.path.join(simulators.paths["parameters"], "{}_params.dat".format(star))
    params = parse_paramfile(param_file, path=None)
    obs_name = os.path.join(
        simulators.paths["spectra"], "{0}-{1}-mixavg-tellcorr_{2}.fits".format(star, obs_num, chip))

    output_prefix = os.path.join(
        simulators.paths["output_dir"], star.upper(),
        "{0}-{1}_{2}_bhm_chisqr_results".format(star.upper(), obs_num, chip))
    return obs_name, params, output_prefix


def get_model_pars(params, method="close"):
    method = method.lower()
    if method == "all":
        raise NotImplementedError("Cant yet choose all parameters.")
    elif method == "close":
        host_params = [params["temp"], params["logg"], params["fe_h"]]
        # comp_params = [params["comp_temp"], params["logg"], params["fe_h"]]
        closest_host_model = closest_model_params(*host_params)

        # Model parameters to try iterate over.
        model_pars = list(generate_close_params(closest_host_model))
    else:
        raise ValueError("The method '{0}' is not valid".format(method))

    return model_pars


def save_pd_cvs(name, data):
    # Take dict of data to save to csv called name
    df = pd.DataFrame(data=data)
    df.to_csv(name, sep=',', index=False)
    return 0


def deconstruct_array(array, values):
    """Index of other arrays to apply these values to."""
    print("array shape", array.shape)
    print("array[:5]", array[:5])
    print("values.shape", values.shape)
    values2 = values * np.ones_like(array)
    print("values2.shape", values2.shape)
    print("values2.shape", values2[:5])
    for i in enumerate(array):
        indx = [0]
    gam = [0]
    chi2 = [0]
    return indx, gam, chi2


def main(star, obs_num, chip=None, verbose=False, suffix=None, mask=False, error_off=False, disable_wav_scale=False):
    """Best Host modelling main function."""
    wav_scale = not disable_wav_scale
    star = star.upper()
    setup_dirs(star)
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
    obs_spec = barycorr_crires_spectrum(obs_spec, -22)
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

    # indx, gam, chi2 = deconstruct_array(broadcast_chi2_gamma, gammas)

    # # Save the result to a csv, in a single column
    # save_results = {"temp": TEFF, "logg": LOGG, "fe_h": FEH,
    #                 "model_chisqr": chi2_grids[0],
    #                 "broadcast_chisqr": chi2_grids[3],
    #                 "broadcast_gamma": chi2_grids[4]}
    #
    # # save_pd_cvs(output_name, data=save_results)
    # cols = ["temp", "logg", "fe_h", "model_chisqr", "broadcast_chisqr", "broadcast_gamma"]
    # df = pd.DataFrame(data=save_results)
    # df.to_csv(output_name + ".tsv", sep='\t', index=False, columns=cols)
    # print("Save the results to {}".format(output_name))
    #
    # # Save as astropy table, and all gamma values from broadcasting.
    # save_results2 = {"temp": TEFF, "logg": LOGG, "fe_h": FEH,
    #                  "broadcast_chisqr": chi2_grids[3],
    #                  "broadcast_gamma": chi2_grids[4],
    #                  "chi2_gamma": broadcast_chi2_gamma[5], "gammas": gammas}
    #
    # print("Save the results to {}".format(output_name))
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

            # sys.exit(main(**opts))
