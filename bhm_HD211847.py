import itertools
import json

import numpy as np

import pandas as pd
from astropy.table import Table  # , Column
from best_host_model_HD211847 import bhm_analysis
from Chisqr_of_observation import load_spectrum
# from utilites.io import save_pd_csv
from utilities.crires_utilities import barycorr_crires_spectrum
from utilities.param_file import parse_paramfile
from utilities.phoenix_utils import closest_model_params, generate_close_params


def bhm_helper_function(star, obs_num, chip):
    param_file = "/home/jneal/Phd/data/parameter_files/{}_params.dat".format(star)
    params = parse_paramfile(param_file, path=None)
    obs_name = "/home/jneal/.handy_spectra/{0}-{1}-mixavg-tellcorr_{2}.fits".format(star, obs_num, chip)

    output_name = "Analysis/{0}/{0}-{1}_{2}_bhm_chisqr_results.dat".format(star.upper(), obs_num, chip)
    return obs_name, params, output_name


def get_model_pars(params, method="close"):
    method = method.lower()
    if method == "all":
        raise NotImplementedError("Cant yet choose all parameters.")
    elif method == "close":
        host_params = [params["temp"], params["logg"], params["fe_h"]]
        # comp_params = [params["comp_temp"], params["logg"], params["fe_h"]]
        closest_host_model = closest_model_params(*host_params)
        print("Closest model params = ", closest_host_model)
        # Model parameters to try iterate over.
        model_pars = list(generate_close_params(closest_host_model))

    else:
        raise ValueError("The method '{0}' is not valid".format(method))

    return model_pars


def save_pd_cvs(name, data):
    # Take dict of data to save to csv caled name
    df = pd.DataFrame(data=data)
    df.to_csv(name, sep=',', index=False)
    return 0


def save_astropytable(name, data):
    # for gg, gamma in enumerate(gammas):
    #    gamma_list = gamma * np.ones(len(data["teff"]))
    # data["gamma"] = gamma_list
    tb = Table(data)
    print(tb)


def get_maskinfo(star, obs_num, chip):
    with open("/home/jneal/.handy_spectra/detector_masks.json", "r") as f:
        mask_data = json.load(f)
    try:
        this_mask = mask_data[star][obs_num][str(chip)]
        print(this_mask)
        return this_mask
    except:
        print("No Masking data present for {0}-{1}_{2}".format(star, obs_num, chip))
        return []


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


if __name__ == "__main__":
    # ### ADD STAR INFO HERE
    star = "HD211847"
    param_file = "/home/jneal/Phd/data/parameter_files/{}_params.dat".format(star)
    params = parse_paramfile(param_file, path=None)
    obs_num = [1]  # [1, 2]
    chips = [1]    # [1, 2, 3, 4]
    # obs_name = select_observation(star, obs_num, chip)
    print("original_model = Z-0.0/lte05700-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")

    #########################
    # Define the broadcasted gamma grid
    gammas = np.arange(-2, 2, 1)

    iters = itertools.product(obs_num, chips)
    for obs_num, chip in iters:
        obs_name, params, output_name = bhm_helper_function(star, obs_num, chip)
        chip_masks = get_maskinfo(star, obs_num, chip)
        print("The observation used is ", obs_name, "\n")

        model_pars = get_model_pars(params, method="close")

        # Load observation
        obs_spec = load_spectrum(obs_name)
        obs_spec = barycorr_crires_spectrum(obs_spec, -22)
        # obs_spec.flux /= 1.02
        # Mask out bad portion of observed spectra ## HACK
        for mask_limits in chip_masks:
            if len(mask_limits) is not 2:
                raise ValueError("Mask limits in mask file is incorrect for {0}-{1}_{2}".format(star, obs_num, chip))
            obs_spec.wav_select(*mask_limits)

        # if chip == 4:
            # Ignore first 40 pixels
            # obs_spec.wav_select(obs_spec.xaxis[40], obs_spec.xaxis[-1])

    ####
        # chi2_grids = bhm_analysis(obs_spec, model_pars, gammas, verbose=True, norm=True)
        chi2_grids = bhm_analysis(obs_spec, model_pars, gammas, verbose=False, norm=False)
    ####
        (model_chisqr_vals, model_xcorr_vals, model_xcorr_rv_vals,
            broadcast_chisqr_vals, broadcast_gamma, broadcast_chi2_gamma) = chi2_grids

        TEFF = np.array([par[0] for par in model_pars])
        LOGG = np.array([par[1] for par in model_pars])
        FEH = np.array([par[2] for par in model_pars])

        # testing shapes
        print("model_chisqr_vals", model_chisqr_vals.shape)
        print("model_xcorr_vals", model_xcorr_vals.shape)
        print("model_xcorr_rv_vals", model_xcorr_rv_vals.shape)
        print("broadcast_chisqr_vals", broadcast_chisqr_vals.shape)
        print("broadcast_chisqr_vals", broadcast_chisqr_vals[:20])
        print("broadcast_gamma", broadcast_gamma.shape)
        print("broadcast_gamma", broadcast_gamma[:20])
        print("broadcast_chi2_gamma", broadcast_chi2_gamma.shape)
        print("broadcast_chi2_gamma", broadcast_chi2_gamma[:20])

        indx, gam, chi2 = deconstruct_array(broadcast_chi2_gamma, gammas)

        # Save the result to a csv, in a single column
        save_results = {"temp": TEFF, "logg": LOGG, "fe_h": FEH,
                        "model_chisqr": chi2_grids[0],
                        "broadcast_chisqr": chi2_grids[3],
                        "broadcast_gamma": chi2_grids[4]}

        # save_pd_cvs(output_name, data=save_results)
        df = pd.DataFrame(data=save_results)
        df.to_csv(output_name + ".tsv", sep='\b', index=False, cols={"temp", "logg", "fe_h", "model_chisqr", "broadcast_chisqr", "broadcast_gamma"})
        print("Save the results to {}".format(output_name))

        # Save as atropy table, and all gamma values from broadcasting.
        save_results2 = {"temp": TEFF, "logg": LOGG, "fe_h": FEH,
                         "broadcast_chisqr": chi2_grids[3],
                         "broadcast_gamma": chi2_grids[4],
                         "chi2_gamma": broadcast_chi2_gamma[5], "gammas": gammas}

        save_astropytable(output_name, data=save_results2)
        print("Save the results to {}".format(output_name))
    print("Finished chisquare generation")
