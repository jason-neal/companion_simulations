import numpy as np
import pandas as pd
import itertools
from best_host_model_HD211847 import bhm_analysis
from utilities.param_file import parse_paramfile
from utilities.phoenix_utils import closest_model_params, generate_close_params
from Chisqr_of_observation import load_spectrum
# from utilites.io import save_pd_csv
from utilities.crires_utilities import barycorr_crires_spectrum


def bhm_helper_function(star, obs_num, chip):
    param_file = "/home/jneal/Phd/data/parameter_files/{}_params.dat".format(star)
    params = parse_paramfile(param_file, path=None)
    obs_name = "/home/jneal/.handy_spectra/{}-{}-mixavg-tellcorr_{}.fits".format(star, obs_num, chip)

    output_name = "Analysis/{}-{}_{}_bhm_chisqr_results".format(star.upper(), obs_num, chip)
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
    df.to_csv(name + ".csv", sep=',', index=False)
    return 0


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
    gammas = np.arange(-20, 20, 1)

    iters = itertools.product(obs_num, chips)
    for obs_num, chip in iters:
        obs_name, params, output_name = bhm_helper_function(star, obs_num, chip)

        print("The observation used is ", obs_name, "\n")

        model_pars = get_model_pars(params, method="close")


        # Load observation
        obs_spec = load_spectrum(obs_name)
        obs_spec = barycorr_crires_spectrum(obs_spec, -22)
        obs_spec.flux /= 1.02
        # Mask out bad portion of observed spectra ## HACK
        if chip == 4:
            # Ignore first 40 pixels
            obs_spec.wav_select(obs_spec.xaxis[40], obs_spec.xaxis[-1])


    ####
        chi2_grids = bhm_analysis(obs_spec, model_pars, gammas, verbose=True)
    ####
        (model_chisqr_vals, model_xcorr_vals, model_xcorr_rv_vals,
            broadcast_chisqr_vals, broadcast_gamma) = chi2_grids

        TEFF = [par[0] for par in model_pars]
        LOGG = [par[1] for par in model_pars]
        FEH = [par[2] for par in model_pars]

        # testing shapes
        print("model_chisqr_vals", model_chisqr_vals.shape)
        print("model_xcorr_vals", model_xcorr_vals.shape)
        print("model_xcorr_rv_vals", model_xcorr_rv_vals.shape)
        print("broadcast_chisqr_vals", broadcast_chisqr_vals.shape)
        print("broadcast_gamma", broadcast_gamma.shape)

        # SAVE the result to a csv, in a single column
        save_results = {"temp": TEFF, "logg": LOGG, "fe_h": FEH,
                        "model_chisqr": chi2_grids[0],
                        "broadcast_chisqr": chi2_grids[3], "broadcast_gamma": chi2_grids[4]}

        save_pd_cvs(output_name, data=save_results)
        print("Save the results to {}".format(output_name))

    print("Finished chisquare generation")
