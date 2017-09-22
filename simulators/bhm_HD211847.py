"""best_host_mode_HD211847l.py.

Jason Neal
2nd Janurary 2017

Compare observed spectra to many different phoenix-aces spectral models to find which matches one is best fit by itself.

Need to determine the best RV offset to apply to the spectra.
Possibly bu sampling a few and then taking average value (they should be all
the same I would think unless the lines changed dramatically).

"""
from __future__ import division, print_function

import logging
import os
import sys
from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from simulators.bhm_module import bhm_analysis
from utilities.crires_utilities import barycorr_crires_spectrum
from utilities.debug_utils import pv
from utilities.param_file import parse_paramfile
from utilities.phoenix_utils import (closest_model_params,
                                     generate_close_params,
                                     load_starfish_spectrum)
from utilities.spectrum_utils import load_spectrum

logging.basicConfig(level=logging.WARNING,
                    format='%(levelname)s %(message)s')
debug = logging.debug

model_base_dir = ("/home/jneal/Phd/data/PHOENIX-ALL/PHOENIX/")
wav_dir = "/home/jneal/Phd/data/PHOENIX-ALL/PHOENIX/"

wav_model = fits.getdata(os.path.join(wav_dir, "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"))
wav_model /= 10   # turn into nm


def main():
    """Main function."""
    star = "HD211847"
    param_file = "/home/jneal/Phd/data/parameter_files/{}_params.dat".format(star)
    params = parse_paramfile(param_file, path=None)
    host_params = [params["temp"], params["logg"], params["fe_h"]]
    # comp_params = [params["comp_temp"], params["logg"], params["fe_h"]]

    obs_num = 2
    chip = 4

    obs_name = "/home/jneal/.handy_spectra/{}-{}-mixavg-tellcorr_{}.fits".format(star, obs_num, chip)
    print("The observation used is ", obs_name, "\n")

    closest_host_model = closest_model_params(*host_params)  # unpack temp, logg, fe_h with *
    # closest_comp_model = closest_model_params(*comp_params)

    # original_model = "Z-0.0/lte05700-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    # debug(pv("closest_host_model"))
    # debug(pv("closest_comp_model"))
    # debug(pv("original_model"))

    # Function to find the good models I need
    # models = find_phoenix_model_names(model_base_dir, original_model)
    # Function to find the good models I need from parameters
    # model_par_gen = generate_close_params(closest_host_model)
    model_pars = list(generate_close_params(closest_host_model))  # Turn to list

    print("Model parameters", model_pars)

    # Load observation
    obs_spec = load_spectrum(obs_name)
    obs_spec = barycorr_crires_spectrum(obs_spec, -22)
    obs_spec.flux /= 1.02
    # Mask out bad portion of observed spectra ## HACK
    if chip == 4:
        # Ignore first 40 pixels
        obs_spec.wav_select(obs_spec.xaxis[40], obs_spec.xaxis[-1])

    import simulators
    gammas = np.arange(*simulators.sim_grids["gammas"])


    ####
    chi2_grids = bhm_analysis(obs_spec, model_pars, gammas, verbose=True)
    ####
    (model_chisqr_vals, model_xcorr_vals, model_xcorr_rv_vals,
        broadcast_chisqr_vals, broadcast_gamma, broadcast_chisquare) = chi2_grids

    TEFF = [par[0] for par in model_pars]
    LOGG = [par[1] for par in model_pars]
    FEH = [par[2] for par in model_pars]

    plt.plot(TEFF, broadcast_chisqr_vals, "+", label="broadcast")
    plt.plot(TEFF, model_chisqr_vals, ".", label="org")
    plt.title("TEFF vs Broadcast chisqr_vals")
    plt.legend()
    plt.show()
    plt.plot(TEFF, broadcast_gamma, "o")
    plt.title("TEFF vs Broadcast gamma grid")
    plt.show()

    plt.plot(LOGG, broadcast_chisqr_vals, "+", label="broadcast")
    plt.plot(LOGG, model_chisqr_vals, ".", label="org")
    plt.title("LOGG verse Broadcast chisqr_vals")
    plt.legend()
    plt.show()
    plt.plot(LOGG, broadcast_gamma, "o")
    plt.title("LOGG verse Broadcast gamma grid")
    plt.show()

    plt.plot(FEH, broadcast_chisqr_vals, "+", label="broadcast")
    plt.plot(FEH, model_chisqr_vals, ".", label="org")
    plt.title("FEH vs Broadcast chisqr_vals")
    plt.legend()
    plt.show()
    plt.plot(FEH, broadcast_gamma, "o")
    plt.title("FEH vs Broadcast gamma grid")
    plt.show()

    TEFFS_unique = np.array(set(TEFF))
    LOGG_unique = np.array(set(LOGG))
    FEH_unique = np.array(set(FEH))
    X, Y, Z = np.meshgrid(TEFFS_unique, LOGG_unique, FEH_unique)  # set sparse=True for memory efficency
    print("Teff grid", X)
    print("Logg grid", Y)
    print("FEH grid", Z)
    assert len(TEFF) == sum(len(x) for x in (TEFFS_unique, LOGG_unique, FEH_unique))

    chi_ND = np.empty_like(X.shape)
    print("chi_ND.shape", chi_ND.shape)
    print("len(TEFFS_unique)", len(TEFFS_unique))
    print("len(LOGG_unique)", len(LOGG_unique))
    print("len(FEH_unique)", len(FEH_unique))

    for i, tf in enumerate(TEFFS_unique):
        for j, lg in enumerate(LOGG_unique):
            for k, fh in enumerate(FEH_unique):
                print("i,j,k", (i, j, k))
                print("num = t", np.sum(TEFF == tf))
                print("num = lg", np.sum(LOGG == lg))
                print("num = fh", np.sum(FEH == fh))
                mask = (TEFF == tf) * (LOGG == lg) * (FEH == fh)
                print("num = tf, lg, fh", np.sum(mask))
                chi_ND[i, j, k] = broadcast_chisqr_vals[mask]
                print("broadcast val", broadcast_chisqr_vals[mask],
                      "\norg val", model_chisqr_vals[mask])

    # debug(pv("model_chisqr_vals"))
    # debug(pv("model_xcorr_vals"))
    chisqr_argmin_indx = np.argmin(model_chisqr_vals)
    xcorr_argmax_indx = np.argmax(model_xcorr_vals)

    print("Minimum  Chisqr value =", model_chisqr_vals[chisqr_argmin_indx])  # , min(model_chisqr_vals)
    print("Chisqr at max correlation value", model_chisqr_vals[chisqr_argmin_indx])

    print("model_xcorr_vals = {}".format(model_xcorr_vals))
    print("Maximum Xcorr value =", model_xcorr_vals[xcorr_argmax_indx])  # , max(model_xcorr_vals)
    print("Xcorr at min Chiqsr", model_xcorr_vals[chisqr_argmin_indx])

    # debug(pv("model_xcorr_rv_vals"))
    print("RV at max xcorr =", model_xcorr_rv_vals[xcorr_argmax_indx])
    # print("Meadian RV val =", np.median(model_xcorr_rv_vals))
    print(pv("model_xcorr_rv_vals[chisqr_argmin_indx]"))
    print(pv("sp.stats.mode(np.around(model_xcorr_rv_vals))"))

    # print("Max Correlation model = ", models[xcorr_argmax_indx].split("/")[-2:])
    # print("Min Chisqr model = ", models[chisqr_argmin_indx].split("/")[-2:])
    print("Max Correlation model = ", model_pars[xcorr_argmax_indx])
    print("Min Chisqr model = ", model_pars[chisqr_argmin_indx])

    limits = [2110, 2160]

    best_model_params = model_pars[chisqr_argmin_indx]
    best_model_spec = load_starfish_spectrum(best_model_params, limits=limits, normalize=True)

    best_xcorr_model_params = model_pars[xcorr_argmax_indx]
    best_xcorr_model_spec = load_starfish_spectrum(best_xcorr_model_params, limits=limits, normalize=True)

    close_model_spec = load_starfish_spectrum(closest_model_params, limits=limits, normalize=True)

    plt.plot(obs_spec.xaxis, obs_spec.flux, label="Observations")
    plt.plot(best_model_spec.xaxis, best_model_spec.flux, label="Best Model")
    plt.plot(best_xcorr_model_spec.xaxis, best_xcorr_model_spec.flux, label="Best xcorr Model")
    plt.plot(close_model_spec.xaxis, close_model_spec.flux, label="Close Model")
    plt.legend()
    plt.xlim(*limits)
    plt.show()

    debug("After plot")


def plot_spectra(obs, model):
    """Plot two spectra."""
    plt.plot(obs.xaxis, obs.flux, label="obs")
    plt.plot(model.xaxis, model.flux, label="model")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    def time_func(func, *args, **kwargs):
        start = dt.now()
        print("Starting at: {}".format(start))
        result = func(*args, **kwargs)
        end = dt.now()
        print("Endded at: {}".format(end))
        print("Runtime: {}".format(end - start))
        return result

    sys.exit(time_func(main))
