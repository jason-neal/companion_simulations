"""two_compoonent_model.py.

Jason Neal
2nd Janurary 2017

Compare observed spectra to many different phoenix-aces spectral models to find which matches one is best fit by itself.

Need to determine the best RV offset to apply to the spectra.
Possibly bu sampling a few and then taking average value (they should be all
the same I would think unless the lines changed dramatically).

"""
from __future__ import division, print_function

import copy
# import itertools
import logging
import os
import sys
from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from astropy.io import fits
from joblib import Parallel, delayed
from tqdm import tqdm

from Chisqr_of_observation import load_spectrum  # , select_observation
from models.broadcasted_models import two_comp_model
# from spectrum_overload.Spectrum import Spectrum
from utilities.chisqr import chi_squared
from utilities.crires_utilities import barycorr_crires_spectrum
from utilities.debug_utils import timeit  # , pv
from utilities.param_file import parse_paramfile
from utilities.phoenix_utils import (closest_model_params,
                                     generate_close_params,
                                     load_starfish_spectrum)

logging.basicConfig(level=logging.WARNING,
                    format='%(levelname)s %(message)s')
debug = logging.debug

model_base_dir = ("/home/jneal/Phd/data/PHOENIX-ALL/PHOENIX/")
wav_dir = "/home/jneal/Phd/data/PHOENIX-ALL/PHOENIX/"

wav_model = fits.getdata(os.path.join(wav_dir, "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"))
wav_model /= 10   # turn into nm


def main():
    """Main function."""
    parallel = True
    star = "HD211847"
    param_file = "/home/jneal/Phd/data/parameter_files/{}_params.dat".format(star)
    params = parse_paramfile(param_file, path=None)
    host_params = [params["temp"], params["logg"], params["fe_h"]]
    comp_params = [params["comp_temp"], params["logg"], params["fe_h"]]

    obs_num = 2
    chip = 4

    obs_name = "/home/jneal/.handy_spectra/{}-{}-mixavg-tellcorr_{}.fits".format(star, obs_num, chip)
    print("The observation used is ", obs_name, "\n")

    closest_host_model = closest_model_params(*host_params)  # unpack temp, logg, fe_h with *
    closest_comp_model = closest_model_params(*comp_params)

    # original_model = "Z-0.0/lte05700-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    # debug(pv("closest_host_model"))
    # debug(pv("closest_comp_model"))
    # debug(pv("original_model"))

    # Function to find the good models I need
    # models = find_phoenix_model_names(model_base_dir, original_model)
    # Function to find the good models I need from parameters
    # model_par_gen = generate_close_params(closest_host_model)
    model1_pars = list(generate_close_params(closest_host_model))  # Turn to list
    model2_pars = list(generate_close_params(closest_comp_model))

    print("Model parameters", model1_pars)
    print("Model parameters", model2_pars)

    # Load observation
    obs_spec = load_spectrum(obs_name)
    obs_spec = barycorr_crires_spectrum(obs_spec, -22)
    obs_spec.flux /= 1.02
    # Mask out bad portion of observed spectra ## HACK
    if chip == 4:
        # Ignore first 40 pixels
        obs_spec.wav_select(obs_spec.xaxis[40], obs_spec.xaxis[-1])

    gammas = np.arange(-20, 20, 1)
    rvs = np.arange(-20, 20, 2)
    alphas = np.arange(0.01, 0.2, 0.02)
    ####
    if parallel:
        chi2_grids = parallel_tcm_analysis(obs_spec, model1_pars, model2_pars, alphas, rvs, gammas, verbose=True)
    else:
        chi2_grids = tcm_analysis(obs_spec, model1_pars, model2_pars, alphas, rvs, gammas, verbose=True)

    ####
    # bcast_chisqr_vals, bcast_alpha, bcast_rv, bcast_gamma, tcm_bcast_chisquare = chi2_grids
    bcast_chisqr_vals = chi2_grids

    print("tcm broadcast_chisquare shape", bcast_chisqr_vals.shape)
    # TEFF = [par[0] for par in model_pars]
    # LOGG = [par[1] for par in model_pars]
    # FEH = [par[2] for par in model_pars]
    #
    # plt.plot(TEFF, broadcast_chisqr_vals, "+", label="broadcast")
    # plt.plot(TEFF, model_chisqr_vals, ".", label="org")
    # plt.title("TEFF vs Broadcast chisqr_vals")
    # plt.legend()
    # plt.show()
    # plt.plot(TEFF, broadcast_gamma, "o")
    # plt.title("TEFF vs Broadcast gamma grid")
    # plt.show()
    #
    # plt.plot(LOGG, broadcast_chisqr_vals, "+", label="broadcast")
    # plt.plot(LOGG, model_chisqr_vals, ".", label="org")
    # plt.title("LOGG verse Broadcast chisqr_vals")
    # plt.legend()
    # plt.show()
    # plt.plot(LOGG, broadcast_gamma, "o")
    # plt.title("LOGG verse Broadcast gamma grid")
    # plt.show()
    #
    # plt.plot(FEH, broadcast_chisqr_vals, "+", label="broadcast")
    # plt.plot(FEH, model_chisqr_vals, ".", label="org")
    # plt.title("FEH vs Broadcast chisqr_vals")
    # plt.legend()
    # plt.show()
    # plt.plot(FEH, broadcast_gamma, "o")
    # plt.title("FEH vs Broadcast gamma grid")
    # plt.show()
    #
    #
    # TEFFS_unique = np.array(set(TEFF))
    # LOGG_unique = np.array(set(LOGG))
    # FEH_unique = np.array(set(FEH))
    # X, Y, Z = np.meshgrid(TEFFS_unique, LOGG_unique, FEH_unique)  # set sparse=True for memory efficency
    # print("Teff grid", X)
    # print("Logg grid", Y)
    # print("FEH grid", Z)
    # assert len(TEFF) == sum(len(x) for x in (TEFFS_unique, LOGG_unique, FEH_unique))
    #
    # chi_ND = np.empty_like(X.shape)
    # print("chi_ND.shape", chi_ND.shape)
    # print("len(TEFFS_unique)", len(TEFFS_unique))
    # print("len(LOGG_unique)", len(LOGG_unique))
    # print("len(FEH_unique)", len(FEH_unique))
    #
    # for i, tf in enumerate(TEFFS_unique):
    #     for j, lg in enumerate(LOGG_unique):
    #         for k, fh in enumerate(FEH_unique):
    #             print("i,j,k", (i, j, k))
    #             print("num = t", np.sum(TEFF == tf))
    #             print("num = lg", np.sum(LOGG == lg))
    #             print("num = fh", np.sum(FEH == fh))
    #             mask = (TEFF == tf) * (LOGG == lg) * (FEH == fh)
    #             print("num = tf, lg, fh", np.sum(mask))
    #             chi_ND[i, j, k] = broadcast_chisqr_vals[mask]
    #             print("broadcast val", broadcast_chisqr_vals[mask],
    #                   "\norg val", model_chisqr_vals[mask])
    #
    #
    # # debug(pv("model_chisqr_vals"))
    # # debug(pv("model_xcorr_vals"))
    # chisqr_argmin_indx = np.argmin(model_chisqr_vals)
    # xcorr_argmax_indx = np.argmax(model_xcorr_vals)
    #
    # # debug(pv("chisqr_argmin_indx"))
    # # debug(pv("xcorr_argmax_indx"))
    #
    # # debug(pv("model_chisqr_vals"))
    # print("Minimum  Chisqr value =", model_chisqr_vals[chisqr_argmin_indx])  # , min(model_chisqr_vals)
    # print("Chisqr at max correlation value", model_chisqr_vals[chisqr_argmin_indx])
    #
    # print("model_xcorr_vals = {}".format(model_xcorr_vals))
    # print("Maximum Xcorr value =", model_xcorr_vals[xcorr_argmax_indx])  # , max(model_xcorr_vals)
    # print("Xcorr at min Chiqsr", model_xcorr_vals[chisqr_argmin_indx])
    #
    # # debug(pv("model_xcorr_rv_vals"))
    # print("RV at max xcorr =", model_xcorr_rv_vals[xcorr_argmax_indx])
    # # print("Meadian RV val =", np.median(model_xcorr_rv_vals))
    # print(pv("model_xcorr_rv_vals[chisqr_argmin_indx]"))
    # print(pv("sp.stats.mode(np.around(model_xcorr_rv_vals))"))
    #
    # # print("Max Correlation model = ", models[xcorr_argmax_indx].split("/")[-2:])
    # # print("Min Chisqr model = ", models[chisqr_argmin_indx].split("/")[-2:])
    # print("Max Correlation model = ", model_pars[xcorr_argmax_indx])
    # print("Min Chisqr model = ", model_pars[chisqr_argmin_indx])
    #
    # limits = [2110, 2160]
    #
    # best_model_params = model_pars[chisqr_argmin_indx]
    # best_model_spec = load_starfish_spectrum(best_model_params, limits=limits, normalize=True)
    #
    # best_xcorr_model_params = model_pars[xcorr_argmax_indx]
    # best_xcorr_model_spec = load_starfish_spectrum(best_xcorr_model_params, limits=limits, normalize=True)
    #
    # close_model_spec = load_starfish_spectrum(closest_model_params, limits=limits, normalize=True)
    #
    #
    # plt.plot(obs_spec.xaxis, obs_spec.flux, label="Observations")
    # plt.plot(best_model_spec.xaxis, best_model_spec.flux, label="Best Model")
    # plt.plot(best_xcorr_model_spec.xaxis, best_xcorr_model_spec.flux, label="Best xcorr Model")
    # plt.plot(close_model_spec.xaxis, close_model_spec.flux, label="Close Model")
    # plt.legend()
    # plt.xlim(*limits)
    # plt.show()


@timeit
def tcm_analysis(obs_spec, model1_pars, model2_pars, alphas=None, rvs=None, gammas=None, verbose=False, norm=False):
    """Run two component model over all parameter cobinations in model1_pars and model2_pars."""
    if alphas is None:
        alphas = np.array([0])
    elif isinstance(alphas, (float, int)):
        alphas = np.asarray(alphas, dtype=np.float32)
    if rvs is None:
        rvs = np.array([0])
    elif isinstance(rvs, (float, int)):
        rvs = np.asarray(rvs, dtype=np.float32)
    if gammas is None:
        gammas = np.array([0])
    elif isinstance(gammas, (float, int)):
        gammas = np.asarray(gammas, dtype=np.float32)

    if isinstance(model1_pars, list):
        debug("Number of close model_pars returned {}".format(len(model1_pars)))
    if isinstance(model2_pars, list):
        debug("Number of close model_pars returned {}".format(len(model2_pars)))

    print("host params", model1_pars)
    print("companion params", model2_pars)

    # Solution Grids to return
    # model_chisqr_vals = np.empty((len(model1_pars), len(model2_pars)))
    # model_xcorr_vals = np.empty(len(model1_pars), len(model2_pars))
    # model_xcorr_rv_vals = np.empty(len(model1_pars), len(model2_pars))
    broadcast_chisqr_vals = np.empty((len(model1_pars), len(model2_pars)))
    # broadcast_gamma = np.empty((len(model1_pars), len(model2_pars)))
    # full_broadcast_chisquare = np.empty((len(model1_pars), len(model2_pars), len(alphas), len(rvs), len(gammas)))

    normalization_limits = [2105, 2185]   # small as possible?
    # combined_params = itertools.product(model1_pars, model2_pars)
    for ii, params1 in enumerate(tqdm(model1_pars)):
        save_filename = ("Analysis/{0}/tc_{0}_{1}_part{5}_host_pars_{2}_{3}_{4}"
                         ".csv").format(obs_spec.header["OBJECT"],
                                        int(obs_spec.header["MJD-OBS"]),
                                        params1[0], params1[1], params1[2], ii)
        for jj, params2 in enumerate(model2_pars):

            if verbose:
                print("Starting iteration with parameters:\n{0}={1},{2}={3}".format(ii, params1, jj, params2))
            mod1_spec = load_starfish_spectrum(params1, limits=normalization_limits, hdr=True, normalize=True)
            mod2_spec = load_starfish_spectrum(params2, limits=normalization_limits, hdr=True, normalize=True)

            # TODO WHAT IS THE MAXIMUM (GAMMA + RV POSSIBLE? LIMIT IT TO THAT SHIFT?

            # Wavelength selection
            mod1_spec.wav_select(np.min(obs_spec.xaxis) - 5,
                                 np.max(obs_spec.xaxis) + 5)  # +- 5nm of obs for convolution
            mod2_spec.wav_select(np.min(obs_spec.xaxis) - 5,
                                 np.max(obs_spec.xaxis) + 5)
            obs_spec = obs_spec.remove_nans()

            # One component model with broadcasting over gammas
            # two_comp_model(wav, model1, model2, alphas, rvs, gammas)
            assert np.allclose(mod1_spec.xaxis, mod2_spec.xaxis)

            broadcast_result = two_comp_model(mod1_spec.xaxis, mod1_spec.flux, mod2_spec.flux,
                                              alphas=alphas, rvs=rvs, gammas=gammas)
            broadcast_values = broadcast_result(obs_spec.xaxis)

            assert ~np.any(np.isnan(obs_spec.flux)), "Observation is nan"

            # ### NORMALIZATION NEEDED HERE
            if norm:
                return NotImplemented
                obs_flux = broadcast_normalize_observation(
                    obs_spec.xaxis[:, np.newaxis, np.newaxis, np.newaxis],
                    obs_spec.flux[:, np.newaxis, np.newaxis, np.newaxis],
                    broadcast_values)
            else:
                obs_flux = obs_spec.flux[:, np.newaxis, np.newaxis, np.newaxis]
            #####

            broadcast_chisquare = chi_squared(obs_flux, broadcast_values)
            sp_chisquare = sp.stats.chisquare(obs_flux, broadcast_values, axis=0).statistic

            assert np.all(sp_chisquare == broadcast_chisquare)

            print(broadcast_chisquare.shape)
            print(broadcast_chisquare.ravel()[np.argmin(broadcast_chisquare)])
            # New parameters to explore
            broadcast_chisqr_vals[ii, jj] = broadcast_chisquare.ravel()[np.argmin(broadcast_chisquare)]
            # broadcast_gamma[ii, jj] = gammas[np.argmin(broadcast_chisquare)]
            # full_broadcast_chisquare[ii, jj, :] = broadcast_chisquare

            save_full_chisqr(save_filename, params1, params2, alphas, rvs, gammas, broadcast_chisquare)

    return broadcast_chisqr_vals   # Just output the best value for each model pair


def parallel_tcm_analysis(obs_spec, model1_pars, model2_pars, alphas=None,
                          rvs=None, gammas=None, verbose=False, norm=False):
    """Run two component model over all parameter cobinations in model1_pars and model2_pars."""
    if alphas is None:
        alphas = np.array([0])
    elif isinstance(alphas, (float, int)):
        alphas = np.asarray(alphas, dtype=np.float32)
    if rvs is None:
        rvs = np.array([0])
    elif isinstance(rvs, (float, int)):
        rvs = np.asarray(rvs, dtype=np.float32)
    if gammas is None:
        gammas = np.array([0])
    elif isinstance(gammas, (float, int)):
        gammas = np.asarray(gammas, dtype=np.float32)

    if isinstance(model1_pars, list):
        debug("Number of close model_pars returned {}".format(len(model1_pars)))
    if isinstance(model2_pars, list):
        debug("Number of close model_pars returned {}".format(len(model2_pars)))

    print("host params", model1_pars)
    print("companion params", model2_pars)

    # Solution Grids to return
    # model_chisqr_vals = np.empty((len(model1_pars), len(model2_pars)))
    # model_xcorr_vals = np.empty(len(model1_pars), len(model2_pars))
    # model_xcorr_rv_vals = np.empty(len(model1_pars), len(model2_pars))
    # broadcast_chisqr_vals = np.empty((len(model1_pars), len(model2_pars)))
    # broadcast_gamma = np.empty((len(model1_pars), len(model2_pars)))
    # full_broadcast_chisquare = np.empty((len(model1_pars), len(model2_pars), len(alphas), len(rvs), len(gammas)))

    print("parallised running\n\n\n ###################")
    broadcast_chisqr_vals = Parallel(n_jobs=3)(
        delayed(tcm_wrapper)(ii, param, model2_pars, alphas,
                             rvs, gammas, obs_spec, norm=False)
        for ii, param in enumerate(model1_pars))
    # for ii, params1 in enumerate(tqdm(model1_pars)):

    return broadcast_chisqr_vals   # Just output the best value for each model pair


def tcm_wrapper(num, params1, model2_pars, alphas, rvs, gammas, obs_spec, norm=True, verbose=True):
    """Wrapper for iteration loop of tcm. To use with parallization."""
    save_filename = ("Analysis/{0}/tc_{0}_{1}_part{5}_host_pars_{2}_{3}_{4}"
                     ".csv").format(obs_spec.header["OBJECT"],
                                    int(obs_spec.header["MJD-OBS"]),
                                    params1[0], params1[1], params1[2], num)

    broadcast_chisqr_vals = np.empty(len(model2_pars))
    for jj, params2 in enumerate(model2_pars):

        if verbose:
            print("Starting iteration with parameters:\n{0}={1},{2}={3}".format(num, params1, jj, params2))

        normalization_limits = [2105, 2185]   # small as possible?
        mod1_spec = load_starfish_spectrum(params1, limits=normalization_limits, hdr=True, normalize=True)
        mod2_spec = load_starfish_spectrum(params2, limits=normalization_limits, hdr=True, normalize=True)

        # TODO WHAT IS THE MAXIMUM (GAMMA + RV POSSIBLE? LIMIT IT TO THAT SHIFT?

        # Wavelength selection
        mod1_spec.wav_select(np.min(obs_spec.xaxis) - 5,
                             np.max(obs_spec.xaxis) + 5)  # +- 5nm of obs for convolution
        mod2_spec.wav_select(np.min(obs_spec.xaxis) - 5,
                             np.max(obs_spec.xaxis) + 5)
        obs_spec = obs_spec.remove_nans()

        # One component model with broadcasting over gammas
        # two_comp_model(wav, model1, model2, alphas, rvs, gammas)
        assert np.allclose(mod1_spec.xaxis, mod2_spec.xaxis)

        broadcast_result = two_comp_model(mod1_spec.xaxis, mod1_spec.flux, mod2_spec.flux,
                                          alphas=alphas, rvs=rvs, gammas=gammas)
        broadcast_values = broadcast_result(obs_spec.xaxis)

        assert ~np.any(np.isnan(obs_spec.flux)), "Observation is nan"

        # ### NORMALIZATION NEEDED HERE
        if norm:
            return NotImplemented
            obs_flux = broadcast_normalize_observation(
                obs_spec.xaxis[:, np.newaxis, np.newaxis, np.newaxis],
                obs_spec.flux[:, np.newaxis, np.newaxis, np.newaxis],
                broadcast_values)
        else:
            obs_flux = obs_spec.flux[:, np.newaxis, np.newaxis, np.newaxis]
        #####

        broadcast_chisquare = chi_squared(obs_flux, broadcast_values)
        sp_chisquare = sp.stats.chisquare(obs_flux, broadcast_values, axis=0).statistic

        assert np.all(sp_chisquare == broadcast_chisquare)

        print(broadcast_chisquare.shape)
        print(broadcast_chisquare.ravel()[np.argmin(broadcast_chisquare)])

        # New parameters to explore
        broadcast_chisqr_vals[jj] = broadcast_chisquare.ravel()[np.argmin(broadcast_chisquare)]

        save_full_chisqr(save_filename, params1, params2, alphas, rvs, gammas, broadcast_chisquare)

    return broadcast_chisqr_vals


# @timeit
def save_full_chisqr(name, params1, params2, alphas, rvs, gammas, broadcast_chisquare):
    """Save the iterations chisqr values to a cvs."""
    A, R, G = np.meshgrid(alphas, rvs, gammas, indexing='ij')
    assert A.shape == R.shape
    assert R.shape == G.shape
    assert G.shape == broadcast_chisquare.shape
    ravel_size = len(A.ravel())
    p1_0 = np.ones(ravel_size) * params1[0]
    p1_1 = np.ones(ravel_size) * params1[1]
    p1_2 = np.ones(ravel_size) * params1[2]
    p2_0 = np.ones(ravel_size) * params1[0]
    p2_1 = np.ones(ravel_size) * params1[1]
    p2_2 = np.ones(ravel_size) * params1[2]
    assert p2_2.shape == A.ravel().shape
    data = {"teff_1": p1_0, "logg_1": p1_1, "feh_1": p1_2, "teff_2": p2_0, "logg_2": p2_1, "feh_2": p2_2,
            "alpha": A.ravel(), "rv": R.ravel(), "gamma": G.ravel(), "chi2": broadcast_chisquare.ravel()}
    columns = ["teff_1", "logg_1", "feh_1", "teff_2", "logg_2", "feh_2",
               "alpha", "rv", "gamma", "chi2"]
    df = pd.DataFrame(data=data)
    df[columns].to_csv(name, sep=',', index=False, mode="a")  # Append to values cvs
    return None


def plot_spectra(obs, model):
    """Plot two spectra."""
    plt.plot(obs.xaxis, obs.flux, label="obs")
    plt.plot(model.xaxis, model.flux, label="model")
    plt.legend()
    plt.show()


def broadcast_normalize_observation(wav, obs_flux, broadcast_flux, splits=10):
    """Renormalize obs_spec to the linear continum fit along."""
    # Get median values of 10 highest points in the 0.5nm sections of flux

    obs_norm = broadcast_continuum_fit(wav, obs_flux, splits=splits, method="linear", plot=True)
    broad_norm = broadcast_continuum_fit(wav, broadcast_flux, splits=splits, method="linear", plot=True)

    return obs_flux * (broad_norm / obs_norm)


def broadcast_continuum_fit(wave, flux, splits=50, method="linear", plot=True):
    r"""Continuum fit the N-D - flux array.

    Split spectra into many chunks and get the average of top 5\% in each bin.

    Fit to those points and normalize by that.
    """
    org_flux = copy.copy(flux)
    org_wave = copy.copy(wave)

    while len(wave) % splits != 0:
        # Shorten array untill can be evenly split up.
        wave = wave[:-1]
        flux = flux[:-1]
        print(wave.shape)
        print(flux.shape)
    if flux.ndim > wave.ndim:
        wave = wave * np.ones_like(flux)  # Broadcast it out

    wav_split = np.vsplit(wave, splits)
    flux_split = np.vsplit(flux, splits)  # split along axis=0
    print(type(wav_split), type(flux_split))
    print("wav shape", wave.shape)
    print("wav split shape", len(wav_split))
    print("flux shape", flux.shape)
    print("flux split shape", len(flux_split))
    print("wav split[0] shape", wav_split[0].shape)
    print("flux split[0] shape", flux_split[0].shape)

    # TODO!
    flux_split_medians = []
    wave_split_medians = []
    wav_points = np.empty_like(splits)
    print(wav_points.shape)
    flux_points = np.empty(splits)
    f = flux_split
    print("argsort", np.argsort(f[0], axis=0))
    print("f[argsort]", f[np.argsort(f[0], axis=0)])
    print(np.median(f[np.argsort(f[0], axis=0)]))
    for i, (w, f) in enumerate(zip(wav_split, flux_split)):
        wav_points[i] = np.median(w[np.argsort(f, axis=0)[-5:]],
                                  axis=0, keepdims=True)  # Take the median of the wavelength values of max values.
        flux_points[i, ] = np.median(f[np.argsort(f, axis=0)[-5:]], axis=0, keepdims=True)

    print("flux_points", flux_points)
    print("flux_points.shape", flux_points.shape)
    print("flux_points[0].shape", flux_points[0].shape)

    if method == "scalar":
        norm_flux = np.median(flux_split) * np.ones_like(org_wave)
    elif method == "linear":
        z = np.polyfit(wav_points, flux_points, 1)
        p = np.poly1d(z)
        norm_flux = p(org_wave)
    elif method == "quadratic":
        z = np.polyfit(wav_points, flux_points, 2)
        p = np.poly1d(z)
        norm_flux = p(org_wave)
    elif method == "exponential":
        z = np.polyfit(wav_points, np.log(flux_points), deg=1, w=np.sqrt(flux_points))
        p = np.poly1d(z)
        norm_flux = np.exp(p(org_wave))   # Un-log the y values.

    if plot:
        plt.subplot(211)
        plt.plot(wave, flux)
        plt.plot(wav_points, flux_points, "x-", label="points")
        plt.plot(org_wave, norm_flux, label='norm_flux')
        plt.legend()
        plt.subplot(212)
        plt.plot(org_wave, org_flux / norm_flux)
        plt.title("Normalization")
        plt.show()

    return org_flux / norm_flux


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
