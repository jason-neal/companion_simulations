"""best_host_model.py

Jason Neal
2nd Janurary 2017

Compare observed spectra to many different phoenix-aces spectral models to find which matches one is best fit by itself.

Need to determine the best RV offset to apply to the spectra.
Possibly bu sampling a few and then taking average value (they should be all
the same I would think unless the lines changed dramatically).

"""
from __future__ import division, print_function
import os
import sys
import logging
import numpy as np
import scipy as sp
from tqdm import tqdm
from astropy.io import fits
import matplotlib.pyplot as plt
from datetime import datetime as dt
from utilities.debug_utils import pv
from utilities.chisqr import chi_squared
from spectrum_overload.Spectrum import Spectrum
import copy
from utilities.crires_utilities import crires_resolution, barycorr_crires_spectrum
from Chisqr_of_observation import select_observation, load_spectrum
from utilities.param_file import parse_paramfile
from utilities.phoenix_utils import closest_model_params, generate_close_params, load_starfish_spectrum
from models.broadcasted_models import one_comp_model

logging.basicConfig(level=logging.WARNING,
                    format='%(levelname)s %(message)s')
debug = logging.debug

model_base_dir = ("/home/jneal/Phd/data/PHOENIX-ALL/PHOENIX/")
wav_dir = "/home/jneal/Phd/data/PHOENIX-ALL/PHOENIX/"

wav_model = fits.getdata(os.path.join(wav_dir, "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"))
wav_model /= 10   # turn into nm


def xcorr_peak(spectrum, model, plot=False):
    """Find RV offset between a spectrum and a model using pyastronomy.

    Parameters
    ----------
    spectrum: Spectrum
       Target Spectrum object.
    model: Spectrum
        Template Specturm object.

    Returns
    -------
    rv_max: float
        Radial velocity vlaue corresponding to maximum correlation.
    cc_max: float
        Cross-correlation value corresponding to maximum correlation.
    """
    rv, cc = spectrum.crosscorrRV(model, rvmin=-60., rvmax=60.0, drv=0.1,
                                  mode='doppler', skipedge=50)  # Spectrum method

    maxind = np.argmax(cc)
    rv_max, cc_max = rv[maxind], cc[maxind]

    # debug("Cross-correlation function is maximized at dRV = {} km/s".format(rv_max))

    if plot:
        plt.subplot(211)
        plt.plot(spectrum.xaxis, spectrum.flux, label="Target")
        plt.plot(model.xaxis, model.flux, label="Model")
        plt.legend()
        plt.title("Spectra")

        plt.subplot(212)
        plt.plot(rv, cc)
        plt.plot(rv_max, cc_max, "o")
        plt.title("Cross correlation plot")
        plt.show()
    return float(rv[maxind]), float(cc[maxind])


def main():
    """Main function."""
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

    original_model = "Z-0.0/lte05700-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    # debug(pv("closest_host_model"))
    # debug(pv("closest_comp_model"))
    # debug(pv("original_model"))

    # Function to find the good models I need
    # models = find_phoenix_model_names(model_base_dir, original_model)
    # Function to find the good models I need from parameters
    # model_par_gen = generate_close_params(closest_host_model)
    model_pars = list(generate_close_params(closest_host_model))  # Turn to list

    # print(model_pars)



    print("Model parameters", model_pars)

    # Load observation
    obs_spec = load_spectrum(obs_name)
    obs_spec = barycorr_crires_spectrum(obs_spec, -22)
    obs_spec.flux /= 1.02
    # Mask out bad portion of observed spectra ## HACK
    if chip == 4:
        # Ignore first 40 pixels
        obs_spec.wav_select(obs_spec.xaxis[40], obs_spec.xaxis[-1])

    gammas = np.arange(-20, 20, 1)

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

    # debug(pv("chisqr_argmin_indx"))
    # debug(pv("xcorr_argmax_indx"))

    # debug(pv("model_chisqr_vals"))
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


def bhm_analysis(obs_spec, model_pars, gammas=None, verbose=False, norm=False):
    """Run one component model over all parameter cobinations in model_pars."""
    # Gammas
    if gammas is None:
        gammas = np.array([0])
    elif isinstance(gammas, (float, int)):
        gammas = np.asarray(gammas, dtype=np.float32)

    if isinstance(model_pars, list):
        debug("Number of close model_pars returned {}".format(len(model_pars)))

    print(model_pars)
    # Solution Grids to return
    model_chisqr_vals = np.empty(len(model_pars))
    model_xcorr_vals = np.empty(len(model_pars))
    model_xcorr_rv_vals = np.empty(len(model_pars))
    broadcast_chisqr_vals = np.empty(len(model_pars))
    broadcast_gamma = np.empty(len(model_pars))
    full_broadcast_chisquare = np.empty((len(model_pars), len(gammas)))

    normalization_limits = [2105, 2185]   # small as possible?
    for ii, params in enumerate(tqdm(model_pars)):
        if verbose:
            print("Starting iteration with parameter:s\n{}".format(params))
        mod_spec = load_starfish_spectrum(params, limits=normalization_limits, hdr=True, normalize=True)

        # Wavelength selection
        mod_spec.wav_select(np.min(obs_spec.xaxis) - 5,
                            np.max(obs_spec.xaxis) + 5)  # +- 5nm of obs for convolution

        # Find cross correlation RV
        # Should run though all models and find best rv to apply uniformly
        rvoffset, cc_max = xcorr_peak(obs_spec, mod_spec, plot=False)
        if verbose:
            print("Cross correlation RV = {}".format(rvoffset))
            print("Cross correlation max = {}".format(cc_max))

        obs_spec = obs_spec.remove_nans()

        # One component model with broadcasting over gammas
        broadcast_result = one_comp_model(mod_spec.xaxis, mod_spec.flux, gammas=gammas)
        broadcast_values = broadcast_result(obs_spec.xaxis)

        assert ~np.any(np.isnan(obs_spec.flux)), "Observation is nan"

        #### NORMALIZATION NEEDED HERE
        if norm:
            return NotImplemented
            obs_flux = broadcast_normalize_observation(obs_spec.xaxis[:, np.newaxis], obs_spec.flux[:, np.newaxis], broadcast_values)
        else:
            obs_flux = obs_spec.flux[:, np.newaxis]
        #####

        broadcast_chisquare = chi_squared(obs_flux, broadcast_values)
        sp_chisquare = sp.stats.chisquare(obs_flux, broadcast_values, axis=0).statistic

        assert np.all(sp_chisquare == broadcast_chisquare)

        # Interpolate to obs
        mod_spec.spline_interpolate_to(obs_spec)
        # conv_mod_spec.interpolate1d_to(obs_spec)
        model_chi_val = chi_squared(obs_spec.flux, mod_spec.flux)

        # argmax = np.argmax(cc_max)
        model_chisqr_vals[ii] = model_chi_val
        model_xcorr_vals[ii] = cc_max
        model_xcorr_rv_vals[ii] = rvoffset

        # New parameters to explore
        broadcast_chisqr_vals[ii] = broadcast_chisquare[np.argmin(broadcast_chisquare)]
        broadcast_gamma[ii] = gammas[np.argmin(broadcast_chisquare)]
        full_broadcast_chisquare[ii, :] = broadcast_chisquare

    return model_chisqr_vals, model_xcorr_vals, model_xcorr_rv_vals, broadcast_chisqr_vals, broadcast_gamma, full_broadcast_chisquare


def plot_spectra(obs, model):
    """Plot two spectra."""
    plt.plot(obs.xaxis, obs.flux, label="obs")
    plt.plot(model.xaxis, model.flux, label="model")
    plt.legend()
    plt.show()


def broadcast_normalize_observation(wav, obs_flux, broadcast_flux, splits=10):
    """ Renormalize obs_spec to the linear continum fit along."""

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
        wav_points[i] = np.median(w[np.argsort(f, axis=0)[-5:]], axis=0, keepdims=True)  # Take the median of the wavelength values of max values.
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
