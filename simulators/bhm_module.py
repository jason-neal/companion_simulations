import copy
import logging

import numpy as np

import pandas as pd
import scipy as sp
import simulators
from matplotlib import pyplot as plt
from models.broadcasted_models import one_comp_model
from tqdm import tqdm
from utilities.chisqr import chi_squared
from utilities.phoenix_utils import load_starfish_spectrum
from utilities.xcorr import xcorr_peak

debug = logging.debug


def bhm_analysis(obs_spec, model_pars, gammas=None, verbose=False, norm=False):
    """Run one component model over all parameter combinations in model_pars."""
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
        save_name = os.path.join(
            simulators.paths["output_dir"], obs_spec.header["OBJECT"],
            "bhm_{0}_{1}_part{2}.csv".format(
                obs_spec.header["OBJECT"], int(obs_spec.header["MJD-OBS"]), ii))
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

        # ### NORMALIZATION NEEDED HERE
        if norm:
            return NotImplemented
            obs_flux = broadcast_normalize_observation(obs_spec.xaxis[:, np.newaxis],
                                                       obs_spec.flux[:, np.newaxis], broadcast_values)
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

        print("broadcast_chisquare.shape", broadcast_chisquare.shape)
        # New parameters to explore
        broadcast_chisqr_vals[ii] = broadcast_chisquare[np.argmin(broadcast_chisquare)]
        broadcast_gamma[ii] = gammas[np.argmin(broadcast_chisquare)]
        full_broadcast_chisquare[ii, :] = broadcast_chisquare

        npix = obs_flux.shape[0]
        save_bhm_chisqr(save_name, params, gammas, broadcast_chisquare, npix)

    return (model_chisqr_vals, model_xcorr_vals, model_xcorr_rv_vals,
            broadcast_chisqr_vals, broadcast_gamma, full_broadcast_chisquare)


def save_bhm_chisqr(name, params1, gammas, broadcast_chisquare, npix):
    """Save the bhm chisqr values to a cvs."""
    assert gammas.shape == broadcast_chisquare.shape
    ravel_size = len(gammas)
    p1_0 = np.ones(ravel_size) * params1[0]
    p1_1 = np.ones(ravel_size) * params1[1]
    p1_2 = np.ones(ravel_size) * params1[2]

    assert p1_2.shape == gammas.shape
    data = {"teff_1": p1_0, "logg_1": p1_1, "feh_1": p1_2, "gamma": gammas, "chi2": broadcast_chisquare.ravel()}
    columns = ["teff_1", "logg_1", "feh_1", "gamma", "npix", "chi2"]
    df = pd.DataFrame(data=data)
    df["npix"] = npix
    df[columns].to_csv(name, sep=',', index=False, mode="a")  # Append to values cvs
    return None


# Doesn't work yet
def broadcast_normalize_observation(wav, obs_flux, broadcast_flux, splits=10):
    """Re-normalize obs_spec to the linear continuum fit along."""
    # Get median values of 10 highest points in the 0.5nm sections of flux

    obs_norm = broadcast_continuum_fit(wav, obs_flux, splits=splits, method="linear", plot=True)
    broad_norm = broadcast_continuum_fit(wav, broadcast_flux, splits=splits, method="linear", plot=True)

    return obs_flux * (broad_norm / obs_norm)


# Doesn't work yet
def broadcast_continuum_fit(wave, flux, splits=50, method="linear", plot=True):
    r"""Continuum fit the N-D - flux array.

    Split spectra into many chunks and get the average of top 5\% in each bin.

    Fit to those points and normalize by that.
    """
    org_flux = copy.copy(flux)
    org_wave = copy.copy(wave)

    while len(wave) % splits != 0:
        # Shorten array until it can be evenly split up.
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
        # Take the median of the wavelength values of max values.
        wav_points[i] = np.median(w[np.argsort(f, axis=0)[-5:]], axis=0, keepdims=True)
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
