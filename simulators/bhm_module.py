import copy
import logging
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

import simulators
from mingle.models.broadcasted_models import one_comp_model
from mingle.utilities import parse_paramfile
from mingle.utilities.chisqr import chi_squared
from mingle.utilities.norm import chi2_model_norms
from mingle.utilities.phoenix_utils import load_starfish_spectrum, closest_model_params, generate_close_params
from mingle.utilities.xcorr import xcorr_peak


def setup_dirs(star, mode="iam"):
    mode = mode.lower()
    assert mode in ["iam", "tcm", "bhm"]

    basedir = os.path.join(simulators.paths["output_dir"], star.upper(), mode)
    os.makedirs(basedir, exist_ok=True)
    os.makedirs(os.path.join(basedir, "plots"), exist_ok=True)
    return basedir


def setup_bhm_dirs(star):
    setup_dirs(star, mode="bhm")
    return None

from simulators.iam_module import arbitrary_minimums, arbitrary_rescale

def bhm_analysis(obs_spec, model_pars, gammas=None, errors=None, prefix=None, verbose=False, chip=None, norm=False,
                 wav_scale=True):
    """Run one component model over all parameter combinations in model_pars."""
    # Gammas
    if gammas is None:
        gammas = np.array([0])
    elif isinstance(gammas, (float, int)):
        gammas = np.asarray(gammas, dtype=np.float32)

    if isinstance(model_pars, list):
        logging.debug("Number of close model_pars returned {}".format(len(model_pars)))

    # Solution Grids to return
    model_chisqr_vals = np.empty(len(model_pars))
    model_xcorr_vals = np.empty(len(model_pars))
    model_xcorr_rv_vals = np.empty(len(model_pars))
    bhm_grid_chisqr_vals = np.empty(len(model_pars))
    bhm_grid_gamma = np.empty(len(model_pars))
    full_bhm_grid_chisquare = np.empty((len(model_pars), len(gammas)))

    normalization_limits = [2105, 2185]  # small as possible?

    for ii, params in enumerate(tqdm(model_pars)):
        if prefix is None:
            save_name = os.path.join(
                simulators.paths["output_dir"], obs_spec.header["OBJECT"].upper(), "bhm",
                "bhm_{0}_{1}_{3}_part{2}.csv".format(
                    obs_spec.header["OBJECT"].upper(), obs_spec.header["MJD-OBS"], ii, chip))
        else:
            save_name = os.path.join("{0}_part{1}.csv".format(prefix, ii))

        if verbose:
            print("Starting iteration with parameter:s\n{}".format(params))

        mod_spec = load_starfish_spectrum(params, limits=normalization_limits, hdr=True,
                                          normalize=True, wav_scale=wav_scale)

        # Wavelength selection
        mod_spec.wav_select(np.min(obs_spec.xaxis) - 5,
                            np.max(obs_spec.xaxis) + 5)  # +- 5nm of obs

        obs_spec = obs_spec.remove_nans()

        # One component model with broadcasting over gammas
        bhm_grid_func = one_comp_model(mod_spec.xaxis, mod_spec.flux, gammas=gammas)
        bhm_grid_values = bhm_grid_func(obs_spec.xaxis)

        assert ~np.any(np.isnan(obs_spec.flux)), "Observation is nan"

        # ### NORMALIZATION NEEDED HERE
        if norm:
            # return NotImplemented
            # obs_flux = broadcast_normalize_observation(obs_spec.xaxis[:, np.newaxis],
            #                                            obs_spec.flux[:, np.newaxis], bhm_grid_values)
            obs_flux = chi2_model_norms(obs_spec.xaxis, obs_spec.flux, bhm_grid_values)
        else:
            obs_flux = obs_spec.flux[:, np.newaxis]
        #####
        # Simple chi2
        bhm_grid_chisquare_old = chi_squared(obs_flux, bhm_grid_values, error=errors)

        ### Applying arbitrary scalar normalization to continuum
        bhm_norm_grid_values, arb_norm = arbitrary_rescale(bhm_grid_values,
                                                      *simulators.sim_grid["arb_norm"])
       # print("Arbitrary Normalized bhm_grid_values shape.", bhm_norm_grid_values.shape)

        # Calculate Chi-squared
        obs_flux = np.expand_dims(obs_flux, -1)  # expand on last axis to match rescale
        bhm_norm_grid_chisquare = chi_squared(obs_flux, bhm_norm_grid_values, error=errors)

        #print("Broadcast chi-squared values with arb norm", bhm_norm_grid_chisquare.shape)

        # Take minimum chi-squared value along Arbitrary normalization axis
        bhm_grid_chisquare, arbitrary_norms = arbitrary_minimums(bhm_norm_grid_chisquare, arb_norm)
        #print("Broadcast chi-squared values ", bhm_grid_chisquare.shape)


        assert np.all(bhm_grid_chisquare_old >= bhm_grid_chisquare), "All chi2 values are not better or same with arbitary scaling"

        # Interpolate to obs
        mod_spec.spline_interpolate_to(obs_spec)
        org_model_chi_val = chi_squared(obs_spec.flux, mod_spec.flux)

        model_chisqr_vals[ii] = org_model_chi_val   # This is gamma = 0 version

        # New parameters to explore
        bhm_grid_chisqr_vals[ii] = bhm_grid_chisquare[np.argmin(bhm_grid_chisquare)]
        bhm_grid_gamma[ii] = gammas[np.argmin(bhm_grid_chisquare)]
        full_bhm_grid_chisquare[ii, :] = bhm_grid_chisquare


        ################
        #  Find cross correlation RV
        # Should run though all models and find best rv to apply uniformly
        rvoffset, cc_max = xcorr_peak(obs_spec, mod_spec, plot=False)
        if verbose:
            print("Cross correlation RV = {}".format(rvoffset))
            print("Cross correlation max = {}".format(cc_max))

        model_xcorr_vals[ii] = cc_max
        model_xcorr_rv_vals[ii] = rvoffset
        ###################

        npix = obs_flux.shape[0]
        # print("bhm shape", bhm_grid_chisquare.shape)
        save_full_bhm_chisqr(save_name, params, gammas, bhm_grid_chisquare, arbitrary_norms,
                             npix, rvoffset)

    return (model_chisqr_vals, model_xcorr_vals, model_xcorr_rv_vals,
            bhm_grid_chisqr_vals, bhm_grid_gamma, full_bhm_grid_chisquare)


def save_full_bhm_chisqr(name, params1, gammas, bhm_grid_chisquare,
                         arbitrary_norms, npix, xcorr_value=None):
    """Save the bhm chisqr values to a cvs."""
    assert gammas.shape == bhm_grid_chisquare.shape

    data = {"gamma": gammas, "chi2": bhm_grid_chisquare.ravel(), "arbnorm": arbitrary_norms.ravel()}
    df = pd.DataFrame(data=data)
    df["teff_1"] = params1[0]
    df["logg_1"] = params1[1]
    df["feh_1"] = params1[2]
    df["npix"] = npix
    if xcorr_value is None:
        xcorr_value = -9999999
    df["xcorr"] = xcorr_value
    columns = ["teff_1", "logg_1", "feh_1", "gamma", "npix", "chi2", "arbnorm", "xcorr"]
    df[columns].to_csv(name, sep=',', index=False, mode="a")  # Append to values cvs
    return None


def sim_helper_function(star, obsnum, chip, skip_params, mode="iam"):
    mode = mode.lower()
    if mode not in ["iam", "tcm", "bhm"]:
        raise ValueError(f"Mode {mode} for sim_helper_function not in 'iam, tcm, bhm'")
    if not skip_params:
        param_file = os.path.join(simulators.paths["parameters"], "{}_params.dat".format(star))
        params = parse_paramfile(param_file, path=None)
    else:
        params = {}
    obs_name = os.path.join(
        simulators.paths["spectra"], "{0}-{1}-mixavg-tellcorr_{2}.fits".format(star, obsnum, chip))

    output_prefix = os.path.join(
        simulators.paths["output_dir"], star.upper(), mode,
        "{0}-{1}_{2}_{3}_chisqr_results".format(star.upper(), obsnum, chip, mode))
    return obs_name, params, output_prefix


def bhm_helper_function(star, obsnum, chip, skip_params=False):
    return sim_helper_function(star, obsnum, chip, skip_params=skip_params, mode="bhm")


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
