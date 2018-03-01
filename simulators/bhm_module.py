import logging
import os

import numpy as np
import pandas as pd
from logutils import BraceMessage as __
from tqdm import tqdm

import simulators
from mingle.models.broadcasted_models import one_comp_model
from mingle.utilities.chisqr import chi_squared
from mingle.utilities.phoenix_utils import generate_bhm_config_params
from mingle.utilities.phoenix_utils import load_starfish_spectrum, closest_model_params, generate_close_params
from mingle.utilities.xcorr import xcorr_peak
from simulators.common_setup import setup_dirs, sim_helper_function
from simulators.iam_module import arbitrary_minimums, arbitrary_rescale
from simulators.iam_module import renormalization

from numpy import float64, int64, ndarray
from typing import Dict, List, Optional, Tuple, Union


def setup_bhm_dirs(star: str) -> None:
    setup_dirs(star, mode="bhm")
    return None


def bhm_analysis(obs_spec, model_pars, gammas=None, errors=None, prefix=None, verbose=False, chip=None, norm=False,
                 wav_scale=True, norm_method="scalar"):
    """Run one component model over all parameter combinations in model_pars."""
    # Gammas
    if gammas is None:
        gammas = np.array([0])
    elif isinstance(gammas, (float, int)):
        gammas = np.asarray(gammas, dtype=np.float32)

    if isinstance(model_pars, list):
        logging.debug(__("Number of close model_pars returned {0}", len(model_pars)))

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

        # RENORMALIZATION
        if chip == 4:
            # Quadratically renormalize anyway
            obs_spec = renormalization(obs_spec, bhm_grid_values, normalize=True, method="quadratic")
        obs_flux = renormalization(obs_spec, bhm_grid_values, normalize=norm, method=norm_method)

        # Simple chi2
        bhm_grid_chisquare_old = chi_squared(obs_flux, bhm_grid_values, error=errors)

        # Applying arbitrary scalar normalization to continuum
        bhm_norm_grid_values, arb_norm = arbitrary_rescale(bhm_grid_values,
                                                           *simulators.sim_grid["arb_norm"])

        # Calculate Chi-squared
        obs_flux = np.expand_dims(obs_flux, -1)  # expand on last axis to match rescale
        bhm_norm_grid_chisquare = chi_squared(obs_flux, bhm_norm_grid_values, error=errors)

        # Take minimum chi-squared value along Arbitrary normalization axis
        bhm_grid_chisquare, arbitrary_norms = arbitrary_minimums(bhm_norm_grid_chisquare, arb_norm)

        assert np.any(
            bhm_grid_chisquare_old >= bhm_grid_chisquare), "All chi2 values are not better or same with arbitrary scaling"

        # Interpolate to obs
        mod_spec.spline_interpolate_to(obs_spec)
        org_model_chi_val = chi_squared(obs_spec.flux, mod_spec.flux)

        model_chisqr_vals[ii] = org_model_chi_val  # This is gamma = 0 version

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



def bhm_magic_sauce(obs_spec, params, rv1, chip=None,
                    wav_scale=True, norm=False,
                    norm_method="scalar"):
    """Main guts of bhm"""

    normalization_limits = [2105, 2185]  # small as possible?

    mod_spec = load_starfish_spectrum(params, limits=normalization_limits, hdr=True,
                                      normalize=True, wav_scale=wav_scale)

    # Wavelength selection
    mod_spec.wav_select(np.min(obs_spec.xaxis) - 5,
                        np.max(obs_spec.xaxis) + 5)  # +- 5nm of obs

    obs_spec = obs_spec.remove_nans()

    # One component model with broadcasting over gammas
    bhm_grid_func = one_comp_model(mod_spec.xaxis, mod_spec.flux, gammas=rv1)
    bhm_grid_values = bhm_grid_func(obs_spec.xaxis)

    assert ~np.any(np.isnan(obs_spec.flux)), "Observation is nan"

    # RENORMALIZATION
    if chip == 4:
        # Quadratically re-normalize anyway
        obs_spec = renormalization(obs_spec, bhm_grid_values, normalize=True, method="quadratic")
    obs_flux = renormalization(obs_spec, bhm_grid_values, normalize=norm, method=norm_method)


    assert obs_flux.shape() == bhm_grid_values.shape()
    return obs_flux, bhm_grid_values


def save_full_bhm_chisqr(name: str, params1: List[Union[int, float]], gammas: ndarray, bhm_grid_chisquare: ndarray,
                         arbitrary_norms: ndarray, npix: int, xcorr_value: Optional[int] = None) -> None:
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


def bhm_helper_function(star: str, obsnum: Union[int, str], chip: int, skip_params: bool = False) -> Tuple[
    str,  Dict[str, Union[str, float, List[Union[str, float]]]], str]:
    return sim_helper_function(star, obsnum, chip, skip_params=skip_params, mode="bhm")


def get_bhm_model_pars(params: Dict[str, Union[int, float]], method: str = "close") -> List[
    List[Union[int64, float64]]]:
    method = method.lower()

    host_params = [params["temp"], params["logg"], params["fe_h"]]
    closest_host_model = closest_model_params(*host_params)
    if method == "config":
        model_pars = list(generate_bhm_config_params(closest_host_model))
    elif method == "close":
        # Model parameters to try iterate over.
        model_pars = list(generate_close_params(closest_host_model, small=True))
    else:
        raise ValueError("The method '{0}' is not valid".format(method))

    return model_pars
