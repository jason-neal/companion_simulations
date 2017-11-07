import logging
import os

import numpy as np

import pandas as pd
import simulators
from joblib import Parallel, delayed
from models.broadcasted_models import inherent_alpha_model
from scipy import stats
from tqdm import tqdm

from utilities.chisqr import chi_squared
from utilities.norm import chi2_model_norms, continuum
from utilities.param_file import parse_paramfile
from utilities.phoenix_utils import load_starfish_spectrum
from utilities.simulation_utilities import check_inputs, spec_max_delta

debug = logging.debug


def iam_helper_function(star, obs_num, chip):
    param_file = os.path.join(simulators.paths["parameters"], "{}_params.dat".format(star))
    params = parse_paramfile(param_file, path=None)
    obs_name = os.path.join(
        simulators.paths["spectra"], "{0}-{1}-mixavg-tellcorr_{2}.fits".format(star, obs_num, chip))
    output_prefix = os.path.join(
        simulators.paths["output_dir"], star.upper(), "{0}-{1}_{2}_iam_chisqr_results".format(
            star.upper(), obs_num, chip))
    os.makedirs(os.path.join(simulators.paths["output_dir"], star.upper()), exist_ok=True)
    return obs_name, params, output_prefix


def iam_analysis(obs_spec, model1_pars, model2_pars, rvs=None, gammas=None,
                 verbose=False, norm=False, save_only=True, chip=None,
                 prefix=None, errors=None):
    """Run two component model over all model combinations."""
    rvs = check_inputs(rvs)
    gammas = check_inputs(gammas)

    if isinstance(model1_pars, list):
        debug("Number of close model_pars returned {}".format(len(model1_pars)))
    if isinstance(model2_pars, list):
        debug("Number of close model_pars returned {}".format(len(model2_pars)))

    # Solution Grids to return
    iam_grid_chisqr_vals = np.empty((len(model1_pars), len(model2_pars)))

    args = [model2_pars, rvs, gammas, obs_spec]
    kwargs = {"norm": norm, "save_only": save_only, "chip": chip,
              "prefix": prefix, "verbose": verbose, "errors": errors}

    for ii, params1 in enumerate(tqdm(model1_pars)):
        iam_grid_chisqr_vals[ii] = iam_wrapper(ii, params1, *args, **kwargs)

    if save_only:
        return None
    else:
        return iam_grid_chisqr_vals  # Just output the best value for each model pair


def parallel_iam_analysis(obs_spec, model1_pars, model2_pars, rvs=None,
                          gammas=None, verbose=False, norm=False,
                          save_only=True, chip=None, prefix=None,
                          errors=None):
    """Run two component model over all model combinations."""
    rvs = check_inputs(rvs)
    gammas = check_inputs(gammas)

    if isinstance(model1_pars, list):
        debug("Number of close model_pars returned {}".format(len(model1_pars)))
    if isinstance(model2_pars, list):
        debug("Number of close model_pars returned {}".format(len(model2_pars)))

    def filled_iam_wrapper(num, param):
        """Fill in all extra parameters for parrallel wrapper."""
        return iam_wrapper(num, param, model2_pars, rvs, gammas,
                           obs_spec, norm=norm, save_only=save_only,
                           chip=chip, prefix=prefix, verbose=verbose, errors=errors)

    print("Parallelized running\n\n\n ###################")
    # raise NotImplementedError("Need to fix this up")
    iam_grid_chisqr_vals = Parallel(n_jobs=-2)(
        delayed(filled_iam_wrapper)(ii, param) for ii, param in enumerate(model1_pars))
    # iam_grid_chisqr_vals = Parallel(n_jobs=-2)(
    #     delayed(iam_wrapper)(ii, param, model2_pars, rvs, gammas,
    #                          obs_spec, norm=norm, save_only=save_only,
    #                          chip=chip, prefix=prefix, verbose=verbose)
    #     for ii, param in enumerate(model1_pars))

    if prefix is None:
        prefix = ""
    prefix += "_parallel"
    args = [model2_pars, rvs, gammas, obs_spec]
    kwargs = {"norm": norm, "save_only": save_only, "chip": chip,
              "prefix": prefix, "verbose": verbose, "errors": errors}

    iam_grid_chisqr_vals = Parallel(n_jobs=-2)(
        delayed(iam_wrapper)(ii, param, *args, **kwargs)
        for ii, param in enumerate(model1_pars))
    # iam_grid_chisqr_vals = np.empty_like(model1_pars)
    # for ii, param in enumerate(model1_pars):
    #    iam_grid_chisqr_vals[ii] = iam_wrapper(ii, param, *args, **kwargs)

    return iam_grid_chisqr_vals  # Just output the best value for each model pair


def continuum_alpha(model1, model2, chip=None):
    """Inherent flux ratio between the continuum of the two models.

    Assumes already scaled by area.
    Takes mean alpha of chip or full
    """
    assert not np.any(np.isnan(model1.xaxis))
    assert not np.any(np.isnan(model1.flux))
    assert not np.any(np.isnan(model2.xaxis))
    assert not np.any(np.isnan(model2.flux))
    # Fit models with continuum
    cont1 = continuum(model1.xaxis, model1.flux, method="exponential")
    cont2 = continuum(model2.xaxis, model2.flux, method="exponential")

    # Masking for individual chips
    if chip is None:
        chip = -1  # Full Crires range

    all_limits = {-1: [2111, 2169], 1: [2111, 2124], 2: [2125, 2139], 3: [2140, 2152], 4: [2153, 2169]}
    chip_limits = all_limits[chip]

    mask1 = (model1.xaxis > chip_limits[0]) * (model1.xaxis < chip_limits[1])
    mask2 = (model2.xaxis > chip_limits[0]) * (model2.xaxis < chip_limits[1])

    continuum_ratio = cont2[mask2] / cont1[mask1]
    alpha_ratio = np.nanmean(continuum_ratio)

    return alpha_ratio


def iam_wrapper(num, params1, model2_pars, rvs, gammas, obs_spec, norm=True,
                verbose=True, save_only=True, chip=None, prefix=None, errors=None):
    """Wrapper for iteration loop of iam. To use with parallelization."""
    if prefix is None:
        sf = os.path.join(
            simulators.paths["output_dir"], obs_spec.header["OBJECT"],
            "tc_{0}_{1}-{2}_part{6}_host_pars_[{3}_{4}_{5}].csv".format(
                obs_spec.header["OBJECT"], int(obs_spec.header["MJD-OBS"]), chip,
                params1[0], params1[1], params1[2], num))
    else:
        sf = "{0}_part{4}_host_pars_[{1}_{2}_{3}].csv".format(
            prefix, params1[0], params1[1], params1[2], num)
    save_filename = sf

    if os.path.exists(save_filename) and save_only:
        print("'{}' exists, so not repeating calculation.".format(save_filename))
        return None
    else:
        if not save_only:
            iam_grid_chisqr_vals = np.empty(len(model2_pars))
        for jj, params2 in enumerate(model2_pars):
            if verbose:
                print(("Starting iteration with parameters: "
                       "{0}={1},{2}={3}").format(num, params1, jj, params2))

            # ### Main Part ###
            rv_limits = observation_rv_limits(obs_spec, rvs, gammas)

            obs_spec = obs_spec.remove_nans()
            assert ~np.any(np.isnan(obs_spec.flux)), "Observation has nan"

            # Load phoenix models and scale by area and wavelength limit
            mod1_spec, mod2_spec = \
                prepare_iam_model_spectra(params1, params2, limits=rv_limits)

            # Estimated flux ratio from models
            inherent_alpha = continuum_alpha(mod1_spec, mod2_spec, chip)

            # Combine model spectra with iam model
            iam_grid_func = inherent_alpha_model(mod1_spec.xaxis, mod1_spec.flux, mod2_spec.flux,
                                                 rvs=rvs, gammas=gammas)
            iam_grid_models = iam_grid_func(obs_spec.xaxis)

            # Continuum normalize all iam_gird_models
            def axis_continuum(flux):
                """Continuum to apply along axis with predefined variables parameters."""
                return continuum(obs_spec.xaxis, flux, splits=50, method="exponential", top=5)

            iam_grid_continuum = np.apply_along_axis(axis_continuum, 0, iam_grid_models)

            iam_grid_models = iam_grid_models / iam_grid_continuum

            # ### RE-NORMALIZATION to observations?
            print("Shape of iam_grid_models before renormalization", iam_grid_models.shape)
            if norm:
                if verbose:
                    print("Re-normalizing!")
                obs_flux = chi2_model_norms(obs_spec.xaxis, obs_spec.flux,
                                            iam_grid_models, method="scalar")
            else:
                obs_flux = obs_spec.flux[:, np.newaxis, np.newaxis, np.newaxis]
                raise NotImplementedError("Need to check this")


            # Arbitrary_normalization of observation
            arb_norm = np.arange(*simulators.sim_grid["arb_norm"])
            iam_grid_models = iam_grid_models[:, :, :, np.newaxis] * arb_norm
            print("Arbitrary Normalized iam_grid_model shape.", iam_grid_models.shape)

            # Calculate Chi-squared
            obs_flux = obs_flux[:, :, :, np.newaxis]
            iam_norm_grid_chisquare = chi_squared(obs_flux, iam_grid_models, error=errors)

            print("Broadcast chi-squared values with arb norm", iam_norm_grid_chisquare.shape)

            # Take minimum chi-squared value along Arbitrary normalization axis
            min_locations = np.argmin(iam_norm_grid_chisquare, axis=-1)
            iam_grid_chisquare = np.min(iam_norm_grid_chisquare, axis=-1)
            print("Broadcast chi-squared values ", iam_grid_chisquare.shape)
            arbitrary_norms = arb_norm[min_locations]

            npix = obs_flux.shape[0]  # Number of pixels used

            if not save_only:
                iam_grid_chisqr_vals[jj] = iam_grid_chisquare.ravel()[np.argmin(iam_grid_chisquare)]

            save_full_iam_chisqr(save_filename, params1, params2,
                                 inherent_alpha, rvs, gammas,
                                 iam_grid_chisquare, arbitrary_norms, npix, verbose=verbose)

        if save_only:
            return None
        else:
            return iam_grid_chisqr_vals

def observation_rv_limits(obs_spec, rvs, gammas):
    """Calculate wavelenght limits needed to cover RV shifts used."""
    delta = spec_max_delta(obs_spec, rvs, gammas)
    obs_min, obs_max = min(obs_spec.xaxis), max(obs_spec.xaxis)
    return [obs_min - delta, obs_max + delta]


def prepare_iam_model_spectra(params1, params2, limits):
    """Load spectra with same settings."""
    mod1_spec = load_starfish_spectrum(params1, limits=limits,
                                       hdr=True, normalize=False, area_scale=True,
                                       flux_rescale=True)
    mod2_spec = load_starfish_spectrum(params2, limits=limits,
                                       hdr=True, normalize=False, area_scale=True,
                                       flux_rescale=True)
    assert len(mod1_spec.xaxis) > 0 and len(mod2_spec.xaxis) > 0
    assert np.all(mod1_spec.xaxis == mod2_spec.xaxis)

    return mod1_spec, mod2_spec


def save_full_iam_chisqr(filename, params1, params2, alpha, rvs, gammas,
                         iam_grid_chisquare, arbitrary_norms, npix, verbose=False):
    """Save the iterations chisqr values to a cvs."""
    R, G = np.meshgrid(rvs, gammas, indexing='ij')
    # assert A.shape == R.shape
    assert R.shape == G.shape
    assert G.shape == iam_grid_chisquare.shape

    data = {"rv": R.ravel(), "gamma": G.ravel(),
            "chi2": iam_grid_chisquare.ravel(), "arbnorm": arbitrary_norms.ravel()}

    columns = ["rv", "gamma", "chi2", "arbnorm"]
    len_c = len(columns)

    df = pd.DataFrame(data=data, columns=columns)
    # Update all rows with same value.
    for par, value in zip(["teff_2", "logg_2", "feh_2"], params2):
        df[par] = value

    columns = ["teff_2", "logg_2", "feh_2"] + columns

    if "[{}_{}_{}]".format(params1[0], params1[1], params1[2]) not in filename:
        # Need to add the model values.
        for par, value in zip(["teff_1", "logg_1", "feh_1"], params1):
            df[par] = value
        columns = ["teff_1", "logg_1", "feh_1"] + columns

    df["alpha"] = alpha
    df["npix"] = npix
    columns = columns[:-len_c] + ["alpha", "npix"] + columns[-len_c:]

    df = df.round(decimals={"logg_2": 1, "feh_2": 1, "alpha": 4,
                            "rv": 3, "gamma": 3, "chi2": 4})

    exists = os.path.exists(filename)
    if exists:
        df[columns].to_csv(filename, sep=',', mode="a", index=False, header=False)
    else:
        # Add header at the top only
        df[columns].to_csv(filename, sep=',', mode="a", index=False, header=True)

    if verbose:
        print("Saved chi-squared values to {}".format(filename))
    return None
