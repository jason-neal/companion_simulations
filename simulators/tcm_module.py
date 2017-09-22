import os

import logging
import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from tqdm import tqdm

from models.broadcasted_models import two_comp_model

from utilities.norm import chi2_model_norms
from utilities.param_file import parse_paramfile
from utilities.phoenix_utils import load_starfish_spectrum
from utilities.simulation_utilities import max_delta, check_inputs


debug = logging.debug


def tcm_helper_function(star, obs_num, chip):
    param_file = "/home/jneal/Phd/data/parameter_files/{}_params.dat".format(star)
    params = parse_paramfile(param_file, path=None)
    obs_name = "/home/jneal/.handy_spectra/{0}-{1}-mixavg-tellcorr_{2}.fits".format(star, obs_num, chip)

    output_prefix = "Analysis/{0}/{0}-{1}_{2}_bhm_chisqr_results".format(star.upper(), obs_num, chip)
    return obs_name, params, output_prefix


def tcm_analysis(obs_spec, model1_pars, model2_pars, alphas=None, rvs=None,
                 gammas=None, verbose=False, norm=False, save_only=True,
                 chip=None, prefix=None):
    """Run two component model over all parameter cobinations in model1_pars and model2_pars."""
    alphas = check_inputs(alphas)
    rvs = check_inputs(rvs)
    gammas = check_inputs(gammas)

    if isinstance(model1_pars, list):
        debug("Number of close model_pars returned {}".format(len(model1_pars)))
    if isinstance(model2_pars, list):
        debug("Number of close model_pars returned {}".format(len(model2_pars)))

    args = [model2_pars, alphas, rvs, gammas, obs_spec]
    kwargs = {"norm": norm, "save_only": save_only, "chip": chip, "prefix": prefix, "verbose": verbose}

    broadcast_chisqr_vals = np.empty((len(model1_pars), len(model2_pars)))

    for ii, params1 in enumerate(tqdm(model1_pars)):
        broadcast_chisqr_vals[ii] = tcm_wrapper(ii, params1, *args, **kwargs)

    if save_only:
        return None
    else:
        return broadcast_chisqr_vals   # Just output the best value for each model pair


def parallel_tcm_analysis(obs_spec, model1_pars, model2_pars, alphas=None,
                          rvs=None, gammas=None, verbose=False, norm=False, save_only=True, chip=None, prefix=None):
    """Run two component model over all parameter cobinations in model1_pars and model2_pars."""
    alphas = check_inputs(alphas)
    rvs = check_inputs(rvs)
    gammas = check_inputs(gammas)

    if isinstance(model1_pars, list):
        debug("Number of close model_pars returned {}".format(len(model1_pars)))
    if isinstance(model2_pars, list):
        debug("Number of close model_pars returned {}".format(len(model2_pars)))

    # def filled_tcm_wrapper(num, param):
    #     """Fill in all extra parameters for parrallel wrapper."""
    #    return tcm_wrapper(num, params, model2_pars, alphas, rvs, gammas,
    #                       obs_spec, norm=norm, save_only=save_only,
    #                       chip=chip, prefix=prefix, verbose=verbose)

    print("Parallised running\n\n\n ###################")
    raise NotImplementedError("Need to fix this up")
    # broadcast_chisqr_vals = Parallel(n_jobs=-2)(
    #    delayed(filled_tcm_wrapper)(ii, param) for ii, param in enumerate(model1_pars))
    # broadcast_chisqr_vals = Parallel(n_jobs=-2)(
    #     delayed(tcm_wrapper)(ii, param, model2_pars, alphas, rvs, gammas,
    #                          obs_spec, norm=norm, save_only=save_only,
    #                          chip=chip, prefix=prefix, verbose=verbose)
    #     for ii, param in enumerate(model1_pars))

    if prefix is None:
        prefix = ""
    prefix += "_parallel"

    args = [model2_pars, alphas, rvs, gammas, obs_spec]
    kwargs = {"norm": norm, "save_only": save_only, "chip": chip, "prefix": prefix, "verbose": verbose}

    broadcast_chisqr_vals = Parallel(n_jobs=-2)(
        delayed(tcm_wrapper)(ii, param, *args, **kwargs)
        for ii, param in enumerate(model1_pars))
    # broadcast_chisqr_vals = np.empty_like(model1_pars)
    # for ii, param in enumerate(model1_pars):
    #    broadcast_chisqr_vals[ii] = tcm_wrapper(ii, param, *args, **kwargs)

    return broadcast_chisqr_vals   # Just output the best value for each model pair


def tcm_wrapper(num, params1, model2_pars, alphas, rvs, gammas, obs_spec,
                norm=True, verbose=True, save_only=True, chip=None, prefix=None):
    """Wrapper for iteration loop of tcm. To use with parallization."""
    normalization_limits = [2105, 2185]   # small as possible?

    if prefix is None:
        sf = ("Analysis/{0}/tc_{0}_{1}-{2}_part{6}_host_pars_[{3}_{4}_{5}]"
              ".csv").format(obs_spec.header["OBJECT"],
                             int(obs_spec.header["MJD-OBS"]), chip,
                             params1[0], params1[1], params1[2], num)
    else:
        sf = "{0}_part{4}_host_pars_[{1}_{2}_{3}].csv".format(
            prefix, params1[0], params1[1], params1[2], num)
    save_filename = sf

    if os.path.exists(save_filename) and save_only:
        print("''{}' exists, so not repeating calcualtion.".format(save_filename))
        return None
    else:
        if not save_only:
            broadcast_chisqr_vals = np.empty(len(model2_pars))
        for jj, params2 in enumerate(model2_pars):
            if verbose:
                print("Starting iteration with parameters:\n {0}={1},{2}={3}".format(num, params1, jj, params2))

            mod1_spec = load_starfish_spectrum(params1, limits=normalization_limits, hdr=True, normalize=True)
            mod2_spec = load_starfish_spectrum(params2, limits=normalization_limits, hdr=True, normalize=True)

            # Wavelength selection
            delta = max_delta(obs_spec, rvs, gammas)
            obs_min, obs_max = min(obs_spec.xaxis), max(obs_spec.xaxis)

            mod1_spec.wav_select(obs_min - delta, obs_max + delta)
            mod2_spec.wav_select(obs_min - delta, obs_max + delta)
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
                obs_flux = chi2_model_norms(obs_spec.xaxis, obs_spec.flux, broadcast_values)
            else:
                obs_flux = obs_spec.flux[:, np.newaxis, np.newaxis, np.newaxis]

            # sp_chisquare is much faster but dont think I can add masking.
            # broadcast_chisquare = chi_squared(obs_flux, broadcast_values)
            sp_chisquare = stats.chisquare(obs_flux, broadcast_values, axis=0).statistic
            # assert np.all(sp_chisquare == broadcast_chisquare)
            broadcast_chisquare = sp_chisquare

            if not save_only:
                print(broadcast_chisquare.shape)
                print(broadcast_chisquare.ravel()[np.argmin(broadcast_chisquare)])

                broadcast_chisqr_vals[jj] = broadcast_chisquare.ravel()[np.argmin(broadcast_chisquare)]

            save_full_chisqr(save_filename, params1, params2, alphas, rvs, gammas, broadcast_chisquare, verbose=verbose)

        if save_only:
            return None
        else:
            return broadcast_chisqr_vals


def save_full_chisqr(filename, params1, params2, alphas, rvs, gammas, broadcast_chisquare, verbose=False):
    """Save the iterations chisqr values to a cvs."""
    a_grid, r_grid, g_grid = np.meshgrid(alphas, rvs, gammas, indexing='ij')
    assert a_grid.shape == r_grid.shape
    assert r_grid.shape == g_grid.shape
    assert g_grid.shape == broadcast_chisquare.shape

    data = {"alpha": a_grid.ravel(), "rv": r_grid.ravel(), "gamma": g_grid.ravel(),
            "chi2": broadcast_chisquare.ravel()}

    columns = ["alpha", "rv", "gamma", "chi2"]

    df = pd.DataFrame(data=data, columns=columns)

    for par, value in zip(["teff_2", "logg_2", "feh_2"], params2):
        df[par] = value

    columns = ["teff_2", "logg_2", "feh_2"] + columns

    if "[{}_{}_{}]".format(params1[0], params1[1], params1[2]) not in filename:
        for par, value in zip(["teff_1", "logg_1", "feh_1"], params1):
            df[par] = value
        columns = ["teff_1", "logg_1", "feh_1"] + columns

    df = df.round(decimals={"logg_2": 1, "feh_2": 1, "alpha": 4,
                            "rv": 3, "gamma": 3, "chi2": 4})

    exists = os.path.exists(filename)
    if exists:
        df[columns].to_csv(filename, sep=',', mode="a", index=False, header=False)
    else:
        # Add header at the top only
        df[columns].to_csv(filename, sep=',', mode="a", index=False, header=True)

    if verbose:
        print("Saved chi2 values to {}".format(filename))
    return None
