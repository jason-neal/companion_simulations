"""two_compoonent_model.py.

Jason Neal
2nd Janurary 2017

Compare observed spectra to many different phoenix-aces spectral models to find which matches one is best fit by itself.

Need to determine the best RV offset to apply to the spectra.
Possibly bu sampling a few and then taking average value (they should be all
the same I would think unless the lines changed dramatically).

"""
from __future__ import division, print_function

# import itertools
import argparse
import logging
import os
import sys

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
# from utilities.chisqr import chi_squared
from utilities.crires_utilities import barycorr_crires_spectrum
from utilities.debug_utils import timeit2  # , pv
from utilities.norm import chi2_model_norms  # , renormalize_observation
from utilities.param_file import parse_paramfile
from utilities.phoenix_utils import (closest_model_params,
                                     generate_close_params,
                                     load_starfish_spectrum)
from utilities.simulation_utilities import max_delta

logging.basicConfig(level=logging.WARNING,
                    format='%(levelname)s %(message)s')
debug = logging.debug

model_base_dir = ("/home/jneal/Phd/data/PHOENIX-ALL/PHOENIX/")
wav_dir = "/home/jneal/Phd/data/PHOENIX-ALL/PHOENIX/"

wav_model = fits.getdata(os.path.join(wav_dir, "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"))
wav_model /= 10   # turn into nm


gammas = np.arange(-20, 20, 2)
rvs = np.arange(-10, 10, 1)
alphas = np.arange(0.01, 0.2, 0.005)


def _parser():
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='tcm')
    parser.add_argument('--chip', help='Chip Number.', default=None)
    parser.add_argument('-p', '--parallel', help='Use parallelization.', action="store_true")
    parser.add_argument('-s', '--small', help='Use smaller subset of parameters.', action="store_true")

    return parser.parse_args()


def tcm_helper_function(star, obs_num, chip):
    param_file = "/home/jneal/Phd/data/parameter_files/{}_params.dat".format(star)
    params = parse_paramfile(param_file, path=None)
    obs_name = "/home/jneal/.handy_spectra/{0}-{1}-mixavg-tellcorr_{2}.fits".format(star, obs_num, chip)

    output_prefix = "Analysis/{0}/{0}-{1}_{2}_bhm_chisqr_results".format(star.upper(), obs_num, chip)
    return obs_name, params, output_prefix


def main(chip=None, parallel=True, small=True):
    """Main function."""

    star = "HD211847"
    obs_num = 2

    if chip is None:
        chip = 4

    obs_name, params, output_prefix = tcm_helper_function(star, obs_num, chip)

    print("The observation used is ", obs_name, "\n")

    host_params = [params["temp"], params["logg"], params["fe_h"]]
    comp_params = [params["comp_temp"], params["logg"], params["fe_h"]]

    closest_host_model = closest_model_params(*host_params)  # unpack temp, logg, fe_h with *
    closest_comp_model = closest_model_params(*comp_params)

    # Function to find the good models I need
    # models = find_phoenix_model_names(model_base_dir, original_model)
    # Function to find the good models I need from parameters
    model1_pars = list(generate_close_params(closest_host_model, small=small))
    model2_pars = list(generate_close_params(closest_comp_model, small=small))

    # Load observation
    obs_spec = load_spectrum(obs_name)
    obs_spec = barycorr_crires_spectrum(obs_spec)
    # TODO
    # Mask out bad portion of observed spectra ## HACK
    if chip == 4:
        # Ignore first 40 pixels
        obs_spec.wav_select(obs_spec.xaxis[40], obs_spec.xaxis[-1])

    param_iter = len(alphas) * len(rvs) * len(gammas) * len(model2_pars) * len(model1_pars)
    print("STARTING tcm_analysis\nWith {} parameter iterations".format(param_iter))
    print("model1_pars", len(model1_pars), "model2_pars", len(model2_pars))

    ####
    if parallel:
        chi2_grids = parallel_tcm_analysis(obs_spec, model1_pars, model2_pars, alphas, rvs, gammas, verbose=True, norm=True, prefix=output_prefix, save_only=True)
    else:
        chi2_grids = tcm_analysis(obs_spec, model1_pars, model2_pars, alphas, rvs, gammas, verbose=True, norm=True, prefix=output_prefix)

    # Print TODO
    print("TODO: Add joining of sql table here")

    # subprocess.call(make_chi2_bd.py)


def check_inputs(var):
    if var is None:
        var = np.array([0])
    elif isinstance(rvs, (float, int)):
        var = np.asarray(var, dtype=np.float32)
    return var


@timeit2
def tcm_analysis(obs_spec, model1_pars, model2_pars, alphas=None, rvs=None, gammas=None, verbose=False, norm=False, save_only=True, chip=None, prefix=None):
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


def check_inputs(var):
    if var is None:
        var = np.array([0])
    elif isinstance(rvs, (float, int)):
        var = np.asarray(var, dtype=np.float32)
    return var


@timeit2
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

    args = [model2_pars, alphas, rvs, gammas, obs_spec]
    kwargs = {"norm": norm, "save_only": save_only, "chip": chip, "prefix": prefix, "verbose": verbose}

    broadcast_chisqr_vals = Parallel(n_jobs=-2)(
        delayed(tcm_wrapper)(ii, param, *args, **kwargs)
        for ii, param in enumerate(model1_pars))
    # broadcast_chisqr_vals = np.empty_like(model1_pars)
    # for ii, param in enumerate(model1_pars):
    #    broadcast_chisqr_vals[ii] = tcm_wrapper(ii, param, *args, **kwargs)

    return broadcast_chisqr_vals   # Just output the best value for each model pair


def tcm_wrapper(num, params1, model2_pars, alphas, rvs, gammas, obs_spec, norm=True, verbose=True, save_only=True, chip=None, prefix=None):
    """Wrapper for iteration loop of tcm. To use with parallization."""
    if prefix is None:
        sf = ("Analysis/{0}/tc_{0}_{1}-{2}_part{6}_host_pars_[{3}_{4}_{5}]_par"
              ".csv").format(obs_spec.header["OBJECT"],
                             int(obs_spec.header["MJD-OBS"]), chip,
                             params1[0], params1[1], params1[2], num)
    else:
        sf = "{0}_part{4}_host_pars_[{1}_{2}_{3}]_par.csv".format(prefix,
              params1[0], params1[1], params1[2], num)
    save_filename = sf

    broadcast_chisqr_vals = np.empty(len(model2_pars))
    for jj, params2 in enumerate(model2_pars):

        if verbose:
            print("Starting iteration with parameters:\n{0}={1},{2}={3}".format(num, params1, jj, params2))

        normalization_limits = [2105, 2185]   # small as possible?
        mod1_spec = load_starfish_spectrum(params1, limits=normalization_limits, hdr=True, normalize=True)
        mod2_spec = load_starfish_spectrum(params2, limits=normalization_limits, hdr=True, normalize=True)

        # TODO WHAT IS THE MAXIMUM (GAMMA + RV POSSIBLE? LIMIT IT TO THAT SHIFT?

            # Wavelength selection
            delta = max_delta(obs_spec, rvs, gammas)
            obs_min, obs_max = min(obs_spec.xaxis), max(obs_spec.xaxis)

        # One component model with broadcasting over gammas
        # two_comp_model(wav, model1, model2, alphas, rvs, gammas)
        assert np.allclose(mod1_spec.xaxis, mod2_spec.xaxis)
            mod1_spec.wav_select(obs_min - delta, obs_max + delta)
            mod2_spec.wav_select(obs_min - delta, obs_max + delta)
            obs_spec = obs_spec.remove_nans()

        broadcast_result = two_comp_model(mod1_spec.xaxis, mod1_spec.flux, mod2_spec.flux,
                                          alphas=alphas, rvs=rvs, gammas=gammas)
        broadcast_values = broadcast_result(obs_spec.xaxis)

        assert ~np.any(np.isnan(obs_spec.flux)), "Observation is nan"

        # ### NORMALIZATION NEEDED HERE
        if norm:
            obs_flux = chi2_model_norms(obs_spec.xaxis, obs_spec.flux, broadcast_values)
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

        save_full_chisqr(save_filename, params1, params2, alphas, rvs, gammas, broadcast_chisquare, verbose=verbose)

    if save_only:
        return None
    else:
        return broadcast_chisqr_vals


# @timeit
def save_full_chisqr(filename, params1, params2, alphas, rvs, gammas, broadcast_chisquare, verbose=False):
    """Save the iterations chisqr values to a cvs."""
    A, R, G = np.meshgrid(alphas, rvs, gammas, indexing='ij')
    assert A.shape == R.shape
    assert R.shape == G.shape
    assert G.shape == broadcast_chisquare.shape

    data = {"alpha": A.ravel(), "rv": R.ravel(), "gamma": G.ravel(),
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


def plot_spectra(obs, model):
    """Plot two spectra."""
    plt.plot(obs.xaxis, obs.flux, label="obs")
    plt.plot(model.xaxis, model.flux, label="model")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    args = vars(_parser())
    opts = {k: args[k] for k in args}

    # Iterate over chips
    if opts["chip"] is None:
        for chip in range(1, 5):
            opts["chip"] = chip
            res = main(**opts)
        sys.exit(res)
    else:
        sys.exit(main(**opts))
