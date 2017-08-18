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
import logging
import os
import sys

# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import scipy as sp
from astropy.io import fits
from tqdm import tqdm

from Chisqr_of_observation import load_spectrum  # , select_observation
from models.broadcasted_models import two_comp_model
# from spectrum_overload.Spectrum import Spectrum

from two_component_model_HD211847 import save_full_chisqr, tcm_helper_function  # , plot_spectra
from utilities.chisqr import chi_squared
from utilities.crires_utilities import barycorr_crires_spectrum
from utilities.debug_utils import timeit2  # pv,
from utilities.norm import chi2_model_norms  # , renormalize_observation
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

test = True
if test:
    gammas = np.arange(-5, 5, 0.5)
    rvs = np.arange(-2, 2, 0.3)
    alphas = np.arange(0.01, 0.202, 0.05)
    print("Warning only small test feature set tried")
else:
    gammas = np.arange(-20, 20, 1)
    rvs = np.arange(-20, 20, 2)
    alphas = np.arange(0.01, 0.2, 0.02)


@timeit2
def main():
    """Main function."""
    star = "HD211847"
    obs_num = 2
    chip = 4

    obs_name, params, output_prefix = tcm_helper_function(star, obs_num, chip)

    print("The observation used is ", obs_name, "\n")

    host_params = [params["temp"], params["logg"], params["fe_h"]]
    comp_params = [params["comp_temp"], params["logg"], params["fe_h"]]

    closest_host_model = closest_model_params(*host_params)  # unpack temp, logg, fe_h with *
    closest_comp_model = closest_model_params(*comp_params)

    # original_model = "Z-0.0/lte05700-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"

    # Function to find the good models I need
    # models = find_phoenix_model_names(model_base_dir, original_model)
    # Function to find the good models I need from parameters
    # model_par_gen = generate_close_params(closest_host_model)
    model1_pars = list(generate_close_params(closest_host_model))  # Turn to list
    model2_pars = list(generate_close_params(closest_comp_model))

    # print("Model parameters", model1_pars)
    # print("Model parameters", model2_pars)

    # Load observation
    obs_spec = load_spectrum(obs_name)
    obs_spec = barycorr_crires_spectrum(obs_spec, -22)
    obs_spec.flux /= 1.02
    # Mask out bad portion of observed spectra ## HACK
    if chip == 4:
        # Ignore first 40 pixels
        obs_spec.wav_select(obs_spec.xaxis[40], obs_spec.xaxis[-1])

    if test:
        model1_pars = model1_pars[:6]
        model2_pars = model2_pars[:5]
        print("Warning only small test feature set tried")

    param_iter = len(alphas) * len(rvs) * len(gammas) * len(model2_pars) * len(model1_pars)

    print("STARTING tcm_analysis\nWith {} parameter iterations".format(param_iter))
    print("model1_pars", len(model1_pars), "model2_pars", len(model2_pars))
    ####
    chi2_grids = tcm_analysis(obs_spec, model1_pars, model2_pars, alphas, rvs, gammas, verbose=True, norm=True, chip=chip, prefix=output_prefix)

    bcast_chisqr_vals = chi2_grids

    print("tcm broadcast_chisquare shape", bcast_chisqr_vals.shape)
    print("tcm broadcast_chisquare", bcast_chisqr_vals)
    print("model1_pars len", len(model1_pars))
    print("model2_pars len", len(model2_pars))


    print("Done")


@timeit2
def tcm_analysis(obs_spec, model1_pars, model2_pars, alphas=None, rvs=None, gammas=None, verbose=False, norm=False, chip=None, prefix=None):
    """Run two component model over all parameter cobinations in model1_pars and model2_pars."""
    if chip is None:
        chip = ""

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
        if prefix is None:
            sf = ("Analysis/{0}/tc_{0}_{1}-{2}_part{6}_host_pars_[{3}_{4}_{5}]_par"
                  ".csv").format(obs_spec.header["OBJECT"],
                                 int(obs_spec.header["MJD-OBS"]), chip,
                                 params1[0], params1[1], params1[2], ii)

        else:
            sf = "{0}_part{4}_host_pars_[{1}_{2}_{3}]_par.csv".format(prefix,
                  params1[0], params1[1], params1[2], ii)
        save_filename = sf

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

            broadcast_result = two_comp_model(mod1_spec.xaxis, mod1_spec.flux, mod2_spec.flux, alphas=alphas, rvs=rvs, gammas=gammas)
            broadcast_values = broadcast_result(obs_spec.xaxis)

            assert ~np.any(np.isnan(obs_spec.flux)), "Observation is nan"

            #### NORMALIZATION NEEDED HERE
            if norm:
                obs_flux = chi2_model_norms(obs_spec.xaxis, obs_spec.flux, broadcast_values)

            else:
                obs_flux = obs_spec.flux[:, np.newaxis, np.newaxis, np.newaxis]
            #####

            broadcast_chisquare = chi_squared(obs_flux, broadcast_values)
            sp_chisquare = sp.stats.chisquare(obs_flux, broadcast_values, axis=0).statistic

            assert np.all(sp_chisquare == broadcast_chisquare)

            # print(broadcast_chisquare.shape)
            # print("chi squaresd = ", broadcast_chisquare.ravel()[np.argmin(broadcast_chisquare)])
            # New parameters to explore
            broadcast_chisqr_vals[ii, jj] = broadcast_chisquare.ravel()[np.argmin(broadcast_chisquare)]
            # broadcast_gamma[ii, jj] = gammas[np.argmin(broadcast_chisquare)]
            # full_broadcast_chisquare[ii, jj, :] = broadcast_chisquare

            save_full_chisqr(save_filename, params1, params2, alphas, rvs, gammas, broadcast_chisquare, verbose=verbose)

    return broadcast_chisqr_vals   # Just output the best value for each model pair


if __name__ == "__main__":
    sys.exit(main())
