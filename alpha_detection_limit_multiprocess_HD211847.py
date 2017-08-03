#!/usr/bin/env python
# Test alpha variation at which cannot detect a planet

# Create a combined spectra with a planet at an alpha value.
# try and detect it by varying rv and alpha.
# At some stage the alpha will not vary when it becomes to small
# This will be the alpha detection limit.

# Maybe this is a wavelength dependant?

# The goal is to get something working and then try improve the performance
# for complete simulations.

# Create the test spectra.
from __future__ import division, print_function

import os
import sys
import time
import pickle
import logging
import scipy.stats
import numpy as np
from tqdm import tqdm
from joblib import Memory
import matplotlib.pyplot as plt
import multiprocess as mprocess
from collections import defaultdict
from datetime import datetime as dt
from utilities.simulation_utilities import combine_spectra
from Planet_spectral_simulations import load_PHOENIX_hd211847

from utilities.simulation_utilities import spectrum_plotter
from utilities.chisqr import chi_squared
from utilities.chisqr import parallel_chisqr
from models.alpha_model import alpha_model

from utilities.model_convolution import store_convolutions
from utilities.simulate_obs import generate_observations2 as generate_observations

sys.path.append("/home/jneal/Phd/Codes/equanimous-octo-tribble/Convolution")
sys.path.append("/home/jneal/Phd/Codes/UsefulModules/Convolution")

cachedir = "/home/jneal/.simulation_cache"
if not os.path.exists(cachedir):
    os.makedirs(cachedir)
memory = Memory(cachedir=cachedir, verbose=0)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')
debug = logging.debug


def parallel_chisquared(i, j, alpha, rv, res, snr, observation, host_models,
                        companion_models, output1, output2, x_memmap, y_memmap):
    """Function for parallel processing in chisqr for resolution and snr uses numpy memap to store data.

    Inputs:
    i,j = matrix locations of this alpha and rv
    alpha =  flux ratio
    rv = radial velocity offset of companion
    snr = signal to noise ratio of observation

    Output:
    None: Changes data in output1 and output2 numpy memmaps.
    """
    # print("i, j, Resolution, snr, alpha, rv")
    # print(i, j, res, snr, alpha, rv)
    host_model = host_models[res]
    companion_model = companion_models[res]
    companion_model.doppler_shift(rv)
    combined_model = combine_spectra(host_model, companion_model, alpha)
    # model_new = combine_spectra(convolved_star_models[resolution],
    # convolved_planet_models[resolution].doppler_shift(rv), alpha)

    observation.wav_select(2100, 2200)
    # INTERPOLATE COMBINED MODEL VALUES TO OBSERVATION VALUES
    combined_model.spline_interpolate_to(observation)

    # print("i", i, "j", j, "chisqr",
    # scipy.stats.chisquare(observation.flux, combined_model.flux).statistic)

    output1[i, j] = scipy.stats.chisquare(observation.flux,
                                          combined_model.flux).statistic
    # output2[i, j] = chi_squared(observation.flux, combined_model.flux,
    #                            error=observation.flux/snr)
    output2[i, j] = chi_squared(observation.flux, combined_model.flux,
                                error=None)
    x_memmap[i, j] = alpha
    y_memmap[i, j] = rv

    if False:
        # if i == j:
        spectrum_plotter(observation, label="Simulated obs", show=False)
        spectrum_plotter(combined_model, label="This model", show=False)
        plt.title("Parallel printing, alpha={},rv={}".format(alpha, rv))
        plt.show()


def wrapper_parallel_chisquare(args):
    """Wrapper for parallel_chisquare.

    It is needed to unpack the arguments for parallel_chisquare as
    multiprocess.Pool.map does not accept multiple arguments.
    """
    return parallel_chisquared(*args)


def main():
    """Chisquare determinination to detect minimum alpha value."""
    print("Loading Data")

    path = "/home/jneal/Phd/Codes/companion_simulations/saves"  # save path

    chip_limits = [2080, 2220]
    chisqr_limits = [2110, 2120]  # Smaller limits after rv and convolutoin etc.
    org_star_spec, org_bd_spec = load_PHOENIX_hd211847(limits=chip_limits, normalize=True)

    # resolutions = [None, 50000]
    resolutions = [50000, 100000]
    snrs = [50, 100, 1000]   # Signal to noise levels
    alphas = 10**np.linspace(-5, -1, 1000)
    rvs = np.arange(1, 35, 0.15)
    x, y = np.meshgrid(rvs, alphas, indexing="xy")
    # resolutions = [None, 1000, 10000, 50000, 100000, 150000, 200000]
    # snrs = [50, 100, 200, 500, 1000]   # Signal to noise levels
    # alphas = 10**np.linspace(-4, -0.1, 200)
    # rvs = np.arange(-100, 100, 0.05)

    # rv and alpha value of Simulations
    rv_val = 20
    alpha_val = 0.2  # Vary this to determine detection limit
    input_parameters = (rv_val, alpha_val)

    # starting convolution
    print("Begining convolution of models")
    time_init = dt.now()
    convolved_star_models = store_convolutions(org_star_spec, resolutions,
                                               chip_limits=chip_limits)
    convolved_planet_models = store_convolutions(org_bd_spec, resolutions,
                                                 chip_limits=chip_limits)
    print("Convolution of models took {} seconds". format(dt.now() - time_init))

    simulated_observations = generate_observations(convolved_star_models,
                                                   convolved_planet_models,
                                                   rv_val, alpha_val,
                                                   resolutions, snrs,
                                                   chisqr_limits)

    # Not used with gernerator function
    # goal_planet_shifted = copy.copy(org_bd_spec)
    # rv shift BD spectra
    # goal_planet_shifted.doppler_shift(rv_val)

    res_snr_chisqr_dict = defaultdict(dict)  # Dictionary of dictionaries
    error_res_snr_chisqr_dict = defaultdict(dict)  # Dictionary of dictionaries # Empty now
    # multi_alpha = defaultdict(dict)  # Dictionary of dictionaries
    # multi_rv = defaultdict(dict)  # Dictionary of dictionaries

    # Iterable over resolution and snr to process
    # res_snr_iter = itertools.product(resolutions, snrs)
    # Can then store to dict store_dict[res][snr]

    print("Starting loops")

    # multiprocessing part
    n_jobs = 4
    if n_jobs is None:
        n_jobs = mprocess.cpu_count() - 1

    # mprocPool = mprocess.Pool(processes=n_jobs)
    time_init = dt.now()
    for resolution in tqdm(resolutions):

        print("\nSTARTING run of RESOLUTION={}\n".format(resolution))
        # chisqr_snr_dict = dict()  # store 2d array in dict of SNR
        # error_chisqr_snr_dict = dict()
        host_model = convolved_star_models[resolution]
        companion_model = convolved_planet_models[resolution]

        for snr in snrs:
            sim_observation = simulated_observations[resolution][snr]

            # Multiprocessing part
            # scipy_filename = os.path.join(path, "scipychisqr.memmap")
            # my_filename = os.path.join(path, "mychisqr.memmap")
            # scipy_memmap = np.memmap(scipy_filename, dtype='float32',
            #                          mode='w+', shape=x.shape)
            # my_chisqr_memmap = np.memmap(my_filename, dtype='float32',
            #                              mode='w+', shape=x.shape)
            # new_x_memmap = np.memmap(os.path.join(path, "x.memmap"),
            #                          dtype='float32', mode='w+', shape=x.shape)
            # new_y_memmap = np.memmap(os.path.join(path, "y.memmap"),
            #                          dtype='float32', mode='w+', shape=x.shape)
            # # args_generator = tqdm([[i, j, alpha, rv, resolution, snr,
            # # sim_observation, convolved_star_models, convolved_planet_models,
            # # scipy_memmap, my_chisqr_memmap]
            # # for i, alpha in enumerate(alphas) for j, rv in enumerate(rvs)])
            #
            # # mprocPool.map(wrapper_parallel_chisquare, args_generator)
            #
            # Parallel(n_jobs=n_jobs)(delayed(parallel_chisquared)(i, j, alpha,
            #                           rv, resolution, snr, sim_observation,
            #                           convolved_star_models,
            #                           convolved_planet_models, scipy_memmap,
            #                           my_chisqr_memmap, new_x_memmap,
            #                           new_y_memmap)
            #                           for j, alpha in enumerate(alphas)
            #                           for i, rv in enumerate(rvs))
            #
            # print(scipy_memmap)
            # res_snr_chisqr_dict[resolution][snr] = np.copy(scipy_memmap)
            # error_res_snr_chisqr_dict[resolution][snr] = np.copy(my_chisqr_memmap)
            # multi_alpha[resolution][snr] = np.copy(new_x_memmap)
            # multi_rv[resolution][snr] = np.copy(new_y_memmap)

            # Trying new methodolgy
            chisqr_parallel = parallel_chisqr(alphas, rvs, sim_observation, alpha_model,
                                              (host_model, companion_model, chisqr_limits),
                                              n_jobs=n_jobs)
            # chisqr_parallel = parallel_chisqr(alphas, rvs, simlulated_obs, alpha_model,
            #                                   (org_star_spec, org_bd_spec, new_limits),
            #                                   n_jobs=4)
            res_snr_chisqr_dict[resolution][snr] = chisqr_parallel

    # mprocPool.close()
    time_end = dt.now()
    print("Multi-Proc chisqr has been completed in "
          "{} using {}/{} cores.\n".format(time_end - time_init, n_jobs,
                                           mprocess.cpu_count()))

    with open(os.path.join(path, "parallel_chisquare.pickle"), "wb") as f:
        """Pickle all the necessary parameters to store

        """
        pickle.dump((resolutions, snrs, alphas, rvs, input_parameters,
                    simulated_observations, convolved_star_models,
                    convolved_planet_models, res_snr_chisqr_dict,
                    error_res_snr_chisqr_dict), f)

        plot_after_running(resolutions, snrs, alphas, rvs, input_parameters,
                           simulated_observations, convolved_star_models,
                           convolved_planet_models, res_snr_chisqr_dict,
                           error_res_snr_chisqr_dict)


def plot_after_running(resolutions, snrs, alphas, rvs, input_parameters,
                       simulated_observations, convolved_star_models,
                       convolved_planet_models, res_snr_chisqr_dict,
                       error_res_snr_chisqr_dict):

    x, y = np.meshgrid(rvs, alphas)

    for resolution in resolutions:
        for snr in snrs:
            this_chisqr_snr = res_snr_chisqr_dict[resolution][snr]
            # this_error_chisqr_snr = error_res_snr_chisqr_dict[resolution][snr]
            log_chisqr = np.log(this_chisqr_snr)
            # log_error_chisqr = np.log(this_error_chisqr_snr)
            log_chisqr = log_chisqr.reshape(len(alphas), len(rvs))

            # T, U = np.meshgrid(rvs, alphas)
            plt.figure(figsize=(7, 7))
            plt.title("Log Chi squared with SNR = {0}, Resolution = {1}\n Correct rv = {2}, Correct alpha = {3}"
                      "".format(snr, resolution, input_parameters[0], input_parameters[1]), fontsize=16)

            # plt.subplot(2, 1, 1)
            plt.contourf(x, y, log_chisqr, 100)
            plt.ylabel("Flux ratio")
            plt.xlabel("rv (km/s)")
            # plt.title("Chisquared")

        # plt.subplot(2, 1, 2)
        # plt.contourf(x, y, log_error_chisqr, 100)
        # plt.title("Sigma chisquared")
        # plt.ylabel("Flux ratio")

        plt.show()


if __name__ == "__main__":
    start = time.time()
    main()
    print("Time to run = {} seconds".format(time.time() - start))
