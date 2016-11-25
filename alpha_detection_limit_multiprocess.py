#!/usr/bin/env python
# Test alpha variation at which cannot detect a planet

# Create a combined spectra with a planet at an alpha value.
# try and detect it by varying RV and alpha.
# At some stage the alpha will not vary when it becomes to small
# This will be the alpha detection limit.

# Maybe this is a wavelength dependant?

# The goal is to get something working and then try improve the performance
# for complete simulations.

# Create the test spectra.
from __future__ import division, print_function
from IP_multi_Convolution import IPconvolution
import numpy as np
import multiprocess as mprocess
from tqdm import tqdm
import scipy.stats
# from scipy.stats import chisquare
from simulation_utilities import combine_spectra
from Planet_spectral_simulations import load_PHOENIX_hd30501
import itertools
from collections import defaultdict
from datetime import datetime as dt
import time
import pickle
import matplotlib.pyplot as plt
# from astropy.io import fits
from spectrum_overload.Spectrum import Spectrum
import copy
from numba import jit
from joblib import Memory
# from joblib import Parallel, delayed
from simulation_utilities import spectrum_plotter
import os
import sys
sys.path.append("/home/jneal/Phd/Codes/Phd-codes/Simulations")

from new_alpha_detect_limit_simulation import alpha_model, parallel_chisqr
sys.path.append("/home/jneal/Phd/Codes/UsefulModules/Convolution")

path = "/home/jneal/Phd/Codes/Phd-codes/Simulations/saves"  # save path
cachedir = os.path.join(path, "cache")  # save path
memory = Memory(cachedir=cachedir, verbose=0)


@jit
def chi_squared(observed, expected, error=None):
    """Calculate chi squared.
    Same result as as scipy.stats.chisquare
    """
    if np.any(error):
        chisqr = np.sum((observed - expected) ** 2 / (error ** 2))
    else:
        # chisqr = np.sum((observed-expected)**2)
        chisqr = np.sum((observed - expected)**2 / expected)
        # When divided by exted the result is identical to scipy
    return chisqr

from simulation_utilities import add_noise
# @jit
# def add_noise(flux, SNR):
#     "Using the formulation mu/sigma from wikipedia"
#     sigma = flux / SNR
#     # Add normal distributed noise at the SNR level.
#     noisey_flux = flux + np.random.normal(0, sigma)
#     return noisey_flux


@memory.cache
def apply_convolution(model_spectrum, R=None, chip_limits=None):
    """ Apply convolution to spectrum object"""
    if chip_limits is None:
        chip_limits = (np.min(model_spectrum.xaxis),
                       np.max(model_spectrum.xaxis))

    if R is None:
        return copy.copy(model_spectrum)
    else:
        ip_xaxis, ip_flux = IPconvolution(model_spectrum.xaxis[:],
                                          model_spectrum.flux[:], chip_limits,
                                          R, FWHM_lim=5.0, plot=False,
                                          verbose=True)

        new_model = Spectrum(xaxis=ip_xaxis, flux=ip_flux,
                             calibrated=model_spectrum.calibrated,
                             header=model_spectrum.header)

        return new_model


@memory.cache
def store_convolutions(spectrum, resolutions, chip_limits=None):
    """ Convolve spectrum to many resolutions and store in a dict to retreive.
     This prevents multiple convolution at the same resolution.
    """
    d = dict()
    for resolution in resolutions:
        d[resolution] = apply_convolution(spectrum, resolution,
                                          chip_limits=chip_limits)
    return d


@memory.cache
def generate_observations(model_1, model_2, rv, alpha, resolutions, snrs,
                          limits):
    """ Create an simulated observation for combinations of resolution and snr.

    Paramters:
    model_1: and model_2 are Spectrum objects.
    rv: the rv offset applied to model_2
    alpha: flux ratio I(model_2)/I(model_1)
    resolutions: list of resolutions to simulate
    snrs: list of snr values to simulate

    Returns:
    observations: dict[resolution][snr] containing a simulated spectrum.

    """
    observations = defaultdict(dict)
    iterator = itertools.product(resolutions, snrs)
    for resolution, snr in iterator:
        # Preform tasks to simulate an observation
        # spec_1 = model_1[resolution]
        # spec_2 = model_2[resolution]
        # spec_2.doppler_shift(rv)
        # combined_model = combine_spectra(spec_1, spec_2, alpha)

        # Using alpha_model
        combined_model = alpha_model(alpha, rv, model_1[resolution],
                                     model_2[resolution], limits)

        combined_model.flux = add_noise(combined_model.flux, snr)

        observations[resolution][snr] = combined_model

    return observations


def parallel_chisquared(i, j, alpha, rv, res, snr, observation, host_models,
                        companion_models, output1, output2, X_memmap, Y_memmap):
    """ Function for parallel processing in chisqr for resolution and snr
    uses numpy memap to store data.

    Inputs:
    i,j = matrix locations of this alpha and rv
    alpha =  flux ratio
    rv = radial velocity offset of companion
    snr = signal to noise ratio of observation

    Output:
    None: Changes data in output1 and output2 numpy memmaps.
    """
    # print("i, j, Resolution, snr, alpha, RV")
    # print(i, j, res, snr, alpha, rv)
    host_model = host_models[res]
    companion_model = companion_models[res]
    companion_model.doppler_shift(rv)
    combined_model = combine_spectra(host_model, companion_model, alpha)
    # model_new = combine_spectra(convolved_star_models[resolution],
    # convolved_planet_models[resolution].doppler_shift(RV), alpha)

    combined_model.wav_select(2100, 2200)
    observation.wav_select(2100, 2200)

    # print("i", i, "j", j, "chisqr",
    # scipy.stats.chisquare(observation.flux, combined_model.flux).statistic)

    output1[i, j] = scipy.stats.chisquare(observation.flux,
                                          combined_model.flux).statistic
    # output2[i, j] = chi_squared(observation.flux, combined_model.flux,
    #                            error=observation.flux/snr)
    output2[i, j] = chi_squared(observation.flux, combined_model.flux,
                                error=None)
    X_memmap[i, j] = alpha
    Y_memmap[i, j] = rv

    if False:
        # if i == j:
        spectrum_plotter(observation, label="Simulated obs", show=False)
        spectrum_plotter(combined_model, label="This model", show=False)
        plt.title("Parallel printing, alpha={},RV={}".format(alpha, rv))
        plt.show()


def wrapper_parallel_chisquare(args):
    """ Wrapper for parallel_chisquare needed to unpack the arguments for
    parallel_chisquare as multiprocess.Pool.map does not accept multiple
    arguments
    """
    return parallel_chisquared(*args)


# @jit
def main():
    """ Chisquare determinination to detect minimum alpha value"""
    print("Loading Data")

    path = "/home/jneal/Phd/Codes/Phd-codes/Simulations/saves"  # save path

    chip_limits = [2080, 2220]
    chisqr_limits = [2110, 2120]  # Smaller limits after rv and convolutoin etc.
    (w_mod, I_star, I_bdmod,
        hdr_star, hdr_bd) = load_PHOENIX_hd30501(limits=chip_limits,
                                                 normalize=True)

    org_star_spec = Spectrum(xaxis=w_mod, flux=I_star, calibrated=True)
    org_bd_spec = Spectrum(xaxis=w_mod, flux=I_bdmod, calibrated=True)

    # Resolutions = [None, 50000]
    Resolutions = [50000, 100000]
    snrs = [50, 100, 1000]   # Signal to noise levels
    alphas = 10**np.linspace(-5, -1, 1000)
    RVs = np.arange(1, 35, 0.15)
    X, Y = np.meshgrid(RVs, alphas, indexing="xy")
    # Resolutions = [None, 1000, 10000, 50000, 100000, 150000, 200000]
    # snrs = [50, 100, 200, 500, 1000]   # Signal to noise levels
    # alphas = 10**np.linspace(-4, -0.1, 200)
    # RVs = np.arange(-100, 100, 0.05)

    # RV and alpha value of Simulations
    RV_val = 20
    Alpha = 0.005  # Vary this to determine detection limit
    input_parameters = (RV_val, Alpha)

    # starting convolution
    print("Begining convolution of models")
    timeInit = dt.now()
    convolved_star_models = store_convolutions(org_star_spec, Resolutions,
                                               chip_limits=chip_limits)
    convolved_planet_models = store_convolutions(org_bd_spec, Resolutions,
                                                 chip_limits=chip_limits)
    print("Convolution of models took {} seconds". format(dt.now()-timeInit))

    # print(type(convolved_star_models))
    # print(type(convolved_planet_models))
    simulated_observations = generate_observations(convolved_star_models,
                                                   convolved_planet_models,
                                                   RV_val, Alpha,
                                                   Resolutions, snrs,
                                                   chisqr_limits)

    # Not used with gernerator function
    # goal_planet_shifted = copy.copy(org_bd_spec)
    # RV shift BD spectra
    # goal_planet_shifted.doppler_shift(RV_val)

    res_snr_chisqr_dict = defaultdict(dict)  # Dictionary of dictionaries
    error_res_snr_chisqr_dict = defaultdict(dict)  # Dictionary of dictionaries # Empty now
    # multi_alpha = defaultdict(dict)  # Dictionary of dictionaries
    # multi_rv = defaultdict(dict)  # Dictionary of dictionaries

    # Iterable over resolution and snr to process
    # res_snr_iter = itertools.product(Resolutions, snrs)
    # Can then store to dict store_dict[res][snr]

    print("Starting loops")

    # multiprocessing part
    n_jobs = 4
    if n_jobs is None:
        n_jobs = mprocess.cpu_count() - 1

    # mprocPool = mprocess.Pool(processes=n_jobs)
    timeInit = dt.now()
    for resolution in tqdm(Resolutions):

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
            #                          mode='w+', shape=X.shape)
            # my_chisqr_memmap = np.memmap(my_filename, dtype='float32',
            #                              mode='w+', shape=X.shape)
            # new_X_memmap = np.memmap(os.path.join(path, "X.memmap"),
            #                          dtype='float32', mode='w+', shape=X.shape)
            # new_Y_memmap = np.memmap(os.path.join(path, "Y.memmap"),
            #                          dtype='float32', mode='w+', shape=X.shape)
            # # args_generator = tqdm([[i, j, alpha, rv, resolution, snr,
            # # sim_observation, convolved_star_models, convolved_planet_models,
            # # scipy_memmap, my_chisqr_memmap]
            # # for i, alpha in enumerate(alphas) for j, rv in enumerate(RVs)])
            #
            # # mprocPool.map(wrapper_parallel_chisquare, args_generator)
            #
            # Parallel(n_jobs=n_jobs)(delayed(parallel_chisquared)(i, j, alpha,
            #                           rv, resolution, snr, sim_observation,
            #                           convolved_star_models,
            #                           convolved_planet_models, scipy_memmap,
            #                           my_chisqr_memmap, new_X_memmap,
            #                           new_Y_memmap)
            #                           for j, alpha in enumerate(alphas)
            #                           for i, rv in enumerate(RVs))
            #
            # print(scipy_memmap)
            # res_snr_chisqr_dict[resolution][snr] = np.copy(scipy_memmap)
            # error_res_snr_chisqr_dict[resolution][snr] = np.copy(my_chisqr_memmap)
            # multi_alpha[resolution][snr] = np.copy(new_X_memmap)
            # multi_rv[resolution][snr] = np.copy(new_Y_memmap)

            # Trying new methodolgy
            chisqr_parallel = parallel_chisqr(alphas, RVs, sim_observation, alpha_model, (host_model, companion_model, chisqr_limits), n_jobs=n_jobs)
            # chisqr_parallel = parallel_chisqr(alphas, RVs, simlulated_obs, alpha_model, (org_star_spec, org_bd_spec, new_limits), n_jobs=4)
            res_snr_chisqr_dict[resolution][snr] = chisqr_parallel

    # mprocPool.close()
    timeEnd = dt.now()
    print("Multi-Proc chisqr has been completed in "
          "{} using {}/{} cores.\n".format(timeEnd-timeInit, n_jobs,
                                           mprocess.cpu_count()))

    with open(os.path.join(path, "parallel_chisquare.pickle"), "wb") as f:
        """Pickle all the necessary parameters to store

        """
        pickle.dump((Resolutions, snrs, alphas, RVs, input_parameters,
                    simulated_observations, convolved_star_models,
                    convolved_planet_models, res_snr_chisqr_dict,
                    error_res_snr_chisqr_dict), f)

        plot_after_running(Resolutions, snrs, alphas, RVs, input_parameters,
                           simulated_observations, convolved_star_models,
                           convolved_planet_models, res_snr_chisqr_dict,
                           error_res_snr_chisqr_dict)


def plot_after_running(Resolutions, snrs, alphas, RVs, input_parameters,
                       simulated_observations, convolved_star_models,
                       convolved_planet_models, res_snr_chisqr_dict,
                       error_res_snr_chisqr_dict):

    X, Y = np.meshgrid(RVs, alphas)

    for resolution in Resolutions:
        for snr in snrs:
            this_chisqr_snr = res_snr_chisqr_dict[resolution][snr]
            # this_error_chisqr_snr = error_res_snr_chisqr_dict[resolution][snr]
            log_chisqr = np.log(this_chisqr_snr)
            # log_error_chisqr = np.log(this_error_chisqr_snr)
            log_chisqr = log_chisqr.reshape(len(alphas), len(RVs))

            # T, U = np.meshgrid(RVs, alphas)
            plt.figure(figsize=(7, 7))
            plt.title("Log Chi squared with SNR = {0}, Resolution = {1}\n Correct RV = {2}, Correct alpha = {3}"
                      "".format(snr, resolution, input_parameters[0], input_parameters[1]), fontsize=16)

            # plt.subplot(2, 1, 1)
            plt.contourf(X, Y, log_chisqr, 100)
            plt.ylabel("Flux ratio")
            plt.xlabel("RV (km/s)")
            # plt.title("Chisquared")

        # plt.subplot(2, 1, 2)
        # plt.contourf(X, Y, log_error_chisqr, 100)
        # plt.title("Sigma chisquared")
        # plt.ylabel("Flux ratio")

        plt.show()

if __name__ == "__main__":
    start = time.time()
    main()
    print("Time to run = {} seconds".format(time.time()-start))
