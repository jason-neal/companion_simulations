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
import numpy as np
import time
import pickle
# import matplotlib.pyplot as plt
# from astropy.io import fits
from spectrum_overload.Spectrum import Spectrum
import copy
from numba import jit

import os
import sys

from IP_multi_Convolution import IPconvolution
from tqdm import tqdm
from scipy.stats import chisquare
from Planet_spectral_simulations import combine_spectra
from Planet_spectral_simulations import load_PHOENIX_hd30501
import itertools
from collections import defaultdict
sys.path.append("/home/jneal/Phd/Codes/UsefulModules/Convolution")


@jit
def chi_squared(observed, expected, error=None):
    """Calculate chi squared.
    Same result as as scipy.stats.chisquare
    """
    if np.any(error):
        chisqr = np.sum((observed-expected)**2 / error**2)
    else:
        # chisqr = np.sum((observed-expected)**2)
        chisqr = np.sum((observed-expected)**2 / expected)
        # When divided by exted the result is identical to scipy
    return chisqr


@jit
def alternate_chi_squared(observed, expected, error=None):
    """Calculate chi squared.
    Same result as as scipy.stats.chisquare
    """
    if error:
        chisqr = np.sum((observed-expected)**2 / observed)
    else:
        # chisqr = np.sum((observed-expected)**2)
        chisqr = np.sum((observed-expected)**2 / expected)
        # When divided by exted the result is identical to scipy
    return chisqr


@jit
def add_noise(flux, SNR):
    "Using the formulation mu/sigma"
    mu = np.mean(flux)
    sigma = mu / SNR
    # Add normal distributed noise at the SNR level.
    noisey_flux = flux + np.random.normal(0, sigma, len(flux))
    return noisey_flux


@jit
def add_noise2(flux, SNR):
    "Using the formulation mu/sigma"
    sigma = flux / SNR
    # Add normal distributed noise at the SNR level.
    noisey_flux = flux + np.random.normal(0, sigma)
    return noisey_flux


def apply_convolution(model_spectrum, R=None, chip_limits=None):
    """ Apply convolution to spectrum object"""
    if chip_limits is None:
        chip_limits = (np.min(model_spectrum.xaxis), np.max(model_spectrum.xaxis))

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


def store_convolutions(spectrum, resolutions, chip_limits=None):
    """ Convolve spectrum to many resolutions and store in a dict to retreive.
     This prevents multiple convolution at the same resolution.
    """
    d = dict()
    for resolution in resolutions:
        d[resolution] = apply_convolution(spectrum, resolution, chip_limits=chip_limits)

    return d


def generate_observations(model_1, model_2, rv, alpha, resolutions, snrs):
    """ Create an simulated obervation for combinations of resolution and snr.

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
        spec_1 = model_1[resolution]

        spec_2 = model_2[resolution]
        spec_2.doppler_shift(rv)
        # model1 and model2 are already normalized and convovled to each resolution using
        # store_convolutions
        combined_model = combine_spectra(spec_1, spec_2, alpha)

        combined_model.flux = add_noise2(combined_model.flux, snr)

        observations[resolution][snr] = combined_model

    return observations


# @jit
def main():
    """ Chisquare determinination to detect minimum alpha value"""
    print("Loading Data")

    path = "/home/jneal/Phd/Codes/Phd-codes/Simulations/saves"  # save path

    chip_limits = [2080, 2220]

    (w_mod, I_star, I_bdmod,
        hdr_star, hdr_bd) = load_PHOENIX_hd30501(limits=chip_limits,
                                                 normalize=True)

    org_star_spec = Spectrum(xaxis=w_mod, flux=I_star, calibrated=True)
    org_bd_spec = Spectrum(xaxis=w_mod, flux=I_bdmod, calibrated=True)

    Resolutions = [None, 50000]
    snrs = [100, 101, 110, 111]   # Signal to noise levels
    alphas = 10**np.linspace(-5, -0.2, 200)
    RVs = np.arange(10, 30, 0.1)
    # Resolutions = [None, 1000, 10000, 50000, 100000, 150000, 200000]
    # snrs = [50, 100, 200, 500, 1000]   # Signal to noise levels
    # alphas = 10**np.linspace(-4, -0.1, 200)
    # RVs = np.arange(-100, 100, 0.05)

    # RV and alpha value of Simulations
    RV_val = 20
    Alpha = 0.1  # Vary this to determine detection limit
    input_parameters = (RV_val, Alpha)

    convolved_star_model = store_convolutions(org_star_spec, Resolutions, chip_limits=chip_limits)
    convolved_planet_model = store_convolutions(org_bd_spec, Resolutions, chip_limits=chip_limits)

    # print(type(convolved_star_model))
    # print(type(convolved_planet_model))
    simulated_obersvations = generate_observations(convolved_star_model,
                                                   convolved_planet_model,
                                                   RV_val, Alpha,
                                                   Resolutions, snrs)

    # Not used with gernerator function
    goal_planet_shifted = copy.copy(org_bd_spec)
    # RV shift BD spectra
    goal_planet_shifted.doppler_shift(RV_val)

    # These should be replaced by
    res_stored_chisquared = dict()
    res_error_stored_chisquared = dict()
    # This
    res_snr_storage_dict = defaultdict(dict)  # Dictionary of dictionaries
    error_res_snr_storage_dict = defaultdict(dict)  # Dictionary of dictionaries
    # Iterable over resolution and snr to process
    # res_snr_iter = itertools.product(Resolutions, snrs)
    # Can then store to dict store_dict[res][snr]

    print("Starting loop")

    for resolution in tqdm(Resolutions):
        chisqr_snr_dict = dict()  # store 2d array in dict of SNR
        error_chisqr_snr_dict = dict()
        print("\nSTARTING run of RESOLUTION={}\n".format(resolution))

        star_spec = apply_convolution(org_star_spec, R=resolution,
                                      chip_limits=chip_limits)
        goal_planet = apply_convolution(goal_planet_shifted, R=resolution,
                                        chip_limits=chip_limits)

        # if resolution is None:
        #    star_spec = copy.copy(org_star_spec)
        #    goal_planet = copy.copy(goal_planet_shifted)
        # else:
        #    ip_xaxis, ip_flux = IPconvolution(org_star_spec.xaxis,
    #             org_star_spec.flux, chip_limits, resolution,
    #            FWHM_lim=5.0, plot=False, verbose=True)

    #        star_spec = Spectrum(xaxis=ip_xaxis, flux=ip_flux,
        #                                 calibrated=True,
        #                                 header=org_star_spec.header)

    #        ip_xaxis, ip_flux = IPconvolution(goal_planet_shifted.xaxis,
    #            goal_planet_shifted.flux, chip_limits, resolution,
    #            FWHM_lim=5.0, plot=False, verbose=False)

    #        goal_planet = Spectrum(xaxis=ip_xaxis, flux=ip_flux,
    #                                     calibrated=True,
    #                                     header=goal_planet_shifted.header)

        print("Starting SNR loop for resolution value of {}".format(resolution))
        for snr in snrs:
            loop_start = time.time()
            print("Calculation with snr level", snr)
            # This is the signal to try and recover
            Alpha_Combine = combine_spectra(star_spec, goal_planet, Alpha)
            Alpha_Combine.wav_select(2100, 2200)
            Alpha_Combine.flux = add_noise2(Alpha_Combine.flux, snr)

            # Test plot
            # plt.plot(Alpha_Combine.xaxis, Alpha_Combine.flux)
            sim_observation = simulated_obersvations[resolution][snr]
            # plt.plot(this_simulation.xaxis, this_simulation.flux, label="function generatred")
            # plt.legend()
            # plt.show()

            # chisqr_store = np.empty((len(alphas), len(RVs)))
            scipy_chisqr_store = np.empty((len(alphas), len(RVs)))
            error_chisqr_store = np.empty((len(alphas), len(RVs)))
            new_scipy_chisqr_store = np.empty((len(alphas), len(RVs)))
            new_error_chisqr_store = np.empty((len(alphas), len(RVs)))
            for i, alpha in enumerate(alphas):
                for j, RV in enumerate(RVs):
                    # print("RV", RV, "alpha", alpha, "snr", snr, "res", resolution)

                    # Generate model for this RV and alhpa
                    planet_shifted = copy.copy(org_bd_spec)
                    planet_shifted.doppler_shift(RV)
                    model = combine_spectra(star_spec, planet_shifted, alpha)
                    model.wav_select(2100, 2200)

                    # Try scipy chi_squared
                    scipy_chisquare = chisquare(Alpha_Combine.flux, model.flux)
                    error_chisquare = chi_squared(Alpha_Combine.flux, model.flux, error=Alpha_Combine.flux/snr)

                    # print("Mine, scipy", chisqr, scipy_chisquare)
                    error_chisqr_store[i, j] = error_chisquare
                    scipy_chisqr_store[i, j] = scipy_chisquare.statistic

                    #########################
                    # using dictionary values
                    host_model = convolved_star_model[resolution]
                    companion_model = convolved_planet_model[resolution]
                    companion_model.doppler_shift(RV)
                    model_new = combine_spectra(host_model, companion_model,
                                                alpha)

                    # model_new = combine_spectra(convolved_star_model[resolution], convolved_planet_model[resolution].doppler_shift(RV), alpha)
                    model_new.wav_select(2100, 2200)
                    sim_observation.wav_select(2100, 2200)

                    new_scipy_chisquare = chisquare(sim_observation.flux, model_new.flux)
                    new_error_chisquare = chi_squared(sim_observation.flux, model_new.flux, error=sim_observation.flux/snr)

                    new_error_chisqr_store[i, j] = new_error_chisquare
                    new_scipy_chisqr_store[i, j] = new_scipy_chisquare.statistic
                    ##############################

            chisqr_snr_dict[str(snr)] = scipy_chisqr_store
            error_chisqr_snr_dict[str(snr)] = error_chisqr_store

            res_snr_storage_dict[resolution][snr] = new_scipy_chisqr_store
            error_res_snr_storage_dict[resolution][snr] = new_error_chisqr_store

            # Save the results to a file to stop repeating loops

            for key, val in chisqr_snr_dict.items():
                np.save(os.path.join(path,
                        "scipy_chisquare_data_snr_{0}_res{1}".format(key,
                                                                     resolution
                                                                     )
                                     ), val)
            for key, val in error_chisqr_snr_dict.items():
                np.save(os.path.join(path,
                        "error_chisquare_data_snr_{0}_res{1}".format(key,
                                                                     resolution
                                                                     )
                                     ), val)
            # Store in dictionary
            res_stored_chisquared[resolution] = chisqr_snr_dict
            res_error_stored_chisquared[resolution] = error_chisqr_snr_dict

            print("SNR Loop time = {}".format(time.time() - loop_start))

    print("Finished Resolution {}".format(resolution))
    # Save the results to a file to stop repeating loops
    X, Y = np.meshgrid(RVs, alphas)
    np.save(os.path.join(path, "RV_mesgrid"), X)
    np.save(os.path.join(path, "alpha_meshgrid"), Y)
    np.save(os.path.join(path, "snr_values"), snrs)
    np.save(os.path.join(path, "Resolutions"), Resolutions)

    with open(os.path.join(path, "input_params.pickle"), "wb") as f:
        pickle.dump(input_parameters, f)
    # Try pickling the data

    with open(os.path.join(path, "alpha_chisquare.pickle"), "wb") as f:
        pickle.dump((Resolutions, snrs, X, Y, res_stored_chisquared, res_error_stored_chisquared), f)

        with open(os.path.join(path, "new_res_snr_chisquare.pickle"), "wb") as f:
            pickle.dump((Resolutions, snrs, X, Y, res_snr_storage_dict, error_res_snr_storage_dict), f)


if __name__ == "__main__":
    start = time.time()
    main()
    print("Time to run = {} seconds".format(time.time()-start))
