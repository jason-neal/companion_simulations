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
import matplotlib.pyplot as plt
# from astropy.io import fits
from spectrum_overload.Spectrum import Spectrum
import copy
from numba import jit

import os
from scipy.stats import chisquare
from Planet_spectral_simulations import combine_spectra
from Planet_spectral_simulations import load_PHOENIX_hd30501


@jit
def chi_squared(observed, expected, error=None):
    """Calculate chi squared.
    Same result as as scipy.stats.chisquare
    """
    if error:
        chisqr = np.sum((observed-expected)**2 / error**2)
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

# @jit
def main():
    """ Chisquare determinination to detect minimum alpha value"""
    (w_mod, I_star, I_bdmod,
        hdr_star, hdr_bd) = load_PHOENIX_hd30501(limits=[2080, 2220],
                                                 normalize=True)

    star_spec = Spectrum(xaxis=w_mod, flux=I_star, calibrated=True)
    bd_spec = Spectrum(xaxis=w_mod, flux=I_bdmod, calibrated=True)

    # Wavelength selection from 2100-2200nm
    # star_spec.wav_select(2080, 2220)  # extra 20nm for overlaps to be removed
    # bd_spec.wav_select(2080, 2220)
    # star_spec.wav_select(100, 3000)
    # bd_spec.wav_select(100, 3000)

    # RV_shift bd_spec
    RV_val = 20
    planet_shifted = copy.copy(bd_spec)
    # RV shift BD spectra
    planet_shifted.doppler_shift(RV_val)
    Alpha = 0.05  # Vary this to determine detection limit

    snrs = [50, 100, 200]   # Signal to noise levels
    # alphas = 10**np.linspace(-3,-0.5, 10)
    alphas = 10**np.linspace(-7, -0.3, 200)
    RVs = np.arange(0, 100, 0.1)
    Resolutions = [None, 20000, 50000, 100000]

    path = "/home/jneal/Phd/Codes/Phd-codes/Simulations/saves"  # save path
    chisqr_snr_dict = dict()  # store 2d array in dict of SNR
    for resolution in Resolutions:

        for snr in snrs:
            loop_start = time.time()
            print("Calculation with snr level", snr)
            # This is the signal to try and recover
            Alpha_Combine = combine_spectra(star_spec, planet_shifted, Alpha)
            if resolution is None:
                pass
            else:
                pass
            Alpha_Combine.wav_select(2100, 2200)
            Alpha_Combine.flux = add_noise2(Alpha_Combine.flux, snr)

            # Test plot
            # plt.plot(Alpha_Combine.xaxis, Alpha_Combine.flux)
            # plt.show()
            # chisqr_store = np.empty((len(alphas), len(RVs)))
            scipy_chisqr_store = np.empty((len(alphas), len(RVs)))
            error_chisqr_store = np.empty((len(alphas), len(RVs)))

            for i, alpha in enumerate(alphas):
                for j, RV in enumerate(RVs):
                    # print("RV", RV, "alpha", alpha)

                    # Generate model for this RV and alhpa
                    planet_shifted = copy.copy(bd_spec)
                    planet_shifted.doppler_shift(RV)
                    model = combine_spectra(star_spec, planet_shifted, alpha)
                    model.wav_select(2100, 2200)

                    # Try scipy chi_squared
                    scipy_chisquare = chisquare(Alpha_Combine.flux, model.flux)
                    error_chisquare = chi_square(Alpha_Combine.flux, model.flux, error=Alpha_Combine.flux/SNR)

                    # print("Mine, scipy", chisqr, scipy_chisquare)
                    error_chisqr_store[i, j] = error_chisquare
                    scipy_chisqr_store[i, j] = scipy_chisquare.statistic
            chisqr_snr_dict[str(snr)] = scipy_chisqr_store
            error_chisqr_snr_dict[str(snr)] = error_chisqr_store
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
            print("SNR Loop time = {}".format(time.time() - loop_start))

    print("Finished Resolution {}".format(resolution))
    # Save the results to a file to stop repeating loops
    X, Y = np.meshgrid(RVs, alphas)
    np.save(os.path.join(path, "RV_mesgrid"), X)
    np.save(os.path.join(path, "alpha_meshgrid"), Y)
    np.save(os.path.join(path, "snr_values"), snrs)
    np.save(os.path.join(path, "Resolutions", Resolutions))

if __name__ == "__main__":
    start = time.time()
    main()
    print("Time to run = {} seconds".format(time.time()-start))
