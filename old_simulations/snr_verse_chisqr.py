#!/usr/bin/python
"""snr_verse_chisquare.py.

Analyse how the addition of noise effects the chisquare on a spectrum with no companion.
Jason Neal, December 2016

"""
import os
import time
from collections import defaultdict

import numpy as np
import scipy
import tqdm

from mingle.utilities.chisqr import chi_squared
from mingle.utilities.model_convolution import apply_convolution, store_convolutions
from obsolete.simulate_obs import generate_noise_observations
from old_simulations.Planet_spectral_simulations import load_PHOENIX_hd30501


def main():
    """Chisquare determinination to detect minimum alpha value."""
    print("Loading Data")

    path = "/home/jneal/Phd/Codes/Phd-codes/Simulations/saves"  # save path

    chip_limits = [2080, 2220]

    org_star_spec, org_bd_spec = load_PHOENIX_hd30501(limits=chip_limits,
                                                      normalize=True)

    # org_star_spec = Spectrum(xaxis=w_mod, flux=I_star, calibrated=True)
    # org_bd_spec = Spectrum(xaxis=w_mod, flux=I_bdmod, calibrated=True)

    resolutions = [50000]
    snrs = [100, 101, 110, 111]  # Signal to noise levels
    # alphas = 10**np.linspace(-5, -0.2, 200)
    # RVs = np.arange(10, 30, 0.1)

    # RV and alpha value of Simulations
    # rv_val = 0
    # Alpha = 0  # Vary this to determine detection limit
    # input_parameters = (rv_val, Alpha)

    convolved_star_model = store_convolutions(org_star_spec, resolutions, chip_limits=chip_limits)
    # convolved_planet_model = store_convolutions(org_bd_spec, resolutions, chip_limits=chip_limits)

    # print(type(convolved_star_model))
    # print(type(convolved_planet_model))
    noisey_obersvations = generate_noise_observations(convolved_star_model,
                                                      convolved_planet_model,
                                                      rv_val, Alpha,
                                                      resolutions, snrs)

    # Not used with gernerator function
    goal_planet_shifted = copy.copy(org_bd_spec)
    # RV shift BD spectra
    goal_planet_shifted.doppler_shift(rv_val)

    # These should be replaced by
    res_stored_chisquared = dict()
    res_error_stored_chisquared = dict()
    # This
    res_snr_storage_dict = defaultdict(dict)  # Dictionary of dictionaries
    error_res_snr_storage_dict = defaultdict(dict)  # Dictionary of dictionaries
    # Iterable over resolution and snr to process
    # res_snr_iter = itertools.product(resolutions, snrs)
    # Can then store to dict store_dict[res][snr]

    print("Starting loop")

    for resolution in tqdm(resolutions):
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
            alpha_combine = combine_spectra(star_spec, goal_planet, Alpha)
            alpha_combine.wav_select(2100, 2200)
            # alpha_combine.flux = add_noise2(alpha_combine.flux, snr)
            alpha_combine.add_noise(snr)

            # Test plot
            # plt.plot(alpha_combine.xaxis, alpha_combine.flux)
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
                    scipy_chisquare = scipy.stats.chisquare(alpha_combine.flux, model.flux)
                    error_chisquare = chi_squared(alpha_combine.flux, model.flux, error=alpha_combine.flux / snr)

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

                    new_scipy_chisquare = scipy.stats.chisquare(sim_observation.flux, model_new.flux)
                    new_error_chisquare = chi_squared(sim_observation.flux, model_new.flux,
                                                      error=sim_observation.flux / snr)

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


if __name__ == "__main__":
    start = time.time()
    main()
    print("Time to run = {} seconds".format(time.time() - start))
