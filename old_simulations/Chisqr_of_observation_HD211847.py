#!/usr/bin/env python
"""Chi square of actual data observation.

Jason Neal November 2016.
"""

import logging
import os
import pickle
from ajplanet import pl_rv_array
from datetime import datetime as dt

import ephem
import matplotlib.pyplot as plt
import multiprocess as mprocess
import numpy as np
from mingle.utilities.Chisqr_of_observation import plot_obs_with_model
from mingle.utilities.Planet_spectral_simulations import \
    load_PHOENIX_hd211847  # load_PHOENIX_hd30501,

from obsolete.models.alpha_model import alpha_model2
from mingle.utilities.chisqr import parallel_chisqr
from mingle.utilities.crires_utilities import (barycorr_crires_spectrum,
                                               crires_resolution)
from mingle.utilities.debug_utils import pv
from mingle.utilities.model_convolution import convolve_models  # , apply_convolution,
from mingle.utilities.spectrum_utils import load_spectrum, select_observation

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')


def main():
    """Main."""
    star = "HD211847"
    obsnum = "2"
    chip = 1
    obs_name, path = select_observation(star, obsnum, chip)

    # Load observation
    observed_spectra = load_spectrum(obs_name)

    if chip == 4:
        # Ignore first 40 pixels
        observed_spectra.wav_select(observed_spectra.xaxis[40], observed_spectra.xaxis[-1])

    # Load models
    # host_spectrum_model, companion_spectrum_model = load_PHOENIX_hd30501(limits=[2100, 2200], normalize=True)
    host_spectrum_model, companion_spectrum_model = load_PHOENIX_hd211847(limits=[2100, 2200], normalize=True)

    obs_resolution = crires_resolution(observed_spectra.header)

    # Convolve models to resolution of instrument
    host_spectrum_model, companion_spectrum_model = convolve_models((host_spectrum_model, companion_spectrum_model),
                                                                    obs_resolution, chip_limits=None)

    plot_obs_with_model(observed_spectra, host_spectrum_model, companion_spectrum_model,
                        show=False, title="Before BERV Correction")

    # Berv Correct
    # Calculate the star RV relative to synthetic spectum
    #                        [mean_val, K1, omega,   e,     Tau,       Period, starmass (Msun), msini(Mjup), i]
    parameters = {"HD30501": [23.710, 1703.1, 70.4, 0.741, 53851.5, 2073.6, 0.81, 90],
                  "HD211847": [6.689, 291.4, 159.2, 0.685, 62030.1, 7929.4, 0.94, 19.2, 7]}

    try:
        host_params = parameters[star]
    except ValueError:
        print("Parameters for {} are not in parameters list. Improve this.".format(star))
        raise
    host_params[1] = host_params[1] / 1000  # Convert K! to km/s
    host_params[2] = np.deg2rad(host_params[2])  # Omega needs to be in radians for ajplanet

    obs_time = observed_spectra.header["DATE-OBS"]
    print(obs_time, isinstance(obs_time, str))
    print(obs_time.replace("T", " ").split("."))
    jd = ephem.julian_date(obs_time.replace("T", " ").split(".")[0])
    host_rv = pl_rv_array(jd, *host_params[0:6])[0]
    print("host_rv", host_rv, "km/s")

    offset = -host_rv  # -22
    logging.warning("Non-zero rv offset", pv("offset"))
    # offset = 0  # -22
    _berv_corrected_observed_spectra = barycorr_crires_spectrum(observed_spectra, offset)  # Issue with air/vacuum
    # This introduces nans into the observed spectrum
    berv_corrected_observed_spectra.wav_select(*berv_corrected_observed_spectra.xaxis[
        np.isfinite(berv_corrected_observed_spectra.flux)][[0, -1]])
    # Shift to star RV

    plot_obs_with_model(berv_corrected_observed_spectra, host_spectrum_model,
                        companion_spectrum_model, title="After BERV Correction")

    # print("\nWarning!!!\n BERV is not good have added a offset to get rest working\n")

    # Chisquared fitting
    alphas = 10 ** np.linspace(-4, -0.5, 100)
    rvs = np.arange(-50, 50, 0.05)

    # chisqr_store = np.empty((len(alphas), len(rvs)))
    observed_limits = [np.floor(berv_corrected_observed_spectra.xaxis[0]),
                       np.ceil(berv_corrected_observed_spectra.xaxis[-1])]
    print("Observed_limits ", observed_limits)

    n_jobs = 1
    if n_jobs is None:
        n_jobs = mprocess.cpu_count() - 1
    start_time = dt.now()

    if np.all(np.isnan(host_spectrum_model.flux)):
        print("Host spectrum is all Nans")
    if np.all(np.isnan(companion_spectrum_model.flux)):
        print("Companion spectrum is all Nans")

    print("Now performing the Chisqr grid analaysis")
    obs_chisqr_parallel = parallel_chisqr(alphas, rvs, berv_corrected_observed_spectra, alpha_model2,
                                          (host_spectrum_model, companion_spectrum_model,
                                           observed_limits, berv_corrected_observed_spectra), n_jobs=n_jobs)
    # chisqr_parallel = parallel_chisqr(alphas, rvs, simlulated_obs, alpha_model, (org_star_spec,
    #                                   org_bd_spec, new_limits), n_jobs=4)

    end_time = dt.now()
    print("Time to run parallel chisquared = {}".format(end_time - start_time))
    # Plot memmap
    # plt.subplot(2, 1, 1)
    x, y = np.meshgrid(rvs, alphas)
    fig = plt.figure(figsize=(7, 7))
    cf = plt.contourf(x, y, np.log10(obs_chisqr_parallel.reshape(len(alphas), len(rvs))), 100)
    fig.colorbar(cf)
    plt.title("Sigma chisquared")
    plt.ylabel("Flux ratio")
    plt.xlabel("RV (km/s)")
    plt.show()

    # Locate minimum and plot resulting model next to observation
    def find_min_chisquared(x, y, z):
        """Find minimum vlaue in chisqr grid."""
        min_loc = np.argmin(z)
        print("min location", min_loc)

        x_sol = x.ravel()[min_loc]
        y_sol = y.ravel()[min_loc]
        z_sol = z.ravel()[min_loc]
        return x_sol, y_sol, z_sol, min_loc

    rv_solution, alpha_solution, min_chisqr, min_loc = find_min_chisquared(x, y, obs_chisqr_parallel)
    print("Minium Chisqr value {2}\n RV sol = {0}\nAlpha Sol = {1}".format(rv_solution, alpha_solution, min_chisqr))

    solution_model = alpha_model2(alpha_solution, rv_solution, host_spectrum_model, companion_spectrum_model,
                                  observed_limits)
    # alpha_model2(alpha, rv, host, companion, limits, new_x=None):

    plt.plot(solution_model.xaxis, solution_model.flux, label="Min chisqr solution")
    plt.plot(berv_corrected_observed_spectra.xaxis, berv_corrected_observed_spectra.flux, label="Observation")
    plt.legend(loc=0)
    plt.show()

    # Dump the results into a pickle file
    pickle_name = "Chisqr_results_{0}_{1}_chip_{2}.pickle".format(star, obsnum, chip)
    with open(os.path.join(path, pickle_name), "wb") as f:
        # Pickle all the necessary parameters to store.
        pickle.dump((rvs, alphas, berv_corrected_observed_spectra, host_spectrum_model, companion_spectrum_model,
                     rv_solution, alpha_solution, min_chisqr, min_loc, solution_model), f)


if __name__ == "__main__":
    main()
