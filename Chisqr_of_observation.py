#!/usr/bin/env python

# Chi square of actual data observation

# Jason Neal November 2016

import os
import sys
import copy
import ephem
import pickle
import numpy as np
from joblib import Memory
from astropy.io import fits
import multiprocess as mprocess
import matplotlib.pyplot as plt
from datetime import datetime as dt

from Get_filenames import get_filenames
from IP_multi_Convolution import IPconvolution
from spectrum_overload.Spectrum import Spectrum
from simulation_utilities import combine_spectra
from Planet_spectral_simulations import load_PHOENIX_hd30501
sys.path.append("/home/jneal/Phd/Codes/Phd-codes/Simulations")
from new_alpha_detect_limit_simulation import parallel_chisqr  # , alpha_model
from crires_utilities import crires_resolution
from crires_utilities import barycorr_crires_spectrum

from ajplanet import pl_rv_array

path = "/home/jneal/Phd/Codes/Phd-codes/Simulations/saves"  # save path
cachedir = os.path.join(path, "cache")  # save path
memory = Memory(cachedir=cachedir, verbose=0)


# First plot the observation with the model
def plot_obs_with_model(obs, model1, model2=None, show=True, title=None):
    """ Plot the obseved spectrum against the model to check that they are
    "compatiable"
    """
    plt.figure()
    plt.plot(obs.xaxis, obs.flux + 1, label="Observed")
    # plt.plot(obs.xaxis, np.isnan(obs.flux) + 1, "o", ms=15, label="Nans in obs")
    plt.plot(model1.xaxis, model1.flux + 1.1, label="model1")
    if model2:
        plt.plot(model2.xaxis, model2.flux, label="model2")
    plt.legend(loc=0)
    plt.xlim(-1 + obs.xaxis[0], 1 + obs.xaxis[-1])
    if title is not None:
        plt.title(title)
    if show:
        plt.show()


# I should already have these sorts of functions
def select_observation(star, obs_num, chip):
    """ Select the observation to load in

    inputs:
    star: name of host star target
    obs_num: observation number
    chip: crires detetor chip number

    returns:
    crires_name: name of file
    """
    if str(chip) not in "1234":
        print("The Chip is not correct. It needs to be 1,2,3 or 4")
        raise Exception("Chip Error")
    else:

        path = ("/home/jneal/Phd/data/Crires/BDs-DRACS/{}-"
                "{}/Combined_Nods".format(star, obs_num))
        print("Path =", path)
        filenames = get_filenames(path, "CRIRE.*wavecal.tellcorr.fits",
                                  "*_{}.nod.ms.*".format(chip))

        crires_name = filenames[0]
        return os.path.join(path, crires_name)


def load_spectrum(name):
    data = fits.getdata(name)
    hdr = fits.getheader(name)
    # Turn into Spectrum
    # Check for telluric corrected column
    # spectrum = Spectrum(xaxis=data["wavelength"], flux=data["Extracted_DRACS"],
    #                    calibrated=True, header=hdr)
    spectrum = Spectrum(xaxis=data["wavelength"], flux=data["Corrected_DRACS"],
                        calibrated=True, header=hdr)
    return spectrum


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
def convolve_models(models, R, chip_limits=None):
        """ Convolve all model spectra to resolution R.
        This prevents multiple convolution at the same resolution.

        inputs:
        models: list, tuple of spectum objects

        returns:
        new_models: tuple of the convovled spectral models
        """
        new_models = []
        for model in models:
            convovled_model = apply_convolution(model, R,
                                                chip_limits=chip_limits)
            new_models.append(convovled_model)
        return tuple(new_models)


# TO find why answer is all nans
def alpha_model2(alpha, rv, host, companion, limits, new_x=None):
    """ Entangled spectrum model.
    inputs:
    spectrum_1
    spectrum_2
    alpha
    rv - rv offset of spec2
    xrange = location of points to return for spectrum. e.g. observation.xaxis.
    should find better name.

    returns:
    Spectrum object
    """
    # this copy solved my nan issue.
    companion = copy.copy(companion)
    host = copy.copy(host)
    if np.all(np.isnan(companion.flux)):
        print("companion spectrum is all Nans before RV shift")
    if np.all(np.isnan(host.flux)):
        print("Host spectrum is all Nans before combine")
    companion.doppler_shift(rv)
    if np.all(np.isnan(companion.flux)):
        print("companion spectrum is all Nans after RV shift")
    combined = combine_spectra(host, companion, alpha)

    if np.all(np.isnan(combined.flux)):
        print("Combined spectrum is all Nans before interpolation")

    if np.any(new_x):
        # print(new_x)
        # combined.spline_interpolate_to(new_x)
        combined.interpolate1d_to(new_x)
    if np.all(np.isnan(combined.flux)):
        print("Combined spectrum is all Nans after interpolation")
    combined.wav_select(limits[0], limits[1])
    # observation.wav_select(2100, 2200)
    if np.all(np.isnan(combined.flux)):
        print("Combined spectrum is all Nans after wav_select")

    return combined



def main():
    """ """
    star = "HD30501"
    obs_num = "1"
    chip = 1
    obs_name = select_observation(star, obs_num, chip)

    # Load observation
    observed_spectra = load_spectrum(obs_name)


    # Load models
    (w_mod, I_star, I_bdmod, hdr_star, hdr_bd) = load_PHOENIX_hd30501(limits=[2100, 2200], normalize=True)

    obs_resolution = crires_resolution(observed_spectra.header)

    host_spectrum_model = Spectrum(flux=I_star, xaxis=w_mod, calibrated=True, header=hdr_star)
    companion_spectrum_model = Spectrum(flux=I_bdmod, xaxis=w_mod, calibrated=True, header=hdr_bd)

    # Convolve models to resolution of instrument
    host_spectrum_model, companion_spectrum_model = convolve_models((host_spectrum_model, companion_spectrum_model),
                                                                    obs_resolution, chip_limits=None)

    plot_obs_with_model(observed_spectra, host_spectrum_model, companion_spectrum_model, show=False, title="Before BERV Correction")

    # Berv Correct
    # Calculate the star RV
    parameters = {"HD30501":[23.710, 1703.1, 70.4, 0.741, 53851.5, 2073.6, 0.81, 90]}
    try:
        host_params =  parameters[star]
    except:
        raise ValueError("Parameters for {} are not in parameters list. Improve this.".format(star))
    host_params[1] = host_params[1] / 1000   # Convert K! to km/s
    host_params[2] = np.deg2rad(host_params[2]) # Omega needs to be in radians for ajplanet

    obs_time = observed_spectra.header["DATE-OBS"]
    print(obs_time, isinstance(obs_time, str))
    print(obs_time.replace("T"," ").split("."))
    jd = ephem.julian_date(obs_time.replace("T"," ").split(".")[0])
    Host_RV = pl_rv_array(jd, *host_params[0:6])[0]
    print("Host_RV", Host_RV, "km/s")

    offset = -Host_RV  # -22
    # offset = 0  # -22
    berv_corrected_observed_spectra = barycorr_crires_spectrum(observed_spectra, offset)  # Issue with air/vacuum
    # This introduces nans into the observed spectrum
    berv_corrected_observed_spectra.wav_select(*berv_corrected_observed_spectra.xaxis[
                                               np.isfinite(berv_corrected_observed_spectra.flux)][[0, -1]])
    # Shift to star RV

    plot_obs_with_model(berv_corrected_observed_spectra, host_spectrum_model, companion_spectrum_model, title="After BERV Correction")

    #print("\nWarning!!!\n BERV is not good have added a offset to get rest working\n")

    # Chisquared fitting
    alphas = 10**np.linspace(-4, 0.1, 100)
    RVs = np.arange(-50, 50, 0.05)

    # chisqr_store = np.empty((len(alphas), len(RVs)))
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
    obs_chisqr_parallel = parallel_chisqr(alphas, RVs, berv_corrected_observed_spectra, alpha_model2,
                                          (host_spectrum_model, companion_spectrum_model,
                                           observed_limits, berv_corrected_observed_spectra), n_jobs=n_jobs)
    # chisqr_parallel = parallel_chisqr(alphas, RVs, simlulated_obs, alpha_model, (org_star_spec,
    #                                   org_bd_spec, new_limits), n_jobs=4)

    end_time = dt.now()
    print("Time to run parallel chisquared = {}".format(end_time - start_time))
    # Plot memmap
    # plt.subplot(2, 1, 1)
    X, Y = np.meshgrid(RVs, alphas)
    plt.figure(figsize=(7, 7))
    plt.contourf(X, Y, np.log10(obs_chisqr_parallel.reshape(len(alphas), len(RVs))), 100)

    plt.title("Sigma chisquared")
    plt.ylabel("Flux ratio")
    plt.xlabel("RV (km/s)")
    plt.show()


    # Locate minimum and plot resulting model next to observation

    def find_min_chisquared(X, Y, Z):
        """ """
        min_loc = np.argmin(Z)
        print("min location", min_loc)

        X_sol = X.ravel()[min_loc]
        Y_sol = Y.ravel()[min_loc]
        Z_sol = Z.ravel()[min_loc]
        return X_sol, Y_sol, Z_sol, min_loc

    rv_solution, alpha_solution, min_chisqr, min_loc = find_min_chisquared(X, Y, obs_chisqr_parallel)
    print("Minium Chisqr value {2}\n RV sol = {0}\nAlpha Sol = {1}".format(rv_solution, alpha_solution, min_chisqr))

    Solution_model = alpha_model2(alpha_solution, rv_solution, host_spectrum_model, companion_spectrum_model,
                                  observed_limits)
    # alpha_model2(alpha, rv, host, companion, limits, new_x=None):

    plt.plot(Solution_model.xaxis, Solution_model.flux, label="Min chisqr solution")
    plt.plot(berv_corrected_observed_spectra.xaxis, berv_corrected_observed_spectra.flux, label="Observation")
    plt.legend(loc=0)
    plt.show()


    # Dump the results into a pickle file
    pickle_name = "Chisqr_results_{0}_{1}_chip_{2}.pickle".format(star, obs_num, chip)
    with open(os.path.join(path, pickle_name), "wb") as f:
        """Pickle all the necessary parameters to store
        """
        pickle.dump((RVs, alphas, berv_corrected_observed_spectra, host_spectrum_model, companion_spectrum_model,
                    rv_solution, alpha_solution, min_chisqr, min_loc, Solution_model), f)



if __name__ == "__main__":
    main()
