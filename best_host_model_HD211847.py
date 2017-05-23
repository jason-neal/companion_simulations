"""best_host_model.py

Jason Neal
2nd Janurary 2017

Compare observed spectra to many different phoenix-aces spectral models to find which matches one is best fit by itself.

Need to determine the best RV offset to apply to the spectra.
Possibly bu sampling a few and then taking average value (they should be all
the same I would think unless the lines changed dramatically).

"""
from __future__ import division, print_function
import os
import sys
import logging
import numpy as np
import scipy as sp
from tqdm import tqdm
from astropy.io import fits
import matplotlib.pyplot as plt
from datetime import datetime as dt
from utilities.debug_utils import pv
from utilities.chisqr import chi_squared
from spectrum_overload.Spectrum import Spectrum
from utilities.model_convolution import convolve_models
from utilities.phoenix_utils import find_phoenix_model_names2 as find_phoenix_model_names
from utilities.phoenix_utils import phoenix_name_from_params, load_normalized_phoenix_spectrum

from utilities.crires_utilities import crires_resolution, barycorr_crires_spectrum
# from utilities.simulation_utilities import combine_spectra
# from new_alpha_detect_limit_simulation import parallel_chisqr  # , alpha_model
# from utilities.crires_utilities import barycorr_crires_spectrum
# from Chisqr_of_observation import plot_obs_with_model, select_observation, load_spectrum
from Chisqr_of_observation import select_observation, load_spectrum
from utilities.phoenix_utils import spec_local_norm
from utilities.param_file import parse_paramfile

from utilities.phoenix_utils import closest_model_params, generate_close_params, load_starfish_spectrum
# from Get_filenames import get_filenames
# sys.path.append("/home/jneal/Phd/Codes/equanimous-octo-tribble/Convolution")

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')
debug = logging.debug

model_base_dir = ("/home/jneal/Phd/data/PHOENIX-ALL/PHOENIX/")
wav_dir = "/home/jneal/Phd/data/PHOENIX-ALL/PHOENIX/"


def xcorr_peak(spectrum, model, plot=False):
    """Find RV offset between a spectrum and a model using pyastronomy.

    Parameters
    ----------
    spectrum: Spectrum
       Target Spectrum object.
    model: Spectrum
        Template Specturm object.

    Returns
    -------
    rv_max: float
        Radial velocity vlaue corresponding to maximum correlation.
    cc_max: float
        Cross-correlation value corresponding to maximum correlation.
    """
    rv, cc = spectrum.crosscorrRV(model, rvmin=-60., rvmax=60.0, drv=0.1,
                                  mode='doppler', skipedge=50)  # Specturm method

    maxind = np.argmax(cc)
    rv_max, cc_max = rv[maxind], cc[maxind]

    # debug("Cross-correlation function is maximized at dRV = {} km/s".format(rv_max))

    if plot:
        plt.subplot(211)
        plt.plot(spectrum.xaxis, spectrum.flux, label="Target")
        plt.plot(model.xaxis, model.flux, label="Model")
        plt.legend()
        plt.title("Spectra")

        plt.subplot(212)
        plt.plot(rv, cc)
        plt.plot(rv_max, cc_max, "o")
        plt.title("Cross correlation plot")
        plt.show()
    return float(rv[maxind]), float(cc[maxind])


def main():
    """Main function."""
    star = "HD211847"
    param_file = "/home/jneal/Phd/data/parameter_files/{}_params.txt".format(star)
    host_parameters = parse_paramfile(param_file, path=None)
    host_params = [host_parameters["teff"], host_parameters["logg"], host_parameters["fe_h"]]
    comp_params = [host_parameters["comp_teff"], host_parameters["logg"], host_parameters["fe_h"]]

    obs_num = 2
    chip = 4
    obs_name = select_observation(star, obs_num, chip)

    # Load observation
    uncorrected_spectra = load_spectrum(obs_name)
    observed_spectra = load_spectrum(obs_name)
    observed_spectra = barycorr_crires_spectrum(observed_spectra, -22)
    observed_spectra.flux /= 1.02

    obs_resolution = crires_resolution(observed_spectra.header)

    # Mask out bad portion of observed spectra ## HACK
    if chip == 4:
        # Ignore first 40 pixels
        observed_spectra.wav_select(observed_spectra.xaxis[40], observed_spectra.xaxis[-1])

    wav_model = fits.getdata(os.path.join(wav_dir, "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"))
    wav_model /= 10   # turn into nm
    debug("Phoenix wav_model = {}".format(wav_model))

    closest_host_model_name = phoenix_name_from_params(model_base_dir, host_parameters)
    #closest_comp_model = phoenix_name_from_params(model_base_dir, comp_parameters)
    closest_host_model = closest_model_params(*host_params)   # unpack temp, logg, fe_h with *
    debug(pv("comp_params"))
    closest_comp_model = closest_model_params(*comp_params)

    original_model = "Z-0.0/lte05700-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    # debug("closest_host_model name {}".format(closest_host_model_name))
    # debug("original_model {}".format(original_model))
    debug(pv("closest_host_model"))
    debug(pv("closest_comp_model"))


    # Function to find the good models I need
    models = find_phoenix_model_names(model_base_dir, original_model)
    # Function to find the good models I need from parameters
    model_par_gen = generate_close_params(closest_host_model)
    model_pars = list(generate_close_params(closest_host_model))  # turn to list

    print(model_pars)

    # if isinstance(models, list):
    #    debug("Number of close models returned {}".format(len(models)))
    #    print(models)
    if isinstance(model_pars, list):
        debug("Number of close model_pars returned {}".format(len(model_pars)))

    model_chisqr_vals = np.empty_like(model_pars)
    model_xcorr_vals = np.empty_like(model_pars)
    model_xcorr_rv_vals = np.empty_like(model_pars)

    normalization_limits = [2105, 2185]   # samll as possible?
    # for ii, model_name in enumerate(models):
    # for ii, params in enumerate(tqdm(model_par_gen)):
    for ii, params in enumerate(tqdm(model_pars)):
        mod_spectrum = load_starfish_spectrum(params, limits=normalization_limits, hdr=True, normalize=True)
        # mod_flux = fits.getdata(model_name)
        # mod_header = fits.getheader(model_name)
        # mod_spectrum = Spectrum(xaxis=wav_model, flux=mod_flux, header=mod_header, calibrated=True)

        # # Normalize Phoenix Spectrum
        # mod_spectrum.wav_select(*normalization_limits)  # limits for simple normalization
        # # norm_mod_spectrum = simple_normalization(mod_spectrum)
        # norm_mod_spectrum = spec_local_norm(mod_spectrum, plot=False)

        # Wav select
        mod_spectrum.wav_select(np.min(observed_spectra.xaxis) - 5,
                                np.max(observed_spectra.xaxis) + 5)  # +- 5nm of obs for convolution

        # Convolve to resolution of instrument
        # Starfish is already normalized and convovled
        # conv_mod_spectrum = convolve_models([norm_mod_spectrum], obs_resolution, chip_limits=None)[0]
        # conv_mod_spectrum = mod_spectrum
        # plot_spectra(norm_mod_spectrum, conv_mod_spectrum)
        # plot_spectra(mod_spectrum, conv_mod_spectrum)
        # plot_spectra(observed_spectra, mod_spectrum)
        # debug(pv("conv_mod_spectrum"))
        # debug(pv("mod_spectrum"))
        # Find crosscorrelation RV
        # # Should run though all models and find best rv to apply uniformly
        # rvoffset, cc_max = xcorr_peak(observed_spectra, conv_mod_spectrum, plot=False)
        rvoffset, cc_max = xcorr_peak(observed_spectra, mod_spectrum, plot=False)

        # Interpolate to obs
        mod_spectrum.spline_interpolate_to(observed_spectra)
        # conv_mod_spectrum.interpolate1d_to(observed_spectra)
        model_chi_val = chi_squared(observed_spectra.flux, mod_spectrum.flux)

        # argmax = np.argmax(cc_max)
        model_chisqr_vals[ii] = model_chi_val
        model_xcorr_vals[ii] = cc_max
        model_xcorr_rv_vals[ii] = rvoffset

    debug(pv("model_chisqr_vals"))
    debug(pv("model_xcorr_vals"))
    chisqr_argmin_indx = np.argmin(model_chisqr_vals)
    xcorr_argmax_indx = np.argmax(model_xcorr_vals)

    debug(pv("chisqr_argmin_indx"))
    debug(pv("xcorr_argmax_indx"))

    debug(pv("model_chisqr_vals"))
    print("Minimum  Chisqr value =", model_chisqr_vals[chisqr_argmin_indx])  # , min(model_chisqr_vals)
    print("Chisqr at max correlation value", model_chisqr_vals[chisqr_argmin_indx])

    print("model_xcorr_vals = {}".format(model_xcorr_vals))
    print("Maximum Xcorr value =", model_xcorr_vals[xcorr_argmax_indx])  # , max(model_xcorr_vals)
    print("Xcorr at min Chiqsr", model_xcorr_vals[chisqr_argmin_indx])

    debug(pv("model_xcorr_rv_vals"))
    print("RV at max xcorr =", model_xcorr_rv_vals[xcorr_argmax_indx])
    # print("Meadian RV val =", np.median(model_xcorr_rv_vals))
    print(pv("model_xcorr_rv_vals[chisqr_argmin_indx]"))
    print(pv("sp.stats.mode(np.around(model_xcorr_rv_vals))"))

    # print("Max Correlation model = ", models[xcorr_argmax_indx].split("/")[-2:])
    # print("Min Chisqr model = ", models[chisqr_argmin_indx].split("/")[-2:])
    print("Max Correlation model = ", model_pars[xcorr_argmax_indx])
    print("Min Chisqr model = ", model_pars[chisqr_argmin_indx])

    limits = [2110, 2160]
    # best_model = models[chisqr_argmin_indx]
    # best_model_spec = load_normalized_phoenix_spectrum(best_model, limits=limits)
    # best_model_spec = convolve_models([best_model_spec], obs_resolution, chip_limits=None)[0]
    best_model_params = model_pars[chisqr_argmin_indx]
    best_model_spec = load_starfish_spectrum(best_model_params, limits=limits, normalize=True)

    best_xcorr_model_params = model_pars[xcorr_argmax_indx]
    best_xcorr_model_spec = load_starfish_spectrum(best_xcorr_model_params, limits=limits, normalize=True)
    # best_xcorr_model_spec = load_normalized_phoenix_spectrum(best_xcorr_model, limits=limits)
    # best_xcorr_model_spec = convolve_models([best_xcorr_model_spec], obs_resolution, chip_limits=None)[0]

    # close_model_spec = load_normalized_phoenix_spectrum(closest_model[0], limits=limits)
    # close_model_spec = convolve_models([close_model_spec], obs_resolution, chip_limits=None)[0]
    close_model_spec = load_starfish_spectrum(closest_model_params, limits=limits, normalize=True)


    plt.plot(observed_spectra.xaxis, observed_spectra.flux, label="Observations")
    plt.plot(best_model_spec.xaxis, best_model_spec.flux, label="Best Model")
    plt.plot(best_xcorr_model_spec.xaxis, best_xcorr_model_spec.flux, label="Best xcorr Model")
    plt.plot(close_model_spec.xaxis, close_model_spec.flux, label="Close Model")
    plt.legend()
    plt.xlim(*limits)
    plt.show()

    debug("After plot")


def plot_spectra(obs, model):
    """Plot two spectra."""
    plt.plot(obs.xaxis, obs.flux, label="obs")
    plt.plot(model.xaxis, model.flux, label="model")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    def time_func(func, *args, **kwargs):
        start = dt.now()
        print("Starting at: {}".format(start))
        result = func(*args, **kwargs)
        end = dt.now()
        print("Endded at: {}".format(end))
        print("Runtime: {}".format(end - start))
        return result

    sys.exit(time_func(main))