"""best_host_model.py.

Jason Neal
2nd January 2017

Compare observed spectra to many different phoenix-aces spectral models to find which matches one is best fit by itself.

Need to determine the best RV offset to apply to the spectra.
Possibly by sampling a few and then taking average value (they should be all
the same I would think unless the lines changed dramatically).

"""

import logging
import os
import sys
from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from logutils import BraceMessage as __
from spectrum_overload import Spectrum

import simulators
from mingle.utilities.chisqr import chi_squared
from mingle.utilities.crires_utilities import (barycorr_crires_spectrum,
                                               crires_resolution)
from mingle.utilities.debug_utils import pv
from mingle.utilities.model_convolution import convolve_models
from mingle.utilities.param_file import load_param_file
from mingle.utilities.phoenix_utils import find_phoenix_model_names
from mingle.utilities.phoenix_utils import (load_phoenix_spectrum,
                                            phoenix_name_from_params, spec_local_norm)
from mingle.utilities.spectrum_utils import load_spectrum, select_observation
from mingle.utilities.xcorr import xcorr_peak

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

model_base_dir = (simulators.starfish_grid["raw_path"])
wav_dir = simulators.starfish_grid["raw_path"]


def main():
    """Main function."""
    star = "HD30501"
    host_parameters = load_param_file(star)
    obsnum = 1
    chip = 1
    obs_name = select_observation(star, obsnum, chip)

    # Load observation
    # uncorrected_spectra = load_spectrum(obs_name)
    observed_spectra = load_spectrum(obs_name)
    _observed_spectra = barycorr_crires_spectrum(observed_spectra, extra_offset=None)
    observed_spectra.flux /= 1.02

    obs_resolution = crires_resolution(observed_spectra.header)

    wav_model = fits.getdata(os.path.join(wav_dir, "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"))
    wav_model /= 10  # turn into nm
    logging.debug(__("Phoenix wav_model = {0}", wav_model))

    closest_model = phoenix_name_from_params(model_base_dir, host_parameters)
    original_model = "Z-0.0/lte05200-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    logging.debug(__("closest_model {0}", closest_model))
    logging.debug(__("original_model {0}", original_model))

    # Function to find the good models I need
    models = find_phoenix_model_names(model_base_dir, original_model)
    if isinstance(models, list):
        logging.debug(__("Number of close models returned {0}", len(models)))

    model_chisqr_vals = np.empty_like(models)
    model_xcorr_vals = np.empty_like(models)
    model_xcorr_rv_vals = np.empty_like(models)

    for ii, model_name in enumerate(models):
        mod_flux = fits.getdata(model_name)
        mod_header = fits.getheader(model_name)
        mod_spectrum = Spectrum(xaxis=wav_model, flux=mod_flux, header=mod_header, calibrated=True)

        # Normalize Phoenix Spectrum
        # mod_spectrum.wav_select(2080, 2200)  # limits for simple normalization
        mod_spectrum.wav_select(2105, 2165)  # limits for simple normalization
        # norm_mod_spectrum = simple_normalization(mod_spectrum)
        norm_mod_spectrum = spec_local_norm(mod_spectrum, plot=False)

        # Wav select
        norm_mod_spectrum.wav_select(np.min(observed_spectra.xaxis) - 5,
                                     np.max(observed_spectra.xaxis) + 5)  # +- 5nm of obs for convolution

        # Convolve to resolution of instrument
        conv_mod_spectrum = convolve_models([norm_mod_spectrum], obs_resolution, chip_limits=None)[0]

        # Find crosscorrelation RV
        # # Should run though all models and find best rv to apply uniformly
        rvoffset, cc_max = xcorr_peak(observed_spectra, conv_mod_spectrum, plot=False)

        # Interpolate to obs
        conv_mod_spectrum.spline_interpolate_to(observed_spectra)
        # conv_mod_spectrum.interpolate1d_to(observed_spectra)
        model_chi_val = chi_squared(observed_spectra.flux, conv_mod_spectrum.flux)

        # argmax = np.argmax(cc_max)
        model_chisqr_vals[ii] = model_chi_val
        model_xcorr_vals[ii] = cc_max
        model_xcorr_rv_vals[ii] = rvoffset

    logging.debug(pv("model_chisqr_vals"))
    logging.debug(pv("model_xcorr_vals"))
    chisqr_argmin_indx = np.argmin(model_chisqr_vals)
    xcorr_argmax_indx = np.argmax(model_xcorr_vals)

    logging.debug(pv("chisqr_argmin_indx"))
    logging.debug(pv("xcorr_argmax_indx"))

    logging.debug(pv("model_chisqr_vals"))
    print("Minimum  Chisqr value =", model_chisqr_vals[chisqr_argmin_indx])  # , min(model_chisqr_vals)
    print("Chisqr at max correlation value", model_chisqr_vals[chisqr_argmin_indx])

    print("model_xcorr_vals = {}".format(model_xcorr_vals))
    print("Maximum Xcorr value =", model_xcorr_vals[xcorr_argmax_indx])  # , max(model_xcorr_vals)
    print("Xcorr at min Chiqsr", model_xcorr_vals[chisqr_argmin_indx])

    logging.debug(pv("model_xcorr_rv_vals"))
    print("RV at max xcorr =", model_xcorr_rv_vals[xcorr_argmax_indx])
    # print("Median RV val =", np.median(model_xcorr_rv_vals))
    print(pv("model_xcorr_rv_vals[chisqr_argmin_indx]"))
    # print(pv("sp.stats.mode(np.around(model_xcorr_rv_vals))"))

    print("Max Correlation model = ", models[xcorr_argmax_indx].split("/")[-2:])
    print("Min Chisqr model = ", models[chisqr_argmin_indx].split("/")[-2:])

    limits = [2110, 2160]
    best_model = models[chisqr_argmin_indx]
    best_model_spec = load_phoenix_spectrum(best_model, limits=limits, normalize=True)
    best_model_spec = convolve_models([best_model_spec], obs_resolution, chip_limits=None)[0]

    best_xcorr_model = models[xcorr_argmax_indx]
    best_xcorr_model_spec = load_phoenix_spectrum(best_xcorr_model, limits=limits, normalize=True)
    best_xcorr_model_spec = convolve_models([best_xcorr_model_spec], obs_resolution, chip_limits=None)[0]

    close_model_spec = load_phoenix_spectrum(closest_model[0], limits=limits, normalize=True)
    close_model_spec = convolve_models([close_model_spec], obs_resolution, chip_limits=None)[0]

    plt.plot(observed_spectra.xaxis, observed_spectra.flux, label="Observations")
    plt.plot(best_model_spec.xaxis, best_model_spec.flux, label="Best Model")
    plt.plot(best_xcorr_model_spec.xaxis, best_xcorr_model_spec.flux, label="Best xcorr Model")
    plt.plot(close_model_spec.xaxis, close_model_spec.flux, label="Close Model")
    plt.legend()
    plt.xlim(*limits)
    plt.show()
    print("After plot")


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
