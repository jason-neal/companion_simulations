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
from astropy.io import fits
import matplotlib.pyplot as plt
from datetime import datetime as dt

from utilities.chisqr import chi_squared
from spectrum_overload.Spectrum import Spectrum
from utilities.model_convolution import convolve_models
from utilities.phoenix_utils import find_phoenix_models2 as find_phoenix_models
from utilities.phoenix_utils import phoenix_from_params
from utilities.crires_utilities import crires_resolution
# from utilities.simulation_utilities import combine_spectra
# from new_alpha_detect_limit_simulation import parallel_chisqr  # , alpha_model
# from utilities.crires_utilities import barycorr_crires_spectrum
# from Chisqr_of_observation import plot_obs_with_model, select_observation, load_spectrum
from Chisqr_of_observation import select_observation, load_spectrum
from utilities.phoenix_utils import spec_local_norm
from utilities.param_file import parse_paramfile
# from Get_filenames import get_filenames
# sys.path.append("/home/jneal/Phd/Codes/equanimous-octo-tribble/Convolution")
# from IP_multi_Convolution import IPconvolution

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

model_base_dir = "/home/jneal/Phd/data/fullphoenix/phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/"
wav_dir = "/home/jneal/Phd/data/fullphoenix/phoenix.astro.physik.uni-goettingen.de/HiResFITS/"


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

    print("Cross-correlation function is maximized at dRV = ", rv_max, " km/s")
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
    return rv[maxind], cc[maxind]


def main():
    """Main function."""
    star = "HD30501"
    param_file = "/home/jneal/Phd/data/parameter_files/{}_params.txt".format(star)
    host_parameters = parse_paramfile(param_file, path=None)
    obs_num = 1
    chip = 1
    obs_name = select_observation(star, obs_num, chip)

    # Load observation
    observed_spectra = load_spectrum(obs_name)
    obs_resolution = crires_resolution(observed_spectra.header)

    wav_model = fits.getdata(os.path.join(wav_dir, "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"))
    wav_model /= 10   # turn into nm
    logging.debug("Phoenix wav_model = {}".format(wav_model))

    closest_model = phoenix_from_params(model_base_dir, host_parameters)
    original_model = "Z-0.0/lte05200-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    logging.debug("closest_model {}".format(closest_model))
    logging.debug("original_model {}".format(original_model))

    # Function to find the good models I need
    models = find_phoenix_models(model_base_dir, original_model)
    if isinstance(models, list):
        logging.debug("Number of close models returned {}".format(len(models)))

    model_chisqr_vals = np.empty_like(models)
    model_xcorr_vals = np.empty_like(models)
    model_xcorr_rv_vals = np.empty_like(models)

    for ii, model_name in enumerate(models):
        mod_flux = fits.getdata(model_name)
        mod_header = fits.getheader(model_name)
        mod_spectrum = Spectrum(xaxis=wav_model, flux=mod_flux, header=mod_header, calibrated=True)

        # Normalize Phoenix Spectrum
        mod_spectrum.wav_select(2080, 2200)  # limits for simple normalization
        # norm_mod_spectrum = simple_normalization(mod_spectrum)
        norm_mod_spectrum = spec_local_norm(mod_spectrum, plot=True)

        # Wav select
        norm_mod_spectrum.wav_select(np.min(observed_spectra.xaxis) - 5,
                                     np.max(observed_spectra.xaxis) + 5)  # +- 5nm of obs for convolution

        # Convolve to resolution of instrument
        conv_mod_spectrum = convolve_models(norm_mod_spectrum, obs_resolution, chip_limits=None)

        # Find crosscorrelation RV
        # # Should run though all models and find best rv to apply uniformly
        rvoffset, cc_max = xcorr_peak(observed_spectra, conv_mod_spectrum, plot=True)

        # Interpolate to obs
        conv_mod_spectrum.spline_interpolate_to(observed_spectra)
        # conv_mod_spectrum.interpolate1d_to(observed_spectra)
        model_chi_val = chi_squared(observed_spectra.flux, conv_mod_spectrum.flux)

        model_chisqr_vals[ii] = model_chi_val

    print("chisqr vals", model_chisqr_vals)
    argmin_indx = np.argmin(model_chisqr_vals)
    print("chisqr argmin index ", argmin_indx)
    print("min chisqr =", model_chisqr_vals[argmin_indx])
    print("min chisqr model = ", models[argmin_indx])
    return 0


if __name__ == "__main__":
    def time_func(func, *args, **kwargs):
        start = dt.now()
        print("starting at {}".format(start))
        result = func(*args, **kwargs)
        end = dt.now()
        print("End at {}".format(end))
        print("Runtime {}".format(end - start))
        return result

    sys.exit(time_func(main))
