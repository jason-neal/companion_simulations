
""" best_host_model.py
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
import copy
import ephem
import pickle
import numpy as np
from joblib import Memory
from astropy.io import fits
import multiprocess as mprocess
import matplotlib.pyplot as plt
from datetime import datetime as dt

from crires_utilities import crires_resolution
from crires_utilities import barycorr_crires_spectrum
from spectrum_overload.Spectrum import Spectrum
from simulation_utilities import combine_spectra
from Planet_spectral_simulations import load_PHOENIX_hd30501, simple_normalization

# from Get_filenames import get_filenames
sys.path.append("/home/jneal/Phd/Codes/equanimous-octo-tribble/Convolution")
from IP_multi_Convolution import IPconvolution

sys.path.append("/home/jneal/Phd/Codes/Phd-codes/Simulations")
from new_alpha_detect_limit_simulation import parallel_chisqr, chi_squared  # , alpha_model
from Chisqr_of_observation import plot_obs_with_model, select_observation

path = "/home/jneal/Phd/Codes/Phd-codes/Simulations/saves"  # save path
cachedir = os.path.join(path, "cache")  # save path
memory = Memory(cachedir=cachedir, verbose=0)

model_base_dir = "../../../data/fullphoenix/phoenix.astro.physik.uni-goettingen.de/HiResFITS/"


def RV_cross_corr(spectrum, model, plot=False):
    """ Find RV offset between a spectrum and a model using pyastronomy.
    """
    # Cross-correlation
    # from PyAstronomy example
    #
    # TAPAS is the "template" shifted to match Molecfit
    rv, cc = pyasl.crosscorrRV(spectrum.xaxis, spectrum.flux,
                               model.xaxis, model.flux, rvmin=-60.,
                               rvmax=60.0, drv=0.1, mode='doppler', skipedge=50)

    maxind = np.argmax(cc)
    print("Cross-correlation function is maximized at dRV = ", rv[maxind], " km/s")
    if plot:
        plt.plot(rv, cc)
        plt.title("Cross correlation plot")
        plt.show()
    return rv[maxind]


def find_phoenix_models(base_dir, original_model):
    """ Find other phoenix models with similar temp and metalicities.

    Returns list of model name strings"""
    # "lte05200-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    temp = int(original_model[4:8])
    logg = float(original_model[9:13])
    metals = float(original_model[14:17])

    new_temps = np.array(-400, -300, -200, -100, 0, 100, 200, 300, 400) + temp
    new_metals = np.array(-1, -0.5, 0, 0.5, 1) + metals
    new_loggs = np.array(-1, -0.5, 0, 0.5, 1) + logg

    # z = metalicities
    # "Z{new_metal}/lte0{new_temp}-{newlogg}-{new_metal}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    return close_models


def main():
    """ """
    star = "HD30501"
    obs_num = 1
    chip = 1
    obs_name = select_observation(star, obs_num, chip)

    # Load observation
    observed_spectra = load_spectrum(obs_name)
    obs_resolution = crires_resolution(observed_spectra.header)

    wav_model = fits.getdata(model_base_dir + "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
    wav_model /= 10   # turn into nm

    original_model = "lte05200-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"

    # Function to find the good models I need
    models = find_phoenix_models(model_base_dir, original_model)
    # models = ["", "", "", ""]
    model_chisqr_vals = np.empty_like(models)

    for ii, model_name in enumerate(models):
        mod_flux = fits.getdata(model_name)
        mod_header = fits.getheader(model_name)
        mod_spectrum = Spectrum(xaxis=wav_model, flux=mod_flux, header=mod_header, calibrated=True)

        # Normalize
        # Since just with high flux stars (not cool dwarfs) in this case the simple normalization might be enough.
        mod_spectrum.wav_select(2080, 2200)  # limits for simple normalization
        norm_mod_spectrum = simple_normalization(mod_spectrum)
        # norm_mod_spectrum = blackbody_normalization(mod_spectrum)

        # wav select
        norm_mod_spectrum.wav_select(np.min(observed_spectra.xaxis) - 5,
                                     np.max(observed_spectra.xaxis) + 5)  # +- 5nm of obs for convolution

        # Convolve to resolution of instrument
        conv_mod_spectrum = convolve_models(norm_mod_spectrum, obs_resolution, chip_limits=None)

        # Find crosscorrelation RV
        # # Should run though all models and find best rv to apply uniformly
        rvoffset = RV_cross_corr(observed_spectra, conv_mod_spectrum, plot=True)

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


if __name__ == "__main__":
    main()