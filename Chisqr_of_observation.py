#!/usr/bin/env python


# Chi square of actual data observation

# Jason Neal November 2016

import os
import numpy as np
import copy
import matplotlib.pyplot as plt
from IP_multi_Convolution import IPconvolution
from Get_filenames import get_filenames
from scipy.stats import chisquare
from astropy.io import fits
from spectrum_overload.Spectrum import Spectrum
from Planet_spectral_simulations import combine_spectra
from Planet_spectral_simulations import load_PHOENIX_hd30501
from joblib import Memory
from joblib import Parallel, delayed


# First plot the observation with the model
def plot_obs_with_model(obs, model1, model2=None):
    """ Plot the obseved spectrum against the model to check that they are
    "compatiable"
    """
    plt.plot(obs.xaxis, obs.flux + 1, label="Observed")
    plt.plot(model1.xaxis, model1.flux + 1.2, label="model1")
    if model2:
        plt.plot(model2.xaxis, model2.flux, label="model2")
    plt.legend(loc=0)
    plt.show()


# I should already have these sorts of functions
def select_observation(obs_num, chip):
    """ Select the observation to load in

    inputs:
    obs_num: observation number
    chip: crires detetor chip number

    returns:
    crires_name: name of file
    """
    if str(chip) not in "1234":
        print("The Chip is not correct. It needs to be 1,2,3 or 4")
        raise Exception("Chip Error")
    else:

        path = "/home/jneal/Phd/data/Crires/BDs-DRACS/HD30501-", \
               "{}/Combined_Nods".format(obs_num)
        filenames = get_filenames(path, "CRIRE.*wavecal.tellcorr.fits",
                                  "*_{}.nod.ms.*".format(chip))

        crires_name = filenames[0]
        return os.path.join(path, crires_name)


def load_spectrum(name):
    data = fits.getdata(name)
    hdr = fits.getheader(name)
    # Turn into Spectrum
    # Check for telluric corrected column
    spectrum = Spectrum(xaxis=data["wavelength"], flux=data["Extracted_DRACS"],
                        calibrated=True, header=hdr)
    return spectrum


def berv_correct(spectrum):
    """ Berv Correct spectrum from header information """
    spectrum = copy.copy(spectrum)
    date = spectrum.header["Date"]
    # Calculate corrections
    # Need to find the notebook with this
    RV = 0

    spectrum.doppler_shift(RV)
    return spectrum


def set_crires_resolution(header):
    """ Set CRIRES resolution based on rule of thumb equation from the manual.
    Warning! The use of adpative optics is not checked for!!
    # This code has been copied from tapas xml request script.
    """
    instrument = header["INSTRUME"]

    slit_width = header["HIERARCH ESO INS SLIT1 WID"]
    if "CRIRES" in instrument:
        # print("Resolving Power\nUsing the rule of thumb equation from the
        # CRIRES manual. \nWarning! The use of adpative optics is not
        # checked for!!")
        R = 100000*0.2 / slit_width
        resolving_power = int(R)
        # print("Slit width was {0} inches.\n
        # Therefore the resolving_power is set = {1}".format(slit_width,
        # resolving_power))
    else:
        print("Instrument is not CRIRES")
    return resolving_power


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
            convovled_model = apply_convolution(spectrum, resolution,
                                                chip_limits=chip_limits)
        new_models.append(convovled_model)
        return tuple(new_models)


def main():
    """ """
    obs_num = 1
    chip = 1
    obs_name = select_observation(obs_num, chip)

    # Load observation
    observed_spectra = load_spectrum(obs_name)
    # load models
    (w_mod, I_star, I_bdmod, hdr_star,
        hdr_bd) = load_PHOENIX_hd30501(limits=[2100, 2200], normalize=True)

    obs_resolution = set_crires_resolution(observed_spectra.header)

    star_spec = Spectrum(flux=I_star, xaxis=w_mod, header=hdr_star)
    bd_spec = Spectrum(flux=I_bdmod, xaxis=w_mod, header=hdr_bd)

    # Convolve models to resolution of instrument
    star_spec, bd_spec = convolve_models((star_spec, bd_spec), obs_resolution,
                                         chip_limits=None)

    plot_obs_with_model(observed_spectra, star_spec, bd_spec)

    # Prepare data / wavelength select,
    # Berv Correct
    observed_spectra = berv_correct(observed_spectra)

    # Shift to star RV

    # Chisquared fitting
    alphas = 10**np.linspace(-7, -0.3, 100)
    RVs = np.arange(15, 40, 0.1)
    # chisqr_store = np.empty((len(alphas), len(RVs)))

    numProcs = None
    if numProcs is None:
        numProcs = mprocess.cpu_count() - 1

    # mprocPool = mprocess.Pool(processes=numProcs)
        timeInit = dt.now()
        X = np.meshgrid(alphas, RVs)
        map_filename = os.path.join(path, "obs_chisqr.memmap")
        chisqr_memmap = np.memmap(map_filename, dtype='float32', mode='w+',
                                  shape=X.shape)

        # args_generator = tqdm([[i, j, alpha, rv, sim_observation, host_model,
        #                        companion_model, chisqr_memmap]
        #                        for i, alpha in enumerate(alphas)
        #                        for j, rv in enumerate(RVs)])

        # mprocPool.map(wrapper_parallel_chisquare, args_generator)
        Parallel(n_jobs=numProcs)(delayed(parallel_chisquared)(i, j, alpha, rv,
                                  resolution, snr, sim_observation,
                                  convolved_star_models,
                                  convolved_planet_models, chisqr_memmap)
                                  for j, rv in enumerate(RVs)
                                  for i, alpha in enumerate(alphas))

        res_snr_chisqr_dict[resolution][snr] = np.copy(scipy_memmap)
        error_res_snr_chisqr_dict[resolution][snr] = np.copy(my_chisqr_memmap)

    # mprocPool.close()
    timeEnd = dt.now()


    # Incomplete after here
            # Generate model for this RV and alhpa
            planet_shifted = copy.copy(bd_spec)
            planet_shifted.doppler_shift(RV)
            model = combine_spectra(star_spec, planet_shifted, alpha)
            model.wav_select(2100, 2200)

            # Convovle to R50000
            chip_limits = [model.xaxis[0], model.xaxis[-1]]
            R = 50000
            model.xaxis, model.flux = IPconvolution(model.xaxis, model.flux,
                                                    chip_limits, R,
                                                    FWHM_lim=5.0, plot=True,
                                                    verbose=True)
            # Interpolate to observed_spectra

            # Try scipy chi_squared
            chisquared = chisquare(observed_spectra.flux, model.flux)

            chisqr_store[i, j] = chisquared.statistic
    # Save results
    pass







if __name__ == "__main__":
    main()
