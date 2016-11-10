# #!/usr/bin python


# Chi square of actual data observation

# Jason Neal November 2016


import os
import numpy as np
import copy
import matplotlib.pyplot as plt
from Get_filenames import get_filenames
from scipy.stats import chisquare
from astropy.io import fits
from spectrum_overload.Spectrum import Spectrum
from Planet_spectral_simulations import combine_spectra
from Planet_spectral_simulations import load_PHOENIX_hd30501


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
# from Get_filenames import get_filenames
# chip = 1
# obs_num = 1
        path = "/home/jneal/Phd/data/Crires/BDs-DRACS/HD30501-{}/Combined_Nods".format(obs_num)
        filenames = get_filenames(path, "CRIRE.*wavecal.tellcorr.fits","*_{}.nod.ms.*".format(chip))
# print(filenames)
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


def main():
    """ """

    # Load observation
    observed_spectra = load_spectrum()
    # load models
    w_mod, I_star, I_bdmod, hdr_star, hdr_bd =
        load_PHOENIX_hd30501(limits=[2100,2200], normalize=True)

    star_spec = Spectrum(flux=I_star, xaxis=w_mod, header=hdr_star)
    bd_spec = Spectrum(flux=I_bdmod, xaxis=w_mod, header=hdr_bd)

    plot_obs_with_model(observed_spectra, star_spec, bd_spec)

    # Prepare data / wavelength select,
    # Shift to star RV

    # Chisquared fitting
    alphas = 10**np.linspace(-7, -0.3, 100)
    RVs = np.arange(15, 40, 0.1)
    chisqr_store = np.empty((len(alphas), len(RVs)))

    for i, alpha in enumerate(alphas):
        for j, RV in enumerate(RVs):
            # print("RV", RV, "alpha", alpha)

            # Generate model for this RV and alhpa
            planet_shifted = copy.copy(bd_spec)
            planet_shifted.doppler_shift(RV)
            model = combine_spectra(star_spec, planet_shifted, alpha)
            model.wav_select(2100, 2200)

            # Interpolate to observed_spectra

            # Try scipy chi_squared
            chisquared = chisquare(observed_spectra.flux, model.flux)

            chisqr_store[i, j] = chisquared.statistic
    # Save results
    pass


if __name__ == "__main__":
    main()
