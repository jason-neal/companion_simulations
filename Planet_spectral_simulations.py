
# Spectrum Simulations
# To determine the recovery of planetary spectra.

# imports
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from spectrum_overload import Spectrum


def RV_shift():
    """ Doppler shift spectrum"""
    pass

def log_RV_shift():
    """ Doppler shift when log-linear spectrum is used"""
    pass

def add_noise(spectra, snr):
    """Add noise to spectrum at the level given by snr"""

def combine_spectra(star, planet, alpha):
    """"Combine the Spectrum objects "star" and "planet" with a strength ratio of aplha
    spec = star + planet * alpha

    """
    if np.all(star.xaxis == planet.xaxis):   # make sure wavelenghts even first
          # Even though spectrum overload does do it.
        combined_spectrum = star + (planet*alpha)
    else:
        print("Warning! The axis are not equal. Fix this to procede")
    return combined_spectrum

def simple_normalization(spectrum):
    """ Simple normalization of pheonix spectra """
    from astropy.modeling import models, fitting
    p1 = models.Polynomial1D(1)
    p1.c0 = spectrum.flux[0]
    p1.c1 = (spectrum.flux[-1]-spectrum.flux[0]) / (spectrum.xaxis[-1]-spectrum.xaxis[0])

    print("Printing p1", p1)
    pfit = fitting.LinearLSQFitter()
    new_model = pfit(p1, spectrum.xaxis, spectrum.flux)
    print(new_model)
    fit_norm = new_model(spectrum.xaxis)
    norm_spectrum = spectrum / fit_norm
    flux = norm_spectrum.flux

    # Normalization (use first 50 points below 1.2 as continuum)
    maxes = flux[(flux < 1.2)].argsort()[-50:][::-1]
    norm_spectrum = norm_spectrum / np.median(flux[maxes])

    return norm_spectrum
    # Split spectrum into a bunch, then fit along it along the top

    #length = len(spectrum)
    #indexes = range(0, length, 100)
    # This is bad coding I am just trying to get something together
    #maxes = []
    #for index1, index2 in zip(indexes[:-1], indexes[1:]):
#        section = np.argmax(spectrum[index1:index2])
#        m = argmax(section)
#        maxes.append(m)
#    print("maxes", maxes)
#    print("indixes", indexes)



def main():
    # Load in the pheonix spectra
    pathwave = "/home/jneal/Phd/data/phoenixmodels/" \
               "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
    bd_model = "/home/jneal/Phd/data/phoenixmodels/" \
               "HD30501b-lte02500-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    star_model = "/home/jneal/Phd/data/phoenixmodels/" \
                 "HD30501-lte05200-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"

    I_bdmod = fits.getdata(bd_model)
    I_star = fits.getdata(star_model)
    hdr_bd = fits.getheader(bd_model)
    hdr_star = fits.getheader(star_model)
    w_mod = fits.getdata(pathwave)

    w_mod /= 10   # turn into nm

    star_spec = Spectrum.Spectrum(xaxis=w_mod, flux=I_star, calibrated=True)
    bd_spec = Spectrum.Spectrum(xaxis=w_mod, flux=I_bdmod, calibrated=True)

    # Wavelength selection from 2100-2200nm
    star_spec.wav_select(2100,2200)
    bd_spec.wav_select(2100,2200)

    star_spec = simple_normalization(star_spec)
    bd_spec = simple_normalization(bd_spec)

    # Loop over different things
    print(star_spec.xaxis)
    print(bd_spec.xaxis)
    combined = combine_spectra(star_spec, bd_spec, alpha = 0.01)

    plotter(star_spec)
    plotter(bd_spec)
    plotter(combined, show=True)
    # why does this plot things twice???





def plotter(spectra, show=False):
    """ """
    plt.plot(spectra.xaxis, spectra.flux)
    if show:
        plt.show()

main()
if __name__ == "__main__":
    main()
