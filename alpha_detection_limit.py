
# Test alpha variation at which cannot detect a planet

# Create a combined spectra with a planet at an alpha value.
# try and detect it by varying RV and alpha.
# At some stage the alpha will not vary when it becomes to small
# This will be the alpha detection limit.

# Maybe this is a wavelength dependant?

# The goal is to get something working and then try imporive the performance for complete simulations.

# Create the test spectra.
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from spectrum_overload import Spectrum
import copy
import os

from Planet_spectral_simulations import combine_spectra
from Planet_spectral_simulations import load_PHOENIX_hd30501



def main():

    w_mod, I_star, I_bdmod, hdr_star, hdr_bd = load_PHOENIX_hd30501()

    star_spec = Spectrum.Spectrum(xaxis=w_mod, flux=I_star, calibrated=True)
    bd_spec = Spectrum.Spectrum(xaxis=w_mod, flux=I_bdmod, calibrated=True)

    # Wavelength selection from 2100-2200nm
    star_spec.wav_select(2100, 2200)
    bd_spec.wav_select(2100, 2200)
    # star_spec.wav_select(100, 3000)
    # bd_spec.wav_select(100, 3000)

    # RV_shift bd_spec
    RV_val = 10
    planet_shifted = copy.copy(bd_spec)
    # RV shift BD spectra
    planet_shifted.doppler_shift(RV_val)
    Alpha = 0.1   # Vary this to determine detection limit

    # This is the signal to try and recover
    Aplha_Combine = combine_spectra(star_spec, planet_shifted, Alpha)


    alphas = 10**np.linspace(-3,-0.5, 10)
    #print(alphas)
    RVs = np.arange(8, 12, 1)
    for alpha in alphas:
        for RV in RVs:
            print("RV", RV, "alpha", alpha)











main()


if __name__ == "__main__":
    main()
