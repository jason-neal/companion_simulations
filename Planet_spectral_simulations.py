#!/usr/bin/env python
# -*- coding: utf8 -*-

# Spectrum Simulations
# To determine the recovery of planetary spectra.

# imports
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from spectrum_overload import Spectrum
import copy
import os
os.path
from todcor import todcor
from todcor import create_cross_correlations

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
        # Add
        planet.interpolate1d_to(star)
        combined_spectrum = star + (planet*alpha)
        #print("\nWarning! The axis are not equal. Fix this to procede\n")
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

def load_PHOENIX_hd30501():
    """ Load in the phoenix spectra of HD30501 and HD30501b

    Returns:
    w_mod : Wavelength in nm
    I_star : stellar intensity
    I_bd : companion intensity
    hdr_star : Star header
    hdr_bd : Companion header
    """

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

    return w_mod, I_star, I_bdmod, hdr_star, hdr_bd

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
    star_spec.wav_select(2100, 2200)
    bd_spec.wav_select(2100, 2200)
    # star_spec.wav_select(100, 3000)
    # bd_spec.wav_select(100, 3000)

    star_spec = simple_normalization(star_spec)
    bd_spec = simple_normalization(bd_spec)

    # Loop over different things
    print(star_spec.xaxis)
    print(bd_spec.xaxis)
    combined = combine_spectra(star_spec, bd_spec, alpha = 0.01)
    combined2 = combine_spectra(star_spec, bd_spec, alpha = 0.1)

    plotter(bd_spec, label="HD30501b")
    plt.title("Pheonix Spectra HD30501A and HD30501b")
    plt.ylabel("Flux [ergs/s/cm^2/cm]")
    #plt.ylabel("Normalized Flux")
    plt.xlabel("Wavelength [nm]")
    plotter(star_spec + 1, label="HD30501A")

    #plotter(combined2)
    plotter(combined + 1, label="Combined", show=True)

    # Why does this plot things twice???
    # Add rv attributes to spectrum for TODCOR  # Don't yet know what the parameters are though
    #bd_spec.rv = 10
    #star_spec.rv = 10

    RV_shift = np.arange(0, 20, 3)
    alpha = 0.005
    #alpha = 0.01
    for RV in RV_shift:
        """ """
        bd_shifted = copy.copy(bd_spec)
        # RV shift BD spectra
        bd_shifted.doppler_shift(RV)

        combined_spec = combine_spectra(star_spec, bd_shifted, alpha=alpha)

        combined_spec.wav_select(2158, 2159)
        plotter(combined_spec + (RV/200), label="RV shift={} km/s".format(RV))

    plt.title("Combined spectra (star + alpha*planet), alpha = {}".format(alpha))
    plt.legend(loc=0)#, bbox_to_anchor=(1.4, 0.9), ncol=1, fancybox=True, shadow=True)
    plt.ylabel("Norm Flux")
    plt.xlabel("Wavelength (nm)")
    plt.show()

    # Try run todoc stuff

    # Probably need to change my formating into ajriddles format to use his code.
    #ccf1,ccf2,ccf12,images = create_cross_correlations(combined2, star_spec, bd_spec)

    #plt.plot(ccf1)
    #plt.plot(ccf2)
    #plt.plot(ccf12)
    #plt.show()
    #pshift = 5     #????
    #sshift = 0    #????
    #cps_R,cps_m,cps_fits,cps_vp,cps_vs,cps_cntr = todcor(ccf1,ccf2,ccf12,pshift,sshift,images)

def plotter(spectra, label=None, show=False):
    """ """
    plt.plot(spectra.xaxis, spectra.flux, label=label)
    if label:
        plt.legend(loc=0, bbox_to_anchor=(1.4, 0.9), ncol=1, fancybox=True, shadow=True)
    if show:
        plt.show()

main()


if __name__ == "__main__":
    main()
