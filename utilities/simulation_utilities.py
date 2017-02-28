#!/usr/bin/env python
# Simulation Utilities

# File to contain function necessary for the chi_square simulations
from __future__ import division, print_function
import copy
import numpy as np
import matplotlib.pyplot as plt


def add_noise(flux, SNR):
    "Using the formulation mu/sigma from wikipedia"
    sigma = flux / SNR
    # Add normal distributed noise at the SNR level.
    noisey_flux = flux + np.random.normal(0, sigma)
    return noisey_flux


def spectrum_plotter(spectra, label=None, show=False):
    """ Plot a Spectrum object """
    plt.figure()
    plt.plot(spectra.xaxis, spectra.flux, label=label)
    plt.ylabel("Flux")
    plt.xlabel("xaxis")
    if label:
        plt.legend(loc=0, bbox_to_anchor=(1.4, 0.9), ncol=1,
                   fancybox=True, shadow=True)
    if show:
        plt.show()


def combine_spectra(star, planet, alpha):
    """"Combine the Spectrum objects "star" and "planet" with a strength ratio of aplha
    spec = star + planet * alpha

    """
    star = copy.copy(star)
    planet = copy.copy(planet)

    if np.all(star.xaxis == planet.xaxis):   # make sure wavelenghts even first
        pass
    else:
        planet.interpolate1d_to(star)
    # combined_spectrum = star + (planet*alpha)
    # Combined spectra with proper normalization
    norm_factor = 1 / (1 + alpha)
    combined_spectrum = (star + (planet * alpha)) * norm_factor

    return combined_spectrum