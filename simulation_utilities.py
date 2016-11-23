#!/usr/bin/env python
# Simulation Utilities

# File to contain function necessary for the chi_square simulations
from __future__ import division, print_function
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
