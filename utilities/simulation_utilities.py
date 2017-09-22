#!/usr/bin/env python
"""Simulation Utilities."""

# File to contain function necessary for the chi_square simulations
from __future__ import division, print_function

import copy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt


def add_noise(flux, snr):
    """Using the formulation mu/sigma from wikipedia.

    Applies noise based on the flux at ech pixel.
    """
    sigma = flux / snr
    # Add normal distributed noise at the snr level.
    noisey_flux = flux + np.random.normal(0, sigma)
    return noisey_flux


def spectrum_plotter(spectra, label=None, show=False):
    """Plot a Spectrum object."""
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
    """Combine the Spectrum objects "star" and "planet".

    Strength ratio of aplha
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


def max_delta(obs_spec, rvs, gammas):
    """Calcualte max doppler shift.

    Given a spectrum, and some doppler shifts, find the wavelength limit to have full coverage without much wastage computations.

    # Currently set at 2*delta.
    """
    shift_max = np.max(np.abs(rvs)) + np.max(np.abs(gammas))

    obs_limits = np.array([np.min(obs_spec.xaxis), np.max(obs_spec.xaxis)])

    for lim in obs_limits:
        delta = [lim * shift_max / 299792.458 for lim in obs_limits]

    return 2 * round(max(delta), 3)


def plot_spectra(obs, model):
    """Plot two spectra."""
    plt.plot(obs.xaxis, obs.flux, label="obs")
    plt.plot(model.xaxis, model.flux, label="model")
    plt.legend()
    plt.show()