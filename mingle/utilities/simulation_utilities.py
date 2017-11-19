#!/usr/bin/env python
"""Simulation Utilities."""

# File to contain function necessary for the chi_square simulations

import copy

import numpy as np


def add_noise(flux, snr):
    """Using the formulation mu/sigma from wikipedia.

    Applies noise based on the flux at ech pixel.
    """
    sigma = flux / snr
    # Add normal distributed noise at the snr level.
    noisy_flux = flux + np.random.normal(0, sigma)
    return noisy_flux


def combine_spectra(star, planet, alpha):
    """Combine the Spectrum objects "star" and "planet".

    Strength ratio of alpha
    spec = star + planet * alpha

    """
    star = copy.copy(star)
    planet = copy.copy(planet)

    if np.all(star.xaxis == planet.xaxis):  # make sure wavelengths even first
        pass
    else:
        planet.interpolate1d_to(star)
    # combined_spectrum = star + (planet*alpha)
    # Combined spectra with proper normalization
    norm_factor = 1 / (1 + alpha)
    combined_spectrum = (star + (planet * alpha)) * norm_factor

    return combined_spectrum


def spec_max_delta(obs_spec, rvs, gammas):
    """Calculate max doppler shift of a spectrum."""
    return max_delta(obs_spec.xaxis, rvs, gammas)


def max_delta(wavelength, rvs, gammas):
    """Calculate max doppler shift.

    Given a spectrum, and some doppler shifts, find the wavelength limit
    to have full coverage without much wastage computations.

    # Currently set at 2*delta.
    """
    check_inputs(rvs)
    check_inputs(gammas)

    shift_max = np.max(np.abs(rvs)) + np.max(np.abs(gammas))

    obs_limits = np.array([np.min(wavelength), np.max(wavelength)])

    delta = [lim * shift_max / 299792.458 for lim in obs_limits]

    return 2 * round(max(delta), 3)


def check_inputs(var):
    """Turn inputs into numpy arrays.

    Defaults to zero if None.
    """
    if (var is None) or ("None" in str(var)):
        var = np.array([0])
    elif isinstance(var, (np.float, np.int)):
        var = np.asarray([var], dtype=np.float32)

    if len(var) == 0:  # Empty sequence
        raise ValueError("Empty variable vector. Check config.yaml\n"
                         "var = {0}".format(var))
    return var
