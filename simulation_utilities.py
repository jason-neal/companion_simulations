#!/usr/bin/env python
# Simulation Utilities

# File to contain function necessary for the chi_square simulations
from __future__ import division, print_function
import numpy as np

def add_noise(flux, SNR):
    "Using the formulation mu/sigma from wikipedia"
    sigma = flux / SNR
    # Add normal distributed noise at the SNR level.
    noisey_flux = flux + np.random.normal(0, sigma)
    return noisey_flux
