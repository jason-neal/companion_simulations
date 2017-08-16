"""Collection of phoenix model convolution and storage codes.

To remove duplication in all the different scripts.
"""
import copy
import os
import sys

import numpy as np
from joblib import Memory

from IP_multi_Convolution import ip_convolution
from spectrum_overload.Spectrum import Spectrum

sys.path.append("/home/jneal/Phd/Codes/equanimous-octo-tribble/Convolution")

cachedir = "/home/jneal/.simulation_cache"
if not os.path.exists(cachedir):
    os.makedirs(cachedir)
memory = Memory(cachedir=cachedir, verbose=0)


@memory.cache
def apply_convolution(model_spectrum, R=None, chip_limits=None):
    """Apply convolution to spectrum object."""
    if chip_limits is None:
        chip_limits = (np.min(model_spectrum.xaxis),
                       np.max(model_spectrum.xaxis))

    if R is None:
        return copy.copy(model_spectrum)
    else:
        ip_xaxis, ip_flux = ip_convolution(model_spectrum.xaxis[:],
                                           model_spectrum.flux[:], chip_limits,
                                           R, fwhm_lim=5.0, plot=False, progbar=True)

        new_model = Spectrum(xaxis=ip_xaxis, flux=ip_flux,
                             calibrated=model_spectrum.calibrated,
                             header=model_spectrum.header)

        return new_model


@memory.cache
def store_convolutions(spectrum, resolutions, chip_limits=None):
    """Convolve spectrum to many resolutions and store in a dict to retreive.

    This prevents performing multiple convolution at the same resolution.
    """
    d = dict()
    for resolution in resolutions:
        d[resolution] = apply_convolution(spectrum, resolution,
                                          chip_limits=chip_limits)
    return d


@memory.cache
def convolve_models(models, R, chip_limits=None):
    """Convolve all model spectra to resolution R.

    This prevents multiple convolution at the same resolution.

    inputs:
    models: list, tuple of spectum objects

    returns:
    new_models: tuple of the convovled spectral models
    """
    new_models = []
    for model in models:
        convovled_model = apply_convolution(model, R,
                                            chip_limits=chip_limits)
        new_models.append(convovled_model)
    return tuple(new_models)
