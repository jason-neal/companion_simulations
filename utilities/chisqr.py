"""Chisquared calculation."""

import numpy as np
from numba import jit


@jit
def chi_squared(observed, expected, error=None):
    """Calculate chi squared.

    Same result as as scipy.stats.chisquare
    """
    if np.any(error):
        chisqr = np.sum((observed - expected)**2 / error**2)
    else:
        # chisqr = np.sum((observed-expected)**2)
        chisqr = np.sum((observed - expected)**2 / expected)
        # When divided by expected the result is identical to scipy
    return chisqr


def spectrum_chisqr(spectrum_1, spectrum_2, error=None):
    """Chi squared for specturm objects."""
    # Spectrum wrapper for chissquare
    # make sure xaxis is the Same
    nan_number = np.sum(np.isnan(spectrum_1.flux))
    if nan_number:
        print("There are {} nans in spectrum_1".format(nan_number))

    if np.all(np.isnan(spectrum_2.flux)):
        print("spectrum 2 is all nans")

    if np.all(spectrum_1.xaxis == spectrum_2.xaxis):
        c2 = chi_squared(spectrum_1.flux, spectrum_2.flux, error=error)

        if np.isnan(c2):
            print(" Nan chisqr")
            # print(spectrum_1.xaxis, spectrum_1.flux, spectrum_2.xaxis, spectrum_2.flux)
        return c2
    else:
        print("Spectrum_1", len(spectrum_1))
        print("Spectrum_2", len(spectrum_2))

        raise Exception("TODO: make xaxis equal in chisquare of spectrum")
