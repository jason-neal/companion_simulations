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
        # When divided by exted the result is identical to scipy
    return chisqr


@jit
def alternate_chi_squared(observed, expected, error=None):
    """Calculate chi squared.

    Same result as as scipy.stats.chisquare
    """
    if error:
        chisqr = np.sum((observed - expected)**2 / observed)
    else:
        # chisqr = np.sum((observed-expected)**2)
        chisqr = np.sum((observed - expected)**2 / expected)
        # When divided by exted the result is identical to scipy
    return chisqr
