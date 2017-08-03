"""Chisquared calculation."""

import numpy as np
from numba import jit
from joblib import Parallel, delayed


@jit
def chi_squared(observed, expected, error=None, axis=0):
    """Calculate chi squared.

    Same result as as scipy.stats.chisquare
    """
    if np.any(error):
        chisqr = np.sum((observed - expected)**2 / error**2, axis=axis)
    else:
        # chisqr = np.sum((observed-expected)**2)
        chisqr = np.sum((observed - expected)**2 / expected, axis=axis)
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
        chi2 = chi_squared(spectrum_1.flux, spectrum_2.flux, error=error)

        if np.isnan(chi2):
            print(" Nan chisqr")
            # print(spectrum_1.xaxis, spectrum_1.flux, spectrum_2.xaxis, spectrum_2.flux)
        return chi2
    else:
        print("Spectrum_1", len(spectrum_1))
        print("Spectrum_2", len(spectrum_2))

        raise Exception("TODO: make xaxis equal in chisquare of spectrum")


def model_chisqr_wrapper(spectrum_1, model, params, error=None):
    """Evaluate model and call chisquare."""
    # print("params for model", params)
    # params = copy.copy(params)
    evaluated_model = model(*params)  # unpack parameters

    if np.all(np.isnan(evaluated_model.flux)):
        raise Exception("Evaluated model is all Nans")

    return spectrum_chisqr(spectrum_1, evaluated_model, error=error)


def parallel_chisqr(iter1, iter2, observation, model_func, model_params, n_jobs=1):
    """Parallel chisqr calculation with two iterables."""
    grid = Parallel(n_jobs=n_jobs)(delayed(model_chisqr_wrapper)(observation,
                                                                 model_func, (a, b, *model_params))
                                   for a in iter1 for b in iter2)
    return np.asarray(grid)
