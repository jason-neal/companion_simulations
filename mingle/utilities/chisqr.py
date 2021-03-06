"""Chi-squared calculation."""

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import chi2


def chi_squared(observed, expected, error=None, axis=0):
    """Calculate chi squared.

    Same result as as scipy.stats.chisquare
    """
    if np.any(error):
        chisqr = np.sum((observed - expected) ** 2 / error ** 2, axis=axis)
    else:
        # Equal to scipy.stats.chisquare(observed, expected).statistic
        chisqr = np.sum((observed - expected) ** 2 / expected, axis=axis)  # identical to scipy
    return chisqr


def reduced_chi_squared(chi_squared, N, P):
    """
    :param chi_squared: Chi squared value
    :param N: (int) number of observations
    :param P: (int) number of important parameters
    :return: Reduced Chi squared
    """
    return chi_squared / (N - P)


def spectrum_chisqr(spectrum_1, spectrum_2, error=None):
    """Chi squared for spectrum objects."""
    # Spectrum wrapper for chisquare
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
        return chi2
    else:
        print("Spectrum_1", len(spectrum_1))
        print("Spectrum_2", len(spectrum_2))

        raise Exception("TODO: make xaxis equal in chisquare of spectrum")


def model_chisqr_wrapper(spectrum_1, model, params, error=None):
    """Evaluate model and call chisquare."""
    evaluated_model = model(*params)  # unpack parameters

    if np.all(np.isnan(evaluated_model.flux)):
        raise Exception("Evaluated model is all Nans")

    return spectrum_chisqr(spectrum_1, evaluated_model, error=error)


def parallel_chisqr(iter1, iter2, observation, model_func, model_params, n_jobs=1):
    """Parallel chisquared calculation with two iterators."""
    grid = Parallel(n_jobs=n_jobs)(delayed(model_chisqr_wrapper)(observation,
                                                                 model_func, (a, b, *model_params))
                                   for a in iter1 for b in iter2)
    return np.asarray(grid)


def chi2_at_sigma(sigma, dof=1):
    """Use inverse survival function to calculate the chi2 value for significances.

    Updated values from https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function
    """
    sigma_percent = {0: 0, 1: 0.682689492137, 2: 0.954499736104, 3: 0.997300203937,
                     4: 0.999936657516, 5: 0.999999426697, 6: 0.999999998027}
    p = 1 - sigma_percent[sigma]  # precentage
    return chi2(dof).isf(p)