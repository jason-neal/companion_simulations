"""Chi-squared calculation."""

from typing import Optional, Union

import numpy as np
from numpy import float64, ndarray
from scipy.stats import chi2
from spectrum_overload.spectrum import Spectrum


def chi_squared(observed: ndarray, expected: ndarray, error: Optional[ndarray] = None, axis: int = 0) -> Union[float64, ndarray]:
    """Calculate chi squared.

    Same result as as scipy.stats.chisquare
    """
    if np.any(error):
        chisqr = np.sum((observed - expected) ** 2 / error ** 2, axis=axis)
    else:
        # Equal to scipy.stats.chisquare(observed, expected).statistic
        chisqr = np.sum((observed - expected) ** 2 / expected, axis=axis)  # identical to scipy
    return chisqr


def reduced_chi_squared(chi_squared: Union[int, ndarray, float], N: int, P: int) -> Union[ndarray, float]:
    """
    :param chi_squared: Chi squared value
    :param N: (int) number of observations
    :param P: (int) number of important parameters
    :return: Reduced Chi squared
    """
    return chi_squared / (N - P)


def spectrum_chisqr(spectrum_1: Spectrum, spectrum_2: Spectrum, error: None = None) -> Union[float64, ndarray]:
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


def chi2_at_sigma(sigma: int, dof: int=1) -> float:
    """Use inverse survival function to calculate the chi2 value for significances.

    Updated values from https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function
    """
    sigma_percent = {0: 0, 1: 0.682689492137, 2: 0.954499736104, 3: 0.997300203937,
                     4: 0.999936657516, 5: 0.999999426697, 6: 0.999999998027}
    p = 1 - sigma_percent[sigma]  # Precentage
    return chi2(dof).isf(p)
