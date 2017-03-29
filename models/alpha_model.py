"""Alpha combination models."""
import copy
import numpy as np
from utilities.simulation_utilities import combine_spectra


def alpha_model(alpha, rv, host, companion, limits, new_x=None):
    """Entangled spectrum model.

    inputs:
    spectrum_1
    spectrum_2
    alpha
    rv - rv offset of spec2
    xrange = location of points to return for spectrum. e.g. observation.xaxis.
    should find better name.

    returns:
    Spectrum object

    """
    # this copy solved my nan issue.
    companion = copy.copy(companion)
    host = copy.copy(host)

    companion.doppler_shift(rv)
    combined = combine_spectra(host, companion, alpha)

    if np.any(new_x):
        combined.spline_interpolate_to(new_x)
    combined.wav_select(limits[0], limits[1])
    # observation.wav_select(2100, 2200)

    return combined


# aplha model 2 is from Chisqr_of_obersvation.py
# TO find why answer is all nans
def alpha_model2(alpha, rv, host, companion, limits, new_x=None):
    """Entangled spectrum model.

    inputs:
    spectrum_1
    spectrum_2
    alpha
    rv - rv offset of spec2
    xrange = location of points to return for spectrum. e.g. observation.xaxis.
    should find better name.

    returns:
    Spectrum object
    """
    # this copy solved my nan issue.
    companion = copy.copy(companion)
    host = copy.copy(host)
    if np.all(np.isnan(companion.flux)):
        print("companion spectrum is all Nans before RV shift")
    if np.all(np.isnan(host.flux)):
        print("Host spectrum is all Nans before combine")
    companion.doppler_shift(rv)
    if np.all(np.isnan(companion.flux)):
        print("companion spectrum is all Nans after RV shift")
    combined = combine_spectra(host, companion, alpha)

    if np.all(np.isnan(combined.flux)):
        print("Combined spectrum is all Nans before interpolation")

    if np.any(new_x):
        # print(new_x)
        # combined.spline_interpolate_to(new_x)
        combined.interpolate1d_to(new_x)
    if np.all(np.isnan(combined.flux)):
        print("Combined spectrum is all Nans after interpolation")
    combined.wav_select(limits[0], limits[1])
    # observation.wav_select(2100, 2200)
    if np.all(np.isnan(combined.flux)):
        print("Combined spectrum is all Nans after wav_select")

    return combined


def double_shifted_alpha_model(alpha, rv1, rv2, host, companion, limits, new_x=None):
    """Entangled spectrum model.

    inputs:
    spectrum_1
    spectrum_2
    alpha
    rv - rv offset of spec2
    xrange = location of points to return for spectrum. e.g. observation.xaxis.
    should find better name.

    returns:
    Spectrum object

    """
    # this copy solved my nan issue.
    companion = copy.copy(companion)
    host = copy.copy(host)

    host.doppler_shift(rv1)
    companion.doppler_shift(rv2)

    combined = combine_spectra(host, companion, alpha)

    if np.any(new_x):
        combined.spline_interpolate_to(new_x)
    combined.wav_select(limits[0], limits[1])
    # observation.wav_select(2100, 2200)

    return combined
