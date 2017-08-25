"""Companion simulat models useing Boradcasting."""
import numpy as np
from scipy.interpolate import interp1d

from utilities.debug_utils import timeit2


@timeit2
def one_comp_model(wav, model1, gammas):
    """Make 1 component simulations, broadcasting over gamma values."""
    # Enable single scalar inputs (turn to 1d np.array)
    gammas = check_broadcastable(gammas).squeeze(axis=1)

    m1 = model1
    m1g = np.empty(model1.shape + (len(gammas),))   # am2rvm1g = am2rvm1 with gamma doppler-shift

    for j, gamma in enumerate(gammas):
        wav_j = (1 + gamma / 299792.458) * wav
        m1g[:, j] = interp1d(wav_j, m1, axis=0, bounds_error=False)(wav)

    assert m1g.shape == (len(model1), len(gammas)), "Dimensions of broadcast output not correct"
    return interp1d(wav, m1g, axis=0)    # pass it the wavelength values to return


def check_broadcastable(var):
    # My version of broadcasable with 1s on the right
    var = np.atleast_2d(var)
    v_shape = var.shape
    # to make it (N, 1)
    if v_shape[0] == 1 and v_shape[0] < v_shape[1]:
        var = np.swapaxes(var, 0, 1)
    return var


@timeit2
def two_comp_model(wav, model1, model2, alphas, rvs, gammas):
    """Make 2 component simulations, broadcasting over alpha, rv, gamma values."""
    # Enable single scalar inputs (turn to 1d np.array)
    alphas = check_broadcastable(alphas).squeeze(axis=1)
    rvs = check_broadcastable(rvs).squeeze(axis=1)
    gammas = check_broadcastable(gammas).squeeze(axis=1)

    am2 = model2[:, np.newaxis] * alphas  # alpha * Model2 (am2)
    am2rv = np.empty(am2.shape + (len(rvs),))  # am2rv = am2 with rv doppler-shift

    for i, rv in enumerate(rvs):
        wav_i = (1 - rv / 299792.458) * wav
        am2rv[:, :, i] = interp1d(wav_i, am2, axis=0, bounds_error=False)(wav)

    # Normalize by (1 / 1 + alpha)
    am2rv = am2rv / (1 + alphas)[np.newaxis, :, np.newaxis]
    am2rvm1 = model1[:, np.newaxis, np.newaxis] + am2rv  # am2rvm1 = am2rv + model_1
    am2rvm1g = np.empty(am2rvm1.shape + (len(gammas),))  # am2rvm1g = am2rvm1 with gamma doppler-shift
    for j, gamma in enumerate(gammas):
        wav_j = (1 + gamma / 299792.458) * wav
        am2rvm1g[:, :, :, j] = interp1d(wav_j, am2rvm1, axis=0, bounds_error=False)(wav)

    assert am2rvm1g.shape == (len(model1), len(alphas), len(rvs), len(gammas)), "Dimensions of broadcast not correct"
    return interp1d(wav, am2rvm1g, axis=0)  # pass it the wavelength values to return


@timeit2
def two_comp_model_with_transpose(wav, model1, model2, alphas, rvs, gammas):
    """Make 2 component simulations, broadcasting over alpha, rv, gamma values."""
    # Enable single scalar inputs (turn to 1d np.array)
    alphas = check_broadcastable(alphas)
    rvs = check_broadcastable(rvs)
    gammas = check_broadcastable(gammas)

    am2 = (model2.T * alphas.T).T  # alpha * Model2 (am2)
    am2rv = np.empty(am2.shape + (len(rvs),))  # am2rv = am2 with rv doppler-shift

    for i, rv in enumerate(rvs):
        wav_i = (1 - rv / 299792.458) * wav
        am2rv[:, :, i] = interp1d(wav_i, am2, axis=0, bounds_error=False)(wav)

    # Normalize by (1 / 1 + alpha)
    am2rv = am2rv / (1 + alphas)[np.newaxis, :, np.newaxis]
    am2rvm1 = (model1.T + am2rv.T).T  # am2rvm1 = am2rv + model_1
    am2rvm1g = np.empty(am2rvm1.shape + (len(gammas),))  # am2rvm1g = am2rvm1 with gamma doppler-shift
    for j, gamma in enumerate(gammas):
        wav_j = (1 + gamma / 299792.458) * wav
        am2rvm1g[:, :, :, j] = interp1d(wav_j, am2rvm1, axis=0, bounds_error=False)(wav)

    assert am2rvm1g.shape == (len(model1), len(alphas), len(rvs), len(gammas)), "Dimensions of broadcast not correct"
    return interp1d(wav, am2rvm1g, axis=0)  # pass it the wavelength values to return


@timeit2
def inherint_alpha_model(wav, model1, model2, rvs, gammas):
    """Make 2 component simulations, broadcasting over, rv, gamma values."""
    # Enable single scalar inputs (turn to 1d np.array)
    # alphas = check_broadcastable(alphas)
    rvs = check_broadcastable(rvs)
    gammas = check_broadcastable(gammas)

    # am2 = (model2.T * alphas.T).T  # alpha * Model2 (am2)
    m2rv = np.empty(model2.shape + (len(rvs),))  # m2rv = model2 with rv doppler-shift

    for i, rv in enumerate(rvs):
        wav_i = (1 - rv / 299792.458) * wav
        m2rv[:, i] = interp1d(wav_i, model2, axis=0, bounds_error=False)(wav)

    m2rvm1 = (model1.T + m2rv.T).T  # m2rvm1 = am2rv + model_1
    m2rvm1g = np.empty(m2rvm1.shape + (len(gammas),))  # m2rvm1g = m2rvm1 with gamma doppler-shift
    for j, gamma in enumerate(gammas):
        wav_j = (1 + gamma / 299792.458) * wav
        m2rvm1g[:, :, j] = interp1d(wav_j, m2rvm1, axis=0, bounds_error=False)(wav)

    assert m2rvm1g.shape == (len(model1), len(rvs), len(gammas)), "Dimensions of broadcast not correct"
    return interp1d(wav, m2rvm1g, axis=0)  # pass it the wavelength values to return
