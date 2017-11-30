"""Companion simulate models using Broadcasting."""
import numpy as np
from scipy.interpolate import interp1d


def one_comp_model(wav, model1, *, gammas=None):
    """Make 1 component simulations, broadcasting over gamma values."""
    # Enable single scalar inputs (turn to 1d np.array)
    gammas = check_broadcastable(gammas).squeeze(axis=1)

    m1 = model1
    m1g = np.empty(model1.shape + (len(gammas),))  # am2rvm1g = am2rvm1 with gamma doppler-shift

    for j, gamma in enumerate(gammas):
        wav_j = (1 + gamma / 299792.458) * wav
        m1g[:, j] = interp1d(wav_j, m1, axis=0, bounds_error=False)(wav)

    assert m1g.shape == (len(model1), len(gammas)), "Dimensions of broadcast output not correct"
    return interp1d(wav, m1g, axis=0)  # pass it the wavelength values to return


def check_broadcastable(var):
    # My version of broadcastable with 1s on the right
    var = np.atleast_2d(var)
    v_shape = var.shape
    # to make it (N, 1)
    if v_shape[0] == 1 and v_shape[0] < v_shape[1]:
        var = np.swapaxes(var, 0, 1)
    return var


def two_comp_model(wav, model1, model2, *, alphas=None, rvs=None, gammas=None, kind="quadratic"):
    """Make 2 component simulations, broadcasting over alpha, rv, gamma values."""
    # Enable single scalar inputs (turn to 1d np.array)
    alphas = check_broadcastable(alphas).squeeze(axis=1)
    rvs = check_broadcastable(rvs).squeeze(axis=1)
    gammas = check_broadcastable(gammas).squeeze(axis=1)

    am2 = model2[:, np.newaxis] * alphas  # alpha * Model2 (am2)
    am2rv = np.empty(am2.shape + (len(rvs),))  # am2rv = am2 with rv doppler-shift

    for i, rv in enumerate(rvs):
        wav_i = (1 + rv / 299792.458) * wav
        am2rv[:, :, i] = interp1d(wav_i, am2, axis=0, kind=kind, bounds_error=False)(wav)

    # Normalize by (1 / 1 + alpha)
    am2rv = am2rv / (1 + alphas)[np.newaxis, :, np.newaxis]
    am2rvm1 = model1[:, np.newaxis, np.newaxis] + am2rv  # am2rvm1 = am2rv + model_1
    am2rvm1g = np.empty(am2rvm1.shape + (len(gammas),))  # am2rvm1g = am2rvm1 with gamma doppler-shift
    for j, gamma in enumerate(gammas):
        wav_j = (1 + gamma / 299792.458) * wav
        am2rvm1g[:, :, :, j] = interp1d(wav_j, am2rvm1, kind=kind, axis=0, bounds_error=False)(wav)

    assert am2rvm1g.shape == (len(model1), len(alphas), len(rvs), len(gammas)), "Dimensions of broadcast not correct"
    return interp1d(wav, am2rvm1g, kind=kind, bounds_error=False, axis=0)  # pass it the wavelength values to return


def two_comp_model_with_transpose(wav, model1, model2, alphas, *, rvs=None, gammas=None):
    """Make 2 component simulations, broadcasting over alpha, rv, gamma values."""
    # Enable single scalar inputs (turn to 1d np.array)
    alphas = check_broadcastable(alphas)
    rvs = check_broadcastable(rvs)
    gammas = check_broadcastable(gammas)

    am2 = (model2.T * alphas.T).T  # alpha * Model2 (am2)
    am2rv = np.empty(am2.shape + (len(rvs),))  # am2rv = am2 with rv doppler-shift

    for i, rv in enumerate(rvs):
        wav_i = (1 + rv / 299792.458) * wav
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


def inherent_alpha_model(wav, model1, model2, *, rvs=None, gammas=None, kind="linear"):
    """Make 2 component simulations, broadcasting over, rv, gamma values."""
    # Enable single scalar inputs (turn to 1d np.array)
    # alphas = check_broadcastable(alphas)
    rvs = check_broadcastable(rvs)
    gammas = check_broadcastable(gammas)

    # am2 = (model2.T * alphas.T).T  # alpha * Model2 (am2)
    m2rv = np.empty(model2.shape + (len(rvs),))  # m2rv = model2 with rv doppler-shift

    for i, rv in enumerate(rvs):
        wav_i = (1 + rv / 299792.458) * wav
        m2rv[:, i] = interp1d(wav_i, model2, axis=0, kind=kind, bounds_error=False)(wav)
    # print("number of values not finite after doppler shift 1", np.sum(~np.isfinite(m2rv)))
    # print("m2rv shape", m2rv.shape)
    # print("locations not finite", np.where(~np.isfinite(m2rv)))
    # assert np.all(np.isfinite(m2rv))
    m2rvm1 = (model1.T + m2rv.T).T  # m2rvm1 = am2rv + model_1
    m2rvm1g = np.empty(m2rvm1.shape + (len(gammas),))  # m2rvm1g = m2rvm1 with gamma doppler-shift
    for j, gamma in enumerate(gammas):
        wav_j = (1 + gamma / 299792.458) * wav
        m2rvm1g[:, :, j] = interp1d(wav_j, m2rvm1, axis=0, kind=kind, bounds_error=False)(wav)
    # assert np.all(np.isfinite(m2rvm1g))
    # print("number of values not finite after doppler shift 2", np.sum(~np.isfinite(m2rvm1g)))
    # print("m2rvm1g shape", m2rvm1g.shape)
    # print("locations not finite", np.where(~np.isfinite(m2rv)))
    assert m2rvm1g.shape == (len(model1), len(rvs), len(gammas)), "Dimensions of broadcast not correct"
    return interp1d(wav, m2rvm1g, kind=kind, axis=0, bounds_error=False)  # pass it the wavelength values to return   # return interp1d(wav, m2rvm1g, axis=0)  # pass it the wavelength values to return   #


def independent_inherent_alpha_model(wav, model1, model2, *, rvs=None, gammas=None, kind="quadratic"):
    """Make 2 component simulations, broadcasting over, rv, gamma values.

    Independent RV and Gamma variables.
    """
    rvs = check_broadcastable(rvs)
    gammas = check_broadcastable(gammas)

    # am2 = (model2.T * alphas.T).T  # alpha * Model2 (am2)
    m2_shifted = np.empty(model2.shape + (len(rvs),))  # m2rv = model2 with rv doppler-shift

    for i, rv in enumerate(rvs):
        wav_i = (1 + rv / 299792.458) * wav
        m2_shifted[:, i] = interp1d(wav_i, model2, axis=0, kind=kind, bounds_error=False)(wav)
    assert np.all(np.isfinite(m2_shifted))

    m1_shifted = np.empty(model1.shape + (len(gammas),))  # m2rvm1g = m2rvm1 with gamma doppler-shift
    for j, gamma in enumerate(gammas):
        wav_j = (1 + gamma / 299792.458) * wav
        m1_shifted[:, j] = interp1d(wav_j, model1, axis=0, kind=kind, bounds_error=False)(wav)
    assert np.all(np.isfinite(m1_shifted))
    # print(m2rvm1g.shape)
    # print(m2rv.shape)
    # print("m1_shifted", m1_shifted.shape)
    # print("m2_shifted", m2_shifted.shape)
    m2rvm1g = (m1_shifted[:, np.newaxis, :] + m2_shifted[:, :, np.newaxis])
    # print(m2rvm1g.shape)
    # print("expected shape", (len(model1), len(rvs), len(gammas)))
    assert m2rvm1g.shape == (len(model1), len(rvs), len(gammas)), "Dimensions of broadcast not correct"
    assert np.all(np.isfinite(m2rvm1g))
    return interp1d(wav, m2rvm1g, kind=kind, axis=0, bounds_error=False)  # pass it the wavelength values to return
