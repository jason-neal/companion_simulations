import numpy as np
from scipy.interpolate import interp1d

def one_comp_model(wav, model1, gammas):
    """ Make 1 component simulations, broadcasting over gamma values.
    """
    # Enable single scalar inputs (turn to 1d np.array)
    if not hasattr(gammas, "__len__"):
        gammas = np.asarray(gammas)[np.newaxis]
        print(len(gammas))

    m1 = model1
    print(model1.shape)

    m1g = np.empty(model1.shape + (len(gammas),))   # am2rvm1g = am2rvm1 with gamma doppler-shift
    print(m1g.shape)
    for j, gamma in enumerate(gammas):
        wav_j = (1 + gamma / 299792.458) * wav
        m1g[:, j] = interp1d(wav_j, m1, axis=0, bounds_error=False)(wav)

    return interp1d(w, m1g, axis=0)    # pass it the wavelength values to return


def two_comp_model(wav, model1, model2, alphas, rvs, gammas):
    # Make 2 component simulations, broadcasting over alpha, rv, gamma values.

    # Enable single scalar inputs (turn to 1d np.array)
    if not hasattr(alphas, "__len__"):
        alphas = np.asarray(alphas)[np.newaxis]
    if not hasattr(rvs, "__len__"):
        rvs = np.asarray(rvs)[np.newaxis]
    if not hasattr(gammas, "__len__"):
        gammas = np.asarray(gammas)[np.newaxis]
        print(len(gammas))

    am2 = model2[:, np.newaxis] * alphas  # alpha * Model2 (am2)
    print(am2.shape)

    am2rv = np.empty(am2.shape + (len(rvs),))  # am2rv = am2 with rv doppler-shift
    print(am2rv.shape)
    for i, rv in enumerate(rvs):
        # nflux, wlprime = dopplerShift(wav, am2, rv)
        # am2rv[:, :, i] = nflux
        wav_i = (1 - rv / c) * wav
        am2rv[:, :, i] = interp1d(wav_i, am2, axis=0, bounds_error=False)(wav)

    # Normalize by (1 / 1 + alpha)
    am2rv = am2rv / (1 + alphas)[np.newaxis, :, np.newaxis]

    am2rvm1 = h[:, np.newaxis, np.newaxis] + am2rv  # am2rvm1 = am2rv + model_1
    print(am2rvm1.shape)

    am2rvm1g = np.empty(am2rvm1.shape + (len(gammas),))  # am2rvm1g = am2rvm1 with gamma doppler-shift
    for j, gamma in enumerate(gammas):
        wav_j = (1 - gamma / 299792.458) * wav
        am2rvm1g[:, :, :, j] = interp1d(wav_j, am2rvm1, axis=0, bounds_error=False)(wav)

    return interp1d(w, am2rvm1g, axis=0)  # pass it the wavelength values to return