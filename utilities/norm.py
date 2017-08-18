"""Normalization codes."""
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from utilities.debug_utils import timeit2


@timeit2
def chi2_model_norms(wave, obs, models, method='scalar', splits=100):
    """Normalize the obs to the continuum of the models.

    Inputs
    ------
    obs: n*1 N-D arrary
    models: n*x*... N Darray
    mehtod: str
        const or linear
    Returns
    -------
    norm_obs: N-D arrary
        Observation normalized to all model continuums.
    """


    obs_continuum = continuum(wave, obs, splits=splits, method=method)

    def model_continuum(flux):
        """Continuum with predefined varaibles parameters."""
        return continuum(wave, flux, splits=splits, method=method, plot=False)

    norm_models = np.apply_along_axis(model_continuum, 0, models)

    #print(norm_models.shape)
    #print(models.shape)

    # Transpose to do automated broadcasting as it left pads dimensions.
    norm_fraction = (obs_continuum.T / norm_models.T).T

    #print(obs_continuum.shape)
    #print(obs.shape)
    #print(norm_fraction.shape)
    return (obs.T * norm_fraction.T).T



def local_normalization(wave, flux, splits=50, method="exponential", plot=False):
    r"""Local minimization for section of Phoenix spectra.

    Split spectra into many chunks and get the average of top 5\% in each bin.

    Fit to those points and normalize by that.
    """
    flux = copy.copy(flux)

    norm_flux = continuum_value(wave, flux, splits=splits, method=method, plot=plot)

    return flux / norm_flux


def continuum(wave, flux, splits=50, method='scalar', plot=False):
    """Fit continuum of flux."""
    org_wave = copy.copy(wave)
    org_flux = copy.copy(flux)

    while len(wave) % splits != 0:
        # Shorten array untill can be evenly split up.
        wave = wave[:-1]
        flux = flux[:-1]

    flux_split = np.split(flux, splits)
    wav_split = np.split(wave, splits)

    wav_points = np.empty(splits)
    flux_points = np.empty(splits)

    for i, (w, f) in enumerate(zip(wav_split, flux_split)):
        wav_points[i] = np.median(w[np.argsort(f)[-20:]])  # Take the median of the wavelength values of max values.
        flux_points[i] = np.median(f[np.argsort(f)[-20:]])

    poly_num = {"linear": 1, "quadratic": 2, "cubic": 3}

    if method == "scalar":
        norm_flux = np.median(flux_split) * np.ones_like(org_wave)
    elif method == "exponential":
        z = np.polyfit(wav_points, np.log(flux_points), deg=1, w=np.sqrt(flux_points))
        p = np.poly1d(z)
        norm_flux = np.exp(p(org_wave))   # Un-log the y values.
    else:
        z = np.polyfit(wav_points, flux_points, poly_num[method])
        p = np.poly1d(z)
        norm_flux = p(org_wave)

    if plot:
        plt.subplot(211)
        plt.plot(wave, flux)
        plt.plot(wav_points, flux_points, "x-", label="points")
        plt.plot(org_wave, norm_flux, label='norm_flux')
        plt.legend()
        plt.subplot(212)
        plt.plot(org_wave, org_flux / norm_flux)
        plt.title("Normalization")
        plt.xlabel("Wavelength (nm)")
        plt.show()

    return norm_flux


if __name__ == "__main__":

    w = np.arange(500)
    f = np.sort(np.random.rand(500))  # .sort()
    print(f)
    models = np.sort(np.random.rand(500, 20, 2, 5), axis=0)
    print(models)
    plt.plot(w, f, label="line")
    plt.plot(w, models[:, :, 0, 0], "--")
    plt.show()

    res = chi2_model_norms(w, f, models, splits=5, method="linear")
    print(models.shape)
    print(res.shape)
    plt.show()
