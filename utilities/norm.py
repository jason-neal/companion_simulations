"""Normalization codes."""
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def chi2_model_norms(wave, obs, models, method='scalar', splits=50):
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

    # Get each model val out of matrix
    # for model_spec in model:
    #     model_continuum = continuum(wave, model_spec, **args)
    def my_continuum(flux):
        """Continuum with predefined varaibles parameters."""
        return continuum(wave, flux, splits=splits, method=method, plot=False)

    norm_models = np.apply_along_axis(my_continuum, 0, models)

    print(norm_models.shape)
    print(models.shape)

    # Transpose to do automated broadcasting as it left pads dimensions.
    norm_fraction = (obs_continuum.T / norm_models.T).T

    print(obs_continuum.shape)
    print(obs.shape)
    print(norm_fraction.shape)
    return (obs.T * norm_fraction.T).T



def local_normalization(wave, flux, splits=50, method="exponential", plot=False):
    r"""Local minimization for section of Phoenix spectra.

    Split spectra into many chunks and get the average of top 5\% in each bin.

    Fit to those points and normalize by that.
    """
    flux = copy.copy(flux)

    norm_flux = continuum_value(wave, flux, splits=splits, method=method, plot=plot)

    return flux / norm_flux


def continuum(wave, flux, splits=50, method='scalar', plot='False'):
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




def renormalize_observation(wav, obs_flux, broadcast_flux, splits=10):
    """Renormalize obs_spec to the linear continum fit along."""
    # Get median values of 10 highest points in the 0.5nm sections of flux

    obs_norm = broadcast_continuum_fit(wav, obs_flux, splits=splits, method="linear", plot=True)
    broad_norm = broadcast_continuum_fit(wav, broadcast_flux, splits=splits, method="linear", plot=True)

    return obs_flux * (broad_norm / obs_norm)


def broadcast_continuum_fit(wave, flux, splits=50, method="linear", plot=True):
    r"""Continuum fit the N-D - flux array.

    Split spectra into many chunks and get the average of top 5\% in each bin.

    Fit to those points and normalize by that.
    """
    org_flux = copy.copy(flux)
    org_wave = copy.copy(wave)

    while len(wave) % splits != 0:
        # Shorten array untill can be evenly split up.
        wave = wave[:-1]
        flux = flux[:-1]
        print(wave.shape)
        print(flux.shape)
    if flux.ndim > wave.ndim:
        wave = wave * np.ones_like(flux)  # Broadcast it out

    wav_split = np.vsplit(wave, splits)
    flux_split = np.vsplit(flux, splits)  # split along axis=0
    print(type(wav_split), type(flux_split))
    print("wav shape", wave.shape)
    print("wav split shape", len(wav_split))
    print("flux shape", flux.shape)
    print("flux split shape", len(flux_split))
    print("wav split[0] shape", wav_split[0].shape)
    print("flux split[0] shape", flux_split[0].shape)


    # Work out how to map or Apply a function to each part of a numpy array

    # TODO!
    flux_split_medians = []
    wave_split_medians = []
    wav_points = np.empty_like(splits)
    print(wav_points.shape)
    flux_points = np.empty(splits)
    f = flux_split
    print("argsort", np.argsort(f[0], axis=0))
    print("f[argsort]", f[np.argsort(f[0], axis=0)])
    print(np.median(f[np.argsort(f[0], axis=0)]))
    for i, (w, f) in enumerate(zip(wav_split, flux_split)):
        wav_points[i] = np.median(w[np.argsort(f, axis=0)[-5:]],
                                  axis=0, keepdims=True)  # Take the median of the wavelength values of max values.
        flux_points[i, ] = np.median(f[np.argsort(f, axis=0)[-5:]], axis=0, keepdims=True)

    print("flux_points", flux_points)
    print("flux_points.shape", flux_points.shape)
    print("flux_points[0].shape", flux_points[0].shape)

    if method == "scalar":
        norm_flux = np.median(flux_split) * np.ones_like(org_wave)
    elif method == "linear":
        z = np.polyfit(wav_points, flux_points, 1)
        p = np.poly1d(z)
        norm_flux = p(org_wave)
    elif method == "quadratic":
        z = np.polyfit(wav_points, flux_points, 2)
        p = np.poly1d(z)
        norm_flux = p(org_wave)
    elif method == "exponential":
        z = np.polyfit(wav_points, np.log(flux_points), deg=1, w=np.sqrt(flux_points))
        p = np.poly1d(z)
        norm_flux = np.exp(p(org_wave))   # Un-log the y values.

    if plot:
        plt.subplot(211)
        plt.plot(wave, flux)
        plt.plot(wav_points, flux_points, "x-", label="points")
        plt.plot(org_wave, norm_flux, label='norm_flux')
        plt.legend()
        plt.subplot(212)
        plt.plot(org_wave, org_flux / norm_flux)
        plt.title("Normalization")
        plt.show()

    return org_flux / norm_flux


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
