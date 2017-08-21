"""Normalization codes."""
import copy

import matplotlib.pyplot as plt
import numpy as np

from utilities.debug_utils import timeit2


def spec_local_norm(spectrum, splits=50, method="quadratic", plot=False, top=20):
    r"""Apply local normalization on Spectrum object.

    Split spectra into many chunks and get the average of top 5\% in each bin.
    """
    norm_spectrum = copy.copy(spectrum)
    flux_norm = local_normalization(norm_spectrum.xaxis, norm_spectrum.flux,
                                    splits=splits, plot=plot, method=method,
                                    top=top)
    norm_spectrum.flux = flux_norm

    return norm_spectrum


def local_normalization(wave, flux, splits=50, method="exponential", plot=False, top=20):
    r"""Local minimization for section of Phoenix spectra.

    Split spectra into many chunks and get the average of top 5\% in each bin.

    Fit to those points and normalize by that.
    """
    flux = copy.copy(flux)

    norm_flux = continuum(wave, flux, splits=splits, method=method, plot=plot, top=top)

    return flux / norm_flux


def get_continuum_points(wave, flux, splits=50, top=20):
    """Get continuum points along a spectrum.

    This splits a spectrum into "splits" number of bins and calculates
    the medain wavelength and flux of the upper "top" number of flux
    values.
    """
    # Shorten array until can be evenly split up.
    remainder = len(flux) % splits
    if remainder:
        # Nozero reainder needs this slicing
        wave = wave[:-remainder]
        flux = flux[:-remainder]

    wave_shaped = wave.reshape((splits, -1))
    flux_shaped = flux.reshape((splits, -1))

    s = np.argsort(flux_shaped, axis=-1)[:, -top:]

    s_flux = np.array([ar1[s1] for ar1, s1 in zip(flux_shaped, s)])
    s_wave = np.array([ar1[s1] for ar1, s1 in zip(wave_shaped, s)])

    wave_points = np.median(s_wave, axis=-1)
    flux_points = np.median(s_flux, axis=-1)
    assert len(flux_points) == splits

    return wave_points, flux_points


def continuum(wave, flux, splits=50, method='scalar', plot=False, top=20):
    """Fit continuum of flux.

    top: is number of top points to take median of continuum.
    """
    org_wave = wave[:]
    org_flux = flux[:]

    # Get continuum value in chunked sections of spectrum.
    wave_points, flux_points = get_continuum_points(wave, flux, splits=splits, top=top)

    poly_num = {"scalar": 0, "linear": 1, "quadratic": 2, "cubic": 3}

    if method == "exponential":
        z = np.polyfit(wave_points, np.log(flux_points), deg=1, w=np.sqrt(flux_points))
        p = np.poly1d(z)
        norm_flux = np.exp(p(org_wave))   # Un-log the y values.
    else:
        z = np.polyfit(wave_points, flux_points, poly_num[method])
        p = np.poly1d(z)
        norm_flux = p(org_wave)

    if plot:
        plt.subplot(211)
        plt.plot(wave, flux)
        plt.plot(wave_points, flux_points, "x-", label="points")
        plt.plot(org_wave, norm_flux, label='norm_flux')
        plt.legend()
        plt.subplot(212)
        plt.plot(org_wave, org_flux / norm_flux)
        plt.title("Normalization")
        plt.xlabel("Wavelength (nm)")
        plt.show()

    return norm_flux


@timeit2
def chi2_model_norms(wave, obs, models, method='scalar', splits=100, top=20):
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

    Notes
    -----
    I was attempting to apply get_cont_points along axis but it would not
    be possible to do the global polynomial fit as the x values were all
    different
    """
    if np.any(np.isnan(models)):
        raise ValueError("NaNS are not allowed in models during normalization, "
                         "check evaulated wavlength.")

    obs_continuum = continuum(wave, obs, splits=splits, method=method, top=top)

    # Try fit the apply_along_axis to get the continuum points only then polyfit n-D array.
    def axis_continuum(flux):
        """Continuum to apply along axis with predefined varaibles parameters."""
        return continuum(wave, flux, splits=splits, method=method, top=top)

    norm_models = np.apply_along_axis(axis_continuum, 0, models)

    # Transpose to do automated broadcasting as it left pads dimensions.
    norm_fraction = (obs_continuum.T / norm_models.T).T

    return (obs.T * norm_fraction.T).T


# # # SPEC local norm_slow  and assiciated are the older version
@timeit2
def chi2_model_norms_slow(wave, obs, models, method='scalar', splits=100, top=20):
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
    if np.any(np.isnan(models)):
        raise ValueError("NaNS are not allowed in models during normalization, "
                         "check evaulated wavlength.")

    obs_continuum = continuum_slow(wave, obs, splits=splits, method=method, top=top)

    def axis_continuum(flux):
        """Continuum to apply along axis with predefined varaibles parameters."""
        return continuum_slow(wave, flux, splits=splits, method=method, top=top)

    norm_models = np.apply_along_axis(axis_continuum, 0, models)

    # Transpose to do automated broadcasting as it left pads dimensions.
    norm_fraction = (obs_continuum.T / norm_models.T).T

    return (obs.T * norm_fraction.T).T


def spec_local_norm_slow(spectrum, splits=50, method="quadratic", plot=False, top=20):
    r"""Apply local normalization on Spectrum object.

    Split spectra into many chunks and get the average of top 5\% in each bin.
    """
    norm_spectrum = copy.copy(spectrum)
    flux_norm = local_normalization_slow(norm_spectrum.xaxis, norm_spectrum.flux,
                                         splits=splits, plot=plot, method=method,
                                         top=top)
    norm_spectrum.flux = flux_norm

    return norm_spectrum


def local_normalization_slow(wave, flux, splits=50, method="exponential", plot=False, top=20):
    r"""Local minimization for section of Phoenix spectra.

    Split spectra into many chunks and get the average of top 5\% in each bin.

    Fit to those points and normalize by that.
    """
    flux = copy.copy(flux)

    norm_flux = continuum_slow(wave, flux, splits=splits, method=method, plot=plot, top=top)

    return flux / norm_flux


def continuum_slow(wave, flux, splits=50, method='scalar', plot=False, top=20):
    """Fit continuum of flux.

    top: is number of top points to take median of to get continuum."""
    org_wave = copy.copy(wave)
    org_flux = copy.copy(flux)

    while len(wave) % splits != 0:
        # Shorten array untill can be evenly split up.
        wave = wave[:-1]
        flux = flux[:-1]

    flux_split = np.split(flux, splits)
    wav_split = np.split(wave, splits)

    wave_points = np.empty(splits)
    flux_points = np.empty(splits)

    for i, (w, f) in enumerate(zip(wav_split, flux_split)):
        # Do the sorting only once to half the time taken here.
        high_args = np.argsort(f)[-top:]
        wave_points[i] = np.median(w[high_args])  # Take the median of the wavelength values of max values.
        flux_points[i] = np.median(f[high_args])

    poly_num = {"scalar": 0, "linear": 1, "quadratic": 2, "cubic": 3}

    if method == "exponential":
        z = np.polyfit(wave_points, np.log(flux_points), deg=1, w=np.sqrt(flux_points))
        p = np.poly1d(z)
        norm_flux = np.exp(p(org_wave))   # Un-log the y values.
    else:
        z = np.polyfit(wave_points, flux_points, poly_num[method])
        p = np.poly1d(z)
        norm_flux = p(org_wave)

    if plot:
        plt.subplot(211)
        plt.plot(wave, flux)
        plt.plot(wave_points, flux_points, "x-", label="points")
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
