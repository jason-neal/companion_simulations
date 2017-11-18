"""Normalization codes."""
import copy

import matplotlib.pyplot as plt
import numpy
import numpy as np


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
    the median wavelength and flux of the upper "top" number of flux
    values.
    """
    # Shorten array until can be evenly split up.
    remainder = len(flux) % splits
    if remainder:
        # Non-zero remainder needs this slicing
        wave = wave[:-remainder]
        flux = flux[:-remainder]

    wave_shaped = wave.reshape((splits, -1))
    flux_shaped = flux.reshape((splits, -1))

    s = np.argsort(flux_shaped, axis=-1)[:, -top:]

    s_flux = np.array([ar1[s1] for ar1, s1 in zip(flux_shaped, s)])
    s_wave = np.array([ar1[s1] for ar1, s1 in zip(wave_shaped, s)])

    wave_points = np.nanmedian(s_wave, axis=-1)
    flux_points = np.nanmedian(s_flux, axis=-1)
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
        continuum_fit = np.exp(p(org_wave))  # Un-log the y values.
    else:
        z = np.polyfit(wave_points, flux_points, poly_num[method])
        p = np.poly1d(z)
        continuum_fit = p(org_wave)

    if plot:
        plt.subplot(211)
        plt.plot(wave, flux)
        plt.plot(wave_points, flux_points, "x-", label="points")
        plt.plot(org_wave, continuum_fit, label='norm_flux')
        plt.legend()
        plt.subplot(212)
        plt.plot(org_wave, org_flux / continuum_fit)
        plt.title("Normalization")
        plt.xlabel("Wavelength (nm)")
        plt.show()

    return continuum_fit


# @timeit2
def chi2_model_norms(wave, obs, models, method='scalar', splits=100, top=20):
    """Normalize the obs to the continuum of the models.

    Inputs
    ------
    obs: n*1 N-D arrary
    models: n*x*... N-D array
    method: str
        const or linear
    Returns
    -------
    norm_obs: N-D arrary
        Observation normalized to all model continuum.

    Notes
    -----
    I was attempting to apply get_cont_points along axis but it would not
    be possible to do the global polynomial fit as the x values were all
    different
    """
    if np.any(np.isnan(models)):
        raise ValueError("NaNS are not allowed in models during normalization, "
                         "check evaluated wavelength.")

    obs_continuum = continuum(wave, obs, splits=splits, method=method, top=top)

    # Try fit the apply_along_axis to get the continuum points only then polyfit n-D array.
    def axis_continuum(flux):
        """Continuum to apply along axis with predefined variables parameters."""
        return continuum(wave, flux, splits=splits, method=method, top=top)

    norm_models = np.apply_along_axis(axis_continuum, 0, models)

    # Transpose to do automated broadcasting as it left pads dimensions.
    norm_fraction = (obs_continuum.T / norm_models.T).T

    return (obs.T * norm_fraction.T).T


def arbitrary_rescale(model_grid, start, stop, step):
    """Arbitrarily rescale the flux of the grid to adjust continuum level.

    Returns
    -------
    new_models: np.ndarray
        Model grids extended by a new axis, an scaled by the values of arb_norm.
    arb_norm: np.ndarray
        Vector of values used for scaling.
    """

    arb_norm = np.arange(start, stop, step)
    # Equivalent to [:, :, :, np.newaxis] if shape was 3d but works for any shape.
    new_models = np.expand_dims(model_grid, -1)  # add newaxis to position -1
    new_models = new_models * arb_norm
    assert new_models.shape == (*model_grid.shape, len(arb_norm))

    return new_models, arb_norm


def arbitrary_minimums(model_grid, last_axis):
    """Find the minimum value along the last dimension."""
    min_locations = np.argmin(model_grid, axis=-1)
    new_model_grid = np.min(model_grid, axis=-1)
    axis_values = np.asarray(last_axis)[min_locations]

    return new_model_grid, axis_values


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
