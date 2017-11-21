"""Cross-correlation of spectrum."""

import matplotlib.pyplot as plt
import numpy as np
from spectrum_overload import Spectrum


def xcorr_peak(spectrum, model, plot=False):
    """Find RV offset between a spectrum and a model using PyAstronomy.

    Parameters
    ----------
    spectrum: Spectrum
       Target Spectrum object.
    model: Spectrum
        Template Spectrum object.
    plot:bool
        Turn on plots.
    Returns
    -------
    rv_max: float
        Radial velocity value corresponding to maximum correlation.
    cc_max: float
        Cross-correlation value corresponding to maximum correlation.
    """
    assert isinstance(spectrum, Spectrum)
    assert isinstance(model, Spectrum)

    rv, cc = spectrum.crosscorr_rv(model, rvmin=-60., rvmax=60.0, drv=0.1,
                                  mode='doppler', skipedge=50)  # Specturm method

    maxind = np.argmax(cc)
    rv_max, cc_max = rv[maxind], cc[maxind]

    if plot:
        plt.subplot(211)
        plt.plot(spectrum.xaxis, spectrum.flux, label="Target")
        plt.plot(model.xaxis, model.flux, label="Model")
        plt.legend()
        plt.title("Spectra")

        plt.subplot(212)
        plt.plot(rv, cc)
        plt.plot(rv_max, cc_max, "o")
        plt.title("Cross correlation plot")
        plt.show()
    return float(rv[maxind]), float(cc[maxind])
