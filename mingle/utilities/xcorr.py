"""Cross-correlation of spectrum."""

import matplotlib.pyplot as plt
import numpy as np
from spectrum_overload import Spectrum
import scipy

def subtractContinuum(s, n=3, plot=False):
	'''Take a 1D array, use spline to subtract off continuum.
			subtractContinuum(s, n=3)
			required:
			s = the array
			optional:
			n = 3, the number of spline points to use
	'''

	x = np.arange(len(s))
	points = (np.arange(n)+1)*len(s)/(n+1)
	spline = scipy.interpolate.LSQUnivariateSpline(x,s,points)
	if plot:
		plt.ion()
		plt.figure()
		plt.plot(x, s)
		plt.plot(x, spline(x), linewidth=5, alpha=0.5)
	return s - spline(x)


def xcorr_peak(spectrum, model, plot=False, subtract_continuum=True):
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
    if subtract_continuum:
        spectrum = spectrum.copy()
        model = model.copy()
        spectrum.flux = subtractContinuum(spectrum.flux, n=3, plot=plot)
        model.flux = subtractContinuum(model.flux, n=3, plot=plot)

    rv, cc = spectrum.crosscorr_rv(model, rvmin=-60., rvmax=60.0, drv=0.1,
                                  mode='doppler', skipedge=50)  # Spectrum method

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
