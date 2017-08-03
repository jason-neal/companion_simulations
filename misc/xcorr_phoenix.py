
import matplotlib.pyplot as plt
# Xcorr the phoenix spectra against observation to find RV value.
# Test using the cross correlation function on a spectrum.
import numpy as np

from astropy.io import fits
from PyAstronomy import pyasl
from spectrum_overload.Spectrum import Spectrum

pathwave = "/home/jneal/Phd/data/phoenixmodels/" \
           "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
bd_model = "/home/jneal/Phd/data/phoenixmodels/" \
           "HD30501b-lte02500-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
star_model = "/home/jneal/Phd/data/phoenixmodels/" \
             "HD30501-lte05200-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"

i_bdmod = fits.getdata(bd_model)
i_star = fits.getdata(star_model)
hdr_bd = fits.getheader(bd_model)
hdr_star = fits.getheader(star_model)
w_mod = fits.getdata(pathwave)

w_mod /= 10   # turn into nm

test_rv = -15.205  # km/s

template_spectrum = Spectrum(xaxis=w_mod, flux=i_star, calibrated=True)
template_spectrum.wav_select(2100, 2200)

obs_spectrum = Spectrum(xaxis=w_mod, flux=i_star, calibrated=True)
obs_spectrum.doppler_shift(test_rv)
obs_spectrum.wav_select(2100, 2200)

print(len(obs_spectrum.xaxis))

rv, cc = pyasl.crosscorrRV(obs_spectrum.xaxis, obs_spectrum.flux, template_spectrum.xaxis, template_spectrum.flux,
                           rvmin=-60., rvmax=60.0, drv=0.1, mode='doppler', skipedge=2000)

maxind = np.argmax(cc)
print("Cross-correlation function is maximized at dRV = ", rv[maxind], " km/s")

# Test xcorr in spectrum
rv2, cc2 = obs_spectrum.crosscorrRV(template_spectrum, rvmin=-60., rvmax=60.0, drv=0.1, mode='doppler', skipedge=2000)
maxind2 = np.argmax(cc2)
print("Spectrum Cross-correlation function is maximized at dRV = ", rv2[maxind2], " km/s")


plt.subplot(211)
plt.plot(obs_spectrum.xaxis, obs_spectrum.flux, label="obs")
plt.plot(template_spectrum.xaxis, template_spectrum.flux, "--", label="template")
plt.legend()
plt.subplot(212)
plt.plot(rv, cc, label="manual")
plt.plot(rv2, cc2 * 1.001, label="spectrum")
plt.show()

# Doppler shift Model
wlcorr = obs_spectrum.xaxis * (1. - rv[maxind] / 299792.)

rva, cca = pyasl.crosscorrRV(wlcorr, obs_spectrum.flux, template_spectrum.xaxis, template_spectrum.flux,
                             rvmin=-60., rvmax=60.0, drv=0.1, mode='doppler', skipedge=2000)

maxind = np.argmax(cca)
print("Corrected Cross-correlation function is maximized at dRV = ", rva[maxind], " km/s")

obs_spectrum2 = obs_spectrum
obs_spectrum2.doppler_shift(-rv2[maxind2])


rv3, cc3 = obs_spectrum2.crosscorrRV(template_spectrum, rvmin=-60., rvmax=60.0, drv=0.1, mode='doppler', skipedge=2000)
maxind3 = np.argmax(cc3)
print("Spectrum Cross-correlation function is maximized at dRV = ", rv3[maxind3], " km/s")
wlcorr2 = wlcorr * (1. - rv[maxind] / 299792.)


plt.subplot(211)
plt.plot(wlcorr2, obs_spectrum.flux, label="obs")
plt.plot(obs_spectrum2.xaxis, obs_spectrum2.flux, "--", label="obs_spect")
plt.plot(template_spectrum.xaxis, template_spectrum.flux, "-.", label="template")
plt.legend()
plt.subplot(212)
plt.plot(rva, cca, label="manual")
plt.plot(rv3, cc3 * 1.001, label="spectrum2")
plt.legend()
plt.show()
