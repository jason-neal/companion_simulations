"""Starfish spectra verse phoenix spectra.

I notice a difference in the flux level of starfish spectra compared to the manually loaded spectra.

Investigating that here.
"""
import numpy as np

import matplotlib.pyplot as plt
import Starfish
from astropy.io import fits
from spectrum_overload import Spectrum
from Starfish.grid_tools import HDF5Interface

myHDF5 = HDF5Interface()
wl = myHDF5.wl

params = [6100, 4.5, 0.0]
flux = myHDF5.load_flux(np.array(params))

# Load direct phoenix spectra

path = simulators.starfish_grid["raw_path"]

phoenix = os.path.join(path, "Z-0.0",
                       "lte{:05d}-{:0.2f}-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(params[0], params[1]))
phoenix_wav = os.path.join(path, "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
print(phoenix)
raw_flux = fits.getdata(phoenix)
wav = fits.getdata(phoenix_wav)
plt.plot(wav, raw_flux, label="raw")
plt.plot(wl, flux * 5e7, label="starfish")
plt.legend()
plt.show()

params2 = [5200, 4.5, 0.0]
flux = myHDF5.load_flux(np.array(params2))

# Load direct phoenix spectra

path = simulators.starfish_grid["raw_path"]

phoenix = os.path.join(path, "Z-0.0",
                       "lte{:05d}-{:0.2f}-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(params2[0], params2[1]))
phoenix_wav = os.path.join(path, "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")

print(phoenix)
raw_flux = fits.getdata(phoenix)
wav = fits.getdata(phoenix_wav)
phoenix_spec = Spectrum(xaxis=wav, flux=raw_flux)
phoenix_spec.wav_select(wl[0], wl[-1])
plt.plot(wav, raw_flux, label="raw")
plt.plot(phoenix_spec.xaxis, phoenix_spec.flux, "--", label="spec")
plt.plot(wl, flux * 4e6, label="starfish")
plt.legend()
plt.show()
