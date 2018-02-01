"""Starfish spectra verse phoenix spectra.

I notice a difference in the flux level of starfish spectra compared to the manually loaded spectra.

Investigating that here.
"""
import os

import Starfish
import matplotlib.pyplot as plt
import numpy as np
from Starfish.grid_tools import HDF5Interface
from astropy.io import fits

import simulators

myHDF5 = HDF5Interface(filename="/home/jneal/Phd/Codes/companion_simulations/starfish_tests/libraries/PHOENIX_50k.hdf5")
myHDF5_air = HDF5Interface(
    filename="/home/jneal/Phd/Codes/companion_simulations/starfish_tests/libraries/PHOENIX_air.hdf5")
myHDF5_norm_air = HDF5Interface(
    filename="/home/jneal/Phd/Codes/companion_simulations/starfish_tests/libraries/PHOENIX_norm_air.hdf5")
myHDF5_norm = HDF5Interface(
    filename="/home/jneal/Phd/Codes/companion_simulations/starfish_tests/libraries/PHOENIX_norm.hdf5")
wl = myHDF5.wl
wl_air = myHDF5_air.wl

# parrange: [[5000, 5200], [4.0, 4.0], [-0.0, 0.0]]
wl_range = Starfish.grid["wl_range"]
print(wl_range)
params = [5100, 4.0, 0.0]
flux = myHDF5.load_flux(np.array(params))
flux_air = myHDF5_air.load_flux(np.array(params))
flux_norm_air = myHDF5_norm_air.load_flux(np.array(params))
flux_norm = myHDF5_norm.load_flux(np.array(params))

# Load direct phoenix spectra
path = simulators.starfish_grid["raw_path"]
phoenix = os.path.join(path, "Z-0.0",
                       "lte{:05d}-{:0.2f}-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(params[0], params[1]))
phoenix_wav = os.path.join(path, "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")

print(phoenix)
raw_flux = fits.getdata(phoenix)
wav = fits.getdata(phoenix_wav)
plt.plot(wav, raw_flux, label="raw")
plt.plot(wl, flux, label="starfish")
plt.plot(wl, flux_air, label="air")
plt.plot(wl, flux_norm_air * 5e7, label="norm_air")
plt.plot(wl, flux_norm * 5e7, label="norm")
# plt.xlim(wl_range)
plt.legend()
plt.show()

# To make issue of last points.
print("wav_max", np.max(wav))
mask = (wav > (wl_range[0] - 100)) & (wav < (wl_range[1] + 100))
masked_wav = wav[mask]
masked_flux = raw_flux[mask]

raw_flux = fits.getdata(phoenix)
wav = fits.getdata(phoenix_wav)
plt.plot(masked_wav, masked_flux, label="Phoenix")
plt.plot(wl, flux, label="Gridtools")
plt.plot(wl, flux_air, label="Gridtools-air")
# plt.plot(wl, flux_norm_air * 5e7, label="norm_air")
# plt.plot(wl, flux_norm * 5e7, label="norm")
plt.legend()
plt.xlabel("Wavelegnth (Angstrom)")
plt.ylabel("Flux")
plt.show()
