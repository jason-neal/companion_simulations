#!/usr/bin/env python
"""CRIRES 50k HDF& creation for companion simulations."""
## Make HDF5Creator to simplify files, optimize the data access.
# 1. Use only a range of spectra that span the likely parameter space of your star. For example, if we know we have an F5 star, maybe we will only use spectra that have 5900 K≤Teff≤6500 K
# .
# 2. Use only the part of the spectrum that overlaps your instrument’s wavelength coverage. For example, if the range of our spectrograph is 4000 - 9000 angstroms, it makes sense to discard the UV and IR portions of the synthetic spectrum.
# 3. Resample the high resolution spectra to a lower resolution more suitably matched to the resolution of your spectrograph. For example, PHOENIX spectra are provided at R∼500,000
# , while the TRES spectrograph has a resolution of R∼44,00

import numpy as np

import Starfish
from Starfish.grid_tools import CIFISTGridInterface as CIFIST
from Starfish.grid_tools import HDF5Creator, Instrument


# Create CRIRES instrument at the resolution we have.
class CRIRES_50k(Instrument):
    '''CRIRES instrument at R=50000.'''
    fwhm = 299792.458 / 50000  # fwhm = c / R

    # Full crires wavelength range=(9500, 53800)
    def __init__(self, name="CRIRES", FWHM=fwhm, wl_range=Starfish.grid["wl_range"]):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)
        # Sets the FWHM and wl_range


print(CRIRES_50k())

# Process the grid and save to the hdf5 file.
mygrid = CIFIST(norm=False, air=False, base=Starfish.grid["raw_path"], wl_range=[10000, 30000])  # Disable normalization to solar boloametic flux.
instrument = CRIRES_50k()
# HDF5Creator(GridInterface, filename, Instrument, ranges=None, key_name='t{0:.0f}g{1:.1f}', vsinis=None)
# Specify hdf5_path in config.yaml file.
creator = HDF5Creator(mygrid, Starfish.grid["hdf5_path"], instrument, key_name=Starfish.grid["key_name"])

creator.process_grid()
