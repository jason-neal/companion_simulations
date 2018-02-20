#!/usr/bin/env python
# Test reading hdf5 file that I created
import numpy as np

from Starfish.grid_tools import HDF5Interface

my_hdf5 = HDF5Interface()
wl = my_hdf5.wl
flux = my_hdf5.load_flux(np.array([6100, 4.5, 0.0]))
