# Test reading hdf5 file that I created
import numpy as np

import Starfish
from Starfish.grid_tools import HDF5Interface

myHDF5 = HDF5Interface()
wl = myHDF5.wl
flux = myHDF5.load_flux(np.array([6100, 4.5, 0.0]))
