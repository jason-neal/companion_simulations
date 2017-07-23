# Test reading hdf5 file that I created
import Starfish
from Starfish.grid_tools import HDF5Interface
import numpy as np

my_hdf5 = HDF5Interface()
wl = my_hdf5.wl
flux = my_hdf5.load_flux(np.array([6100, 4.5, 0.0]))
