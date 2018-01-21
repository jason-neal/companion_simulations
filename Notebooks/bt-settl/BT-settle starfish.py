
# coding: utf-8

# In[1]:


# Play with Starfish BTSETTL Grids:


# In[7]:


import numpy as np

import Starfish
#from Starfish.grid_tools import PHOENIXGridInterfaceNoAlpha as PHOENIX
from Starfish.grid_tools import HDF5Creator, Instrument

from Starfish.grid_tools import BTSettlGridInterface as BTSETTL


# In[8]:


class CRIRES_50k(Instrument):
    '''CRIRES instrument at R=50000.'''
    fwhm = 299792.458 / 50000     # fwhm = c / R
    # Full crires wavelength range=(9500, 53800)
    def __init__(self, name="CRIRES", FWHM=fwhm, wl_range=Starfish.grid["wl_range"]):
        super().__init__(name=name, FWHM=FWHM, wl_range=wl_range)
        # Sets the FWHM and wl_range
        


# In[10]:


mygrid = BTSETTL(norm=False, air=False, base=Starfish.grid["btsettle_raw"])    # Disable normalization to solar boloametic flux.
instrument = CRIRES_50k()
# HDF5Creator(GridInterface, filename, Instrument, ranges=None, key_name='t{0:.0f}g{1:.1f}', vsinis=None)
# Specify hdf5_path in config.yaml file.
creator = HDF5Creator(mygrid, Starfish.grid["btsettl_hdf5_path"], instrument, ranges=Starfish.grid["parrange"])




# Need to change starfish BT-SETTL to change how it handles param_names and points

# Might need own interface as there is hardcoded folder CXXXX2011


# In[11]:


creator.process_grid()


# # Loading BTSETTL Directly

# In[6]:


from Starfish.grid_tools import load_BTSettl
load_BTSettl(2200, 4.5, 0.0)

