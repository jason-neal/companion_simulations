# coding: utf-8

# In[6]:


# Test plotting the test spectra



# In[7]:


from astropy.io import fits
import matplotlib.pyplot as plt
from mingle.utilities.spectrum_utils import load_spectrum
from simulators.bhm_module import bhm_helper_function

from mingle.utilities.crires_utilities import barycorr_crires_spectrum

# In[8]:


for star in ["hdtest2", "hdtest3"]:

    for chip in range(1, 5):
        obs_name, obs_params, output_prefix = bhm_helper_function(star.upper(), "001", chip)
        obs_spec = load_spectrum(obs_name)
        data = fits.getdata("/home/jneal/.handy_spectra/{}-001-mixavg-tellcorr_{}.fits".format(star.upper(), chip))

        plt.plot(data["wavelength"], data["flux"] + 0.01, label="fits")
        obs_spec.plot(label="load spec")
        plt.legend()
        plt.show()

# In[9]:


# Test with bayrrcorrection
for star in ["hdtest2", "hdtest3"]:

    for chip in range(1, 5):
        obs_name, obs_params, output_prefix = bhm_helper_function(star.upper(), "001", chip)
        obs_spec = load_spectrum(obs_name)
        data = fits.getdata("/home/jneal/.handy_spectra/{}-001-mixavg-tellcorr_{}.fits".format(star.upper(), chip))

        plt.plot(data["wavelength"], data["flux"] + 0.01, label="fits")
        obs_spec.plot(label="load spec")
        obs_spec = barycorr_crires_spectrum(obs_spec, extra_offset=None)
        obs_spec.plot(label="barycorr_")
        plt.legend()
        plt.show()



    # In[10]:

# Test with bayrrcorrection  with offset
for star in ["hdtest2", "hdtest3"]:

    for chip in range(1, 5):
        obs_name, obs_params, output_prefix = bhm_helper_function(star.upper(), "001", chip)
        obs_spec = load_spectrum(obs_name)
        data = fits.getdata("/home/jneal/.handy_spectra/{}-001-mixavg-tellcorr_{}.fits".format(star.upper(), chip))

        plt.plot(data["wavelength"], data["flux"] + 0.01, label="fits")
        obs_spec.plot(label="load spec")
        obs_spec = barycorr_crires_spectrum(obs_spec, extra_offset=25)
        obs_spec.plot(label="barycorr_25")
        plt.legend()
        plt.show()
