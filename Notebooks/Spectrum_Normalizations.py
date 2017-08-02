
# coding: utf-8

# # Spectrum Continuum Normalization
# 
# ## Aim:
#    - To perform Chi^2 comparision between PHOENIX ACES spectra and my CRIRES observations.
#      
# ## Problem:
#    - The nomalization of the observed spectra
#    - Differences in the continuum normalization affect the chi^2 comparison when using mixed models of two different spectra. 
#    
# ### Proposed Solution:
#   - equation (1) from [Passegger 2016](https://arxiv.org/pdf/1601.01877.pdf) 
#           Fobs = F obs * (cont_fit model / cont_fit observation) where con_fit is a linear fit to the spectra.
# To take out and linear trends in the continuums and correct the amplitude of the continuum.
#    
#    
# In this notebook I outline what I do currently showing an example.
# 
# 
# 
# 

# In[ ]:


import copy
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# The obeservatios were originally automatically continuum normalized in the iraf extraction pipeline. 
# 
# I believe the continuum is not quite at 1 here anymore due to the divsion by the telluric spectra.

# In[ ]:


# Observation
obs = fits.getdata("/home/jneal/.handy_spectra/HD211847-1-mixavg-tellcorr_1.fits")

plt.plot(obs["wavelength"], obs["flux"])
plt.hlines(1, 2111, 2124, linestyle="--")
plt.title("CRIRES spectra")
plt.xlabel("Wavelength (nm)")
plt.show()


# In[ ]:


The two PHOENIX ACES spectra here are the first best guess of the two spectral components.


# In[ ]:


# Models
wav_model = fits.getdata("/home/jneal/Phd/data/PHOENIX-ALL/PHOENIX/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
wav_model /= 10   # nm
host = "/home/jneal/Phd/data/PHOENIX-ALL/PHOENIX/Z-0.0/lte05700-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
companion = "/home/jneal/Phd/data/PHOENIX-ALL/PHOENIX/Z-0.0/lte02600-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"

host_f = fits.getdata(host)
comp_f = fits.getdata(companion)
plt.plot(wav_model, host_f, label="Host")
plt.plot(wav_model, comp_f, label="Companion")
plt.title("Phoenix spectra")
plt.xlabel("Wavelength (nm)")
plt.legend()

plt.show()

mask = (2000 < wav_model) & (wav_model < 2200)
wav_model = wav_model[mask] 
host_f = host_f[mask] 
comp_f = comp_f[mask] 


plt.plot(wav_model, host_f, label="Host")
plt.plot(wav_model, comp_f, label="Companion")
plt.title("Phoenix spectra")
plt.legend()
plt.xlabel("Wavelength (nm)")
plt.show()


# 

# In[ ]:





# # Current Normalization
# I then continuum normalize the Phoenix spectrum locally around my observations 
# by fitting an **exponenital** to the continuum like so.
# 
# - Split the spectrum into 50 bins
# - Take median of 20 highest points in each bin.
# - Fix an exponetial
# - Evaulate at the orginal wavelength values
# - Divide original by the fit
# 

# In[ ]:


import copy
def local_normalization(wave, flux, splits=50, method="exponential", plot=False):
    r"""Local minimization for section of Phoenix spectra.

    Split spectra into many chunks and get the average of top 5\% in each bin.

    Fit to those points and normalize by that.
    """
    org_flux = copy.copy(flux)
    org_wave = copy.copy(wave)

    while len(wave) % splits != 0:
        # Shorten array untill can be evenly split up.
        wave = wave[:-1]
        flux = flux[:-1]

    flux_split = np.split(flux, splits)
    wav_split = np.split(wave, splits)

    wav_points = np.empty(splits)
    flux_points = np.empty(splits)

    for i, (w, f) in enumerate(zip(wav_split, flux_split)):
        wav_points[i] = np.median(w[np.argsort(f)[-20:]])  # Take the median of the wavelength values of max values.
        flux_points[i] = np.median(f[np.argsort(f)[-20:]])

    if method == "scalar":
        norm_flux = np.median(flux_split) * np.ones_like(org_wave)
    elif method == "linear":
        z = np.polyfit(wav_points, flux_points, 1)
        p = np.poly1d(z)
        norm_flux = p(org_wave)
    elif method == "quadratic":
        z = np.polyfit(wav_points, flux_points, 2)
        p = np.poly1d(z)
        norm_flux = p(org_wave)
    elif method == "exponential":
        z = np.polyfit(wav_points, np.log(flux_points), deg=1, w=np.sqrt(flux_points))
        p = np.poly1d(z)
        norm_flux = np.exp(p(org_wave))   # Un-log the y values.

    if plot:
        plt.subplot(211)
        plt.plot(wave, flux)
        plt.plot(wav_points, flux_points, "x-", label="points")
        plt.plot(org_wave, norm_flux, label='norm_flux')
        plt.legend()
        plt.subplot(212)
        plt.plot(org_wave, org_flux / norm_flux)
        plt.title("Normalization")
        plt.xlabel("Wavelength (nm)")
        plt.show()

    return org_flux / norm_flux



# In[ ]:


host_cont = local_normalization(wav_model, host_f, splits=50, method="exponential", plot=True)


# In[ ]:


comp_cont = local_normalization(wav_model, comp_f, splits=50, method="exponential", plot=True)


# Above the top is the unnormalize spectra, with the median points in orangeand the green line the continuum fit. The bottom plot is the contiuum normalized result

# In[ ]:


plt.plot(wav_model, comp_cont, label="Companion")
plt.plot(wav_model, host_cont-0.3, label="Host")
plt.title("Continuum Normalized ")
plt.xlabel("Wavelength (nm)")
plt.legend()
plt.show()

plt.plot(wav_model[20:200], comp_cont[20:200], label="Companion")
plt.plot(wav_model[20:200], host_cont[20:200], label="Host")
plt.title("Continuum Normalized - close up")
plt.xlabel("Wavelength (nm)")
ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_useOffset(False)
plt.legend()
plt.show()


# # Combining Spectra
# I then mix the models using a combination of the two spectra.
# In this case with NO RV shifts.

# In[ ]:


def mix(h, c, alpha):
    return (h + c * alpha) / (1 + alpha)

mix1 = mix(host_cont, comp_cont, 0.01)   # 1% of the companion spectra  
mix2 = mix(host_cont, comp_cont, 0.05)   # 5% of the companion spectra  

# plt.plot(wav_model[20:100], comp_cont[20:100], label="comp")
plt.plot(wav_model[20:100], host_cont[20:100], label="host")
plt.plot(wav_model[20:100], mix1[20:100], label="mix 1%")
plt.plot(wav_model[20:100], mix2[20:100], label="mix 5%")
plt.xlabel("Wavelength (nm)")
plt.legend()
plt.show()


# The companion is cooler there are many more deeper lines present in the spectra.
# Even a small contribution of the companion spectra reduce the continuum of the mixed spectra considerably.
# 
# When I compare these mixed spectra to my observations

# In[ ]:


mask = (wav_model > np.min(obs["wavelength"])) & (wav_model < np.max(obs["wavelength"]))

plt.plot(wav_model[mask], mix1[mask], label="mix 1%")
plt.plot(wav_model[mask], mix2[mask], label="mix 5%")
plt.plot(obs["wavelength"], obs["flux"], label="obs")
#plt.xlabel("Wavelength (nm)")
plt.legend()
plt.show()


plt.plot(wav_model[mask], mix2[mask], label="mix 5%")
plt.plot(wav_model[mask], mix1[mask], label="mix 1%")
plt.plot(obs["wavelength"], obs["flux"], label="obs")
plt.xlabel("Wavelength (nm)")
plt.legend()
plt.xlim([2112, 2117])
plt.ylim([0.9, 1.1])
plt.title("Zoomed")
plt.show()


# As you can see here my observations are above the continuum most of the time.
# What I have noticed is this drastically affects the chisquared result as the mix model is the one with the least amount of alpha.
# 
# I am thinking of renormalizing my observations by implementing equation (1) from [Passegger 2016](https://arxiv.org/pdf/1601.01877.pdf) *(Fundamental M-dwarf parameters from high-resolution spectra using PHOENIX ACES modesl)*
# 
#             F_obs = F_obs * (continuum_fit model / continuum_fit observation)
#             
# They fit a linear function to the continuum of the observation and computed spectra to account for *"slight differences in the continuum level and possible linear trends between the already noramlized spectra."* 
# 
# - One difference is that they say they normalize the **average** flux of the spectra to unity. Would this make a difference in this method.
# 
# 
# ## Questions
# - Would this be the correct approach to take to solve this? 
# - Should I renomalize the observations first as well?
# - Am I treating the cooler M-dwarf spectra correctly in this approach? 
# 

# In[ ]:


#Try the method:
from scipy.interpolate import interp1d
mix1_norm = local_normalization(wav_model, mix1, splits=50, method="linear", plot=True)
mix2_norm = local_normalization(wav_model, mix2, splits=50, method="linear", plot=True)
obs_norm = local_normalization(obs["wavelength"], obs["flux"], splits=20, method="linear", plot=True)

normalization1 = mix1 / mix1_norm
normalization2 = mix2 / mix2_norm
obs_renorm1 = obs_norm * interp1d(wav_model, normalization1)(obs["wavelength"])
obs_renorm2 = obs_norm * interp1d(wav_model, normalization2)(obs["wavelength"])



# In[ ]:


plt.plot(obs["wavelength"], obs_renorm1)
plt.plot(wav_model[mask], mix1[mask], label="mix 1%")
plt.show()


# In[ ]:


plt.plot(obs["wavelength"], obs_renorm2)
plt.plot(wav_model[mask], mix2[mask], label="mix 5%")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




