# coding: utf-8

# # Broadcasting on a spectrum - One component model

# In[ ]:


from astropy.io import fits
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
from scipy.stats import chisquare
from PyAstronomy.pyasl import dopplerShift
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib')


# In[ ]:


def one_comp_model(wav, model1, gammas):
    # Make 1 component simulations, broadcasting over gamma values.

    # Enable single scalar inputs (turn to 1d np.array)
    if not hasattr(gammas, "__len__"):
        gammas = np.asarray(gammas)[np.newaxis]
        print(len(gammas))

    m1 = model1
    print(model1.shape)

    m1g = np.empty(model1.shape + (len(gammas),))  # am2rvm1g = am2rvm1 with gamma doppler-shift
    print(m1g.shape)
    for j, gamma in enumerate(gammas):
        wav_j = (1 + gamma / 299792.458) * wav
        m1g[:, j] = interp1d(wav_j, m1, axis=0, bounds_error=False)(wav)

    return interp1d(w, m1g, axis=0)  # pass it the wavelength values to return


# In[ ]:


# Load in the data
wav = "/home/jneal/Phd/data/phoenixmodels/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
host = "/home/jneal/Phd/data/phoenixmodels/HD30501-lte05200-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
comp = "/home/jneal/Phd/data/phoenixmodels/HD30501b-lte02500-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"

w = fits.getdata(wav) / 10
h = fits.getdata(host)
c = fits.getdata(comp)

# In[ ]:


mask = (2111 < w) & (w < 2117)

w = w[mask]
h = h[mask]
c = c[mask]

# crude normalization
h = h / np.max(h)
c = c / np.max(c)

# In[ ]:


# Create a simulated spectrum
# Parameters
c_kms = 299792.458  # km/s
# s_alpha = np.array([0.1])
# s_rv    = np.array([1.5])
s_gamma = np.array([0.5])
answers = (s_gamma,)

# Compact simulation of one component
# comp = interp1d((1 + s_rv / c_kms) * w, s_alpha * c, bounds_error=False)(w)
Sim_func = interp1d((1 + s_gamma / c_kms) * w, h, bounds_error=False, axis=0)
sim_f_orgw = Sim_func(w)

sim_w = np.linspace(2114, 2115, 1024)
sim_f = Sim_func(sim_w)

# In[ ]:


# Simulate with ocm function
sim_ocm_f = one_comp_model(w, h, s_gamma)(sim_w)

# In[ ]:


plt.close()
plt.plot(w, sim_f_orgw, label="org_w")
plt.plot(sim_w, sim_f, label="sim")
plt.plot(sim_w, np.squeeze(sim_ocm_f), label="ocm sim")
plt.legend()
plt.show()

sim_f.shape

# sim_w, sim_f are the observations to perform chisquared against!


# # Parameters for chi-sqruare map

# In[ ]:


gammas = np.arange(-0.9, 1, 0.015)
print(len(gammas))

# In[ ]:


ocm = one_comp_model(w, h, gammas=gammas)

# In[ ]:


# One component model
ocm_obs = ocm(sim_w)  # Interpolate to observed values.
ocm_obs.shape

# # Calcualte Chi-Square

# In[ ]:


chi2 = chisquare(sim_f[:, np.newaxis], ocm_obs).statistic
chi2.shape

# In[ ]:


min_indx = np.unravel_index(chi2.argmin(), chi2.shape)

print(gammas[min_indx[0]])

# In[ ]:


# Compare to ocm generated simulation
chi2_ocm = chisquare(sim_ocm_f, ocm_obs).statistic
min_indx_ocm = np.unravel_index(chi2.argmin(), chi2.shape)

# ocm_chi2_ocm = chisquare(ocm_sim_f[:, np.newaxis], ocm_obs).statistic
# min_indx_ocm = np.unravel_index(chi2.argmin(), chi2.shape)
print("sim results =", gammas[min_indx[0]])
print("ocm results =", gammas[min_indx_ocm[0]])  # observation simulated with the ocm model
print("answer", answers)

# In[ ]:


# Putting resulted min values back into ocm

res = one_comp_model(w, h, gammas[min_indx[0]])
res_sim = res(sim_w)

res_ocm = one_comp_model(w, h, gammas[min_indx_ocm[0]])
res_sim_ocm = res_ocm(sim_w)

# In[ ]:


print(answers)
plt.plot(sim_w, sim_f, "--", label="Obs")
plt.plot(sim_w, np.squeeze(res_sim) + 0.01, label="1 comp")
plt.plot(sim_w, np.squeeze(res_sim_ocm) + 0.02, label="ocm 1 comp")
plt.legend()
plt.show()

# In[ ]:


plt.close()
plt.figure()

# In[ ]:


plt.figure()
plt.plot(gammas, chi2)
plt.xlabel("gammas")
plt.ylabel("Chisquare")

# In[ ]:


plt.figure()
plt.contourf(chi2[:, 1, :])

# In[ ]:


plt.close()
plt.close()


# In[ ]:
