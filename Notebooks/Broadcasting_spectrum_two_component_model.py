
# coding: utf-8

# # Broadcasting a spectrum - Two spectral Components model

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


def two_comp_model(wav, model1, model2, alphas, rvs, gammas):
    # Make 2 component simulations, broadcasting over alpha, rv, gamma values.

    # Enable single scalar inputs (turn to 1d np.array)
    if not hasattr(alphas, "__len__"):
        alphas = np.asarray(alphas)[np.newaxis]
    if not hasattr(rvs, "__len__"):
        rvs = np.asarray(rvs)[np.newaxis]
    if not hasattr(gammas, "__len__"):
        gammas = np.asarray(gammas)[np.newaxis]
        print(len(gammas))

    am2 = model2[:,np.newaxis] * alphas           # alpha * Model2 (am2)
    print(am2.shape) 
    
    am2rv = np.empty(am2.shape + (len(rvs),))     # am2rv = am2 with rv doppler-shift
    print(am2rv.shape)
    for i, rv in enumerate(rvs):
        #nflux, wlprime = dopplerShift(wav, am2, rv)
        #am2rv[:, :, i] = nflux
        wav_i = (1 - rv / c) * wav
        am2rv[:, :, i] = interp1d(wav_i, am2, axis=0, bounds_error=False)(wav)
    
    # Normalize by (1 / 1 + alpha)
    am2rv = am2rv / (1 + alphas)[np.newaxis, :, np.newaxis]

    am2rvm1 = h[:, np.newaxis, np.newaxis] + am2rv                            # am2rvm1 = am2rv + model_1
    print(am2rvm1.shape)
    
    am2rvm1g = np.empty(am2rvm1.shape + (len(gammas),))   # am2rvm1g = am2rvm1 with gamma doppler-shift
    for j, gamma in enumerate(gammas):
        wav_j = (1 - gamma / 299792.458) * wav
        am2rvm1g[:, :, :, j] = interp1d(wav_j, am2rvm1, axis=0, bounds_error=False)(wav)
    
    return interp1d(w, am2rvm1g, axis=0)    # pass it the wavelength values to return
    


# In[ ]:



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
h = h/np.max(h)
c = c/np.max(c)


# In[ ]:


# Create a simulated spectrum
# Parameters
c_kms = 299792.458   # km/s
s_alpha = np.array([0.1])
s_rv    = np.array([1.5])
s_gamma = np.array([0.5])
answers = (s_alpha, s_rv, s_gamma)

# COMPACT SIMULATION
comp = interp1d((1 + s_rv / c_kms) * w, s_alpha * c, bounds_error=False)(w)
Sim_func = interp1d((1 - s_gamma / c_kms) * w, (h + comp) / (1 + s_alpha), bounds_error=False, axis=0)
sim_f_orgw = Sim_func(w)

sim_w = np.linspace(2114, 2115, 1024)
sim_f = Sim_func(sim_w)


# In[ ]:


# Compare output to tcm
tcm_sim_f = two_comp_model(w, h, c, s_alpha, s_rv, s_gamma)(sim_w)
ocm_sim_f = one_comp_model(w, h, s_gamma)(sim_w)


# In[ ]:


plt.close()
plt.plot(w, sim_f_orgw, label="org_w")
plt.plot(sim_w, sim_f, label="sim")
plt.plot(sim_w, np.squeeze(tcm_sim_f), label="tcm sim")
plt.plot(sim_w, np.squeeze(ocm_sim_f), label="ocm sim")
plt.legend()
plt.show()

sim_f.shape

# sim_w, sim_f are the observations to perform chisquared against!


# In[ ]:


alphas = np.linspace(0.1, 0.3, 40)
rvs = np.arange(1.1, 2, 0.05)
gammas = np.arange(-0.9, 1, 0.015)
print(len(alphas), len(rvs), len(gammas))


# In[ ]:


tcm = two_comp_model(w, h, c, alphas=alphas, rvs=rvs, gammas=gammas) 


# In[ ]:


# Two component model
tcm_obs = tcm(sim_w)
tcm_obs.shape


# In[ ]:


chi2 = chisquare(sim_f[:, np.newaxis, np.newaxis, np.newaxis], tcm_obs).statistic

print(chi2.shape)
min_indx = np.unravel_index(chi2.argmin(), chi2.shape)

print("sim results", alphas[min_indx[0]], min_rvs[indx[1]], gammas[min_indx[2]])
print("answer", answers)


# In[ ]:


# Putting resulted sim min values back into tcm model
res = two_comp_model(w, h, c, alphas[min_indx[0]], rvs[min_indx[1]], gammas[min_indx[2]])

res_f = res(sim_w)                  # Flux at the min min chisquare model evaulated at obs points.


# In[ ]:


# Compare to tcm generated simulation

chi2_tcm = chisquare(tcm_sim_f, tcm_obs).statistic
min_indx_tcm = np.unravel_index(chi2.argmin(), chi2.shape)

print("tcm results", alphas[min_indx_tcm[0]], rvs[min_indx_tcm[1]], gammas[min_indx_tcm[2]])
print("answer", answers)


# In[ ]:


# Putting resulted tcm sim min values back into tcm model
res_tcm = two_comp_model(w, h, c, alphas[min_indx[0]], rvs[min_indx[1]], gammas[min_indx[2]])
    
res_tcm_f = res_tcm(sim_w)    # Flux at the min min chisquare model evaulated at obs points.


# In[ ]:



plt.plot(sim_w, sim_f, "--", label="org")
plt.plot(sim_w, np.squeeze(res_f), label= "2 comp")
plt.plot(sim_w, np.squeeze(res_tcm_f), label="fit to tcm sim")
plt.title("Comparison to Simulation")
plt.legend()
plt.show()


# In[ ]:


plt.close()
plt.figure()


# In[ ]:


plt.figure()
plt.contourf(chi2[:,:,0])
plt.figure()
plt.contourf(chi2[0,:,:])


# In[ ]:


plt.figure()
plt.contourf(chi2[:,1,:])
plt.figure()


# In[ ]:



# Slice arrays to make contour maps

xslice = np.arange(0, chi2.shape[0], 5)
yslice = np.arange(0, chi2.shape[1], 5)
zslice = np.arange(0, chi2.shape[2], 5)

for xs in xslice:
    plt.figure()
    plt.contourf(chi2[xs, :, :])
    plt.colorbar()
    plt.title("x alpha = {}".format(alphas[xs]))
    plt.show()




# In[ ]:



for ys in yslice:
    plt.figure()
    plt.contourf(chi2[:, ys, :])
    plt.colorbar()
    plt.title("y rvs = {}".format(rvs[ys]))
    plt.show()


# In[ ]:



for zs in zslice:
    plt.figure()
    plt.contourf(chi2[:, :, zs])
    plt.colorbar()
    plt.title("z gammas = {}".format(gammas[zs]))
    plt.show()


# In[ ]:


for xs in np.concatenate([xslice, yslice, zslice]):
    plt.close()


# In[ ]:





# In[ ]:




