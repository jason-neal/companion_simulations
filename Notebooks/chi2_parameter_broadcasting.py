
# coding: utf-8

# # Interpolate wavelength on multiple dimensions
# 
# ### Jason Neal - 19th July 2017
# To try and interpolate N-D data along the first axis.
# 
# This is to be able to perfrom chisquare analsysis for many parameters.

# In[ ]:


import numpy as np
import scipy as sp
from scipy.stats import chisquare
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# The model we have is obs (w, f), and model (wm, fm).
# 
# the model is combined (x + (y*a)*v) * gamma    two doppler shifts of v and gamma.
# 
# We either need to be able to perform broadcasting inside Pyastronomy.dopplershift, or do it ourselves and interpolate.
# 

# In[ ]:


w = np.arange(20)
A = 1.7
S = 1.1
f = A * np.sin(w) + S

plt.plot(w, f, label="data")
plt.legend()
plt.show()


# In[ ]:


wm = np.linspace(-3,23, 50)
fm = np.sin(wm)
plt.plot(wm, fm, label="model")
plt.plot(w, f, label="data")
plt.show()


# # Add a second axis for the amplitude

# In[ ]:


a = np.arange(1.3, 2, 0.05)
print(a)
a.shape


# In[ ]:


fma  = fm[:, None] * a    # Each column is 
fma.shape


# In[ ]:


# make wavelength axis also the same
wma = wm[:, None] * np.ones_like(a)
wma.shape


# In[ ]:


# Need to interpolate fma from wma to w
# np.interp does not work on 2d.
w_func = sp.interpolate.interp1d(wm, fma, axis=0, kind="slinear")



# In[ ]:


fma_interp = w_func(w)
#fma_cube = w_func(w)
#fma_spl = w_func(w)
fma_interp.shape


# In[ ]:


plt.plot(w, fma_interp)
plt.plot(w, f,  "--", label="data")
plt.legend()
plt.show()


# In[ ]:


chi2 = np.sum((f[:, None] - fma_interp)**2 / fma_interp, axis=0)

plt.plot(a, chi2, label="chi2")
plt.legend()
plt.show()


# In[ ]:


# Find the minimum value
m = np.argmin(chi2)
a_min = a[m]
a_min


# # Add a third axis for a vertical shift
# 

# In[ ]:


shift = np.arange(0.1, 1.3, 0.1)
print(len(shift))
fmas = fma[:, :, None] + shift
fmas.shape


# In[ ]:


wmas = wma[:, :, None] * np.ones_like(shift)
wmas.shape


# In[ ]:


print(wm.shape)
print(fmas.shape)
w_sfunc = sp.interpolate.interp1d(wm, fmas, axis=0, kind="slinear")

fmas_interp = w_sfunc(w)
fmas_interp.shape


# In[ ]:


plt.plot(w, fmas_interp[:,3, :])
plt.plot(w, f,  "--", label="data")
plt.legend()
plt.show()


# In[ ]:


chi2s = np.sum((f[:, None, None] - fmas_interp)**2 / fmas_interp, axis=0)

plt.plot(a, chi2s, label="chi2")
plt.legend()
plt.show()


# In[ ]:


X, Y = np.meshgrid(shift, a)
print(X.shape)
plt.contourf(X, Y, chi2s)
plt.colorbar()
plt.plot()
plt.show()
chi2s.shape


# In[ ]:


c2min = chi2s.argmin()
print(c2min)
chi2s[np.unravel_index(c2min, chi2s.shape)]


# In[ ]:


np.unravel_index(976, (140, 7))



# In[ ]:


plt.contour(chi2s)
plt.show()


# In[ ]:





# # Interpolating different wavelength axis. 
# 
# Each wl dimension has a dopplershift added.
# 

# In[ ]:


c = 500
vc = (1 + np.arange(10) / c)
print(wm.shape)
print(vc.shape)
doppler =  wm[:, np.newaxis] *  vc

print(doppler.shape)
#print(doppler)



# In[ ]:


plt.plot(doppler, fmas[:,:,5])
plt.show()


# In[ ]:


# doppler_interp = sp.interpolate.interp1d(doppler, fm)
print(len(wm))
print(len(vc))
print(fma.shape)    # fma includes the amplitude also.
# Cannot inperpolate directly for all the different wavelengths at once. Therefore
dims = fmas.shape + (len(vc),)  # add extra arry to dim
print(dims)
result = np.empty(dims)
print(result.shape)
for i, v in enumerate(vc):
    wm_vc = wm * v
    #  print(wm_vc.shape)
    #  print(fma.shape)
    func = sp.interpolate.interp1d(wm_vc, fmas, axis=0, bounds_error=False, fill_value=99999)
    # print(func(wm))
    result[:,:, :, i] = func(wm)


# In[ ]:


print(result)


# # This lets me doppler shift the wavelength and return it to wm.
# 
# In the second case for model I will just want to return it to the wavelength values of the observation.

# In[ ]:


# interp to obs
func = sp.interpolate.interp1d(wm, result, axis=0, bounds_error=False, fill_value=np.nan)
fmasd = func(w)
chi2d = np.sum((f[:, None, None, np.newaxis] - fmasd)**2 / fmasd, axis=0)
chi2d


# In[ ]:


chi2d.shape


# In[ ]:


fmasd.shape


# In[ ]:


a.shape


# In[ ]:


# Try a 3d chisquare 

x_2 = chisquare(f[:, np.newaxis, np.newaxis, np.newaxis], fmasd, axis=0).statistic

x_2.argmin()
vals = np.unravel_index(x_2.argmin(), x_2.shape)
print("index of min = ", vals)   # returns the minimum index location




#  This provides a framework for chisquare of large arrays. for my simulations

# In[ ]:


plt.title("shift min")
plt.contourf(x_2[:,3,:])
plt.show()

plt.contourf(x_2[4,:,:])
plt.title("amplitude min")
plt.show()

plt.contourf(x_2[:,:,4])
plt.title("doppler min")
plt.show()


# Currently these plots do not look very informative. I will need to remove the bad interpolation values also.

# In[ ]:




