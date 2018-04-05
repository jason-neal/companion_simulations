
# coding: utf-8

# # Radius of Models:
# 
# 
# 

# In[1]:


from mingle.utilities.phoenix_utils import load_starfish_spectrum, load_btsettl_spectrum, all_aces_params, all_btsettl_params
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# ACES models
teffs, loggs, fehs, alphas = all_aces_params()

TEFF, LOGGS = np.meshgrid(teffs, loggs)

REFFS = np.empty_like(TEFF)
for i, teff in enumerate(teffs):
    for j, logg in enumerate(loggs):
        try:
            spec = load_starfish_spectrum([teff, logg, 0.0], hdr=True)
            reff = spec.header["PHXREFF"]
        except:
            reff = 2e11
        REFFS[j][i] = reff
            
cbar = plt.contourf(TEFF, LOGGS, REFFS)
cbar = plt.colorbar()
plt.show()
        
        


# In[11]:


# BT-SETTL models
teffs2, loggs2, fehs2, alphas2 = all_btsettl_params()
teffs2 = teffs2[teffs2 >= 3]
print(teffs2, loggs2, fehs2, alphas2)
TEFF2, LOGGS2 = np.meshgrid(teffs2, loggs2)

REFFS2 = np.empty_like(TEFF2)
for i, teff in enumerate(teffs2):
    for j, logg in enumerate(loggs2):
        try:
            spec = load_btsettl_spectrum([teff, logg, 0.0], hdr=True)
            reff = spec.header["PHXREFF"]
        except:
            #print("invalid model {}-{}".format(teff, logg))
            reff = 0
        REFFS2[j][i] = reff
            
cbar = plt.contourf(TEFF2, LOGGS2, REFFS2)
cbar = plt.colorbar()
plt.show()
        
        


# In[10]:


spec = load_btsettl_spectrum([2300, 4.5, 0.0], hdr=True)


# In[ ]:


get_ipython().system('pwd')

