
# coding: utf-8

# # Comparing Phoenix Spectra

# Plot Phoenix aces and Phoenix BT-Settl for same parameters
# 
# Also AMES Dusty Cond.

# #### FORMAT OF THE SPECTRA OUTPUT FILES
# 
# You can find the  pre-computed grids, also accessible via links on
# the bottom part of the simulator presentation page, or using this link:
# http://phoenix.ens-lyon.fr/Grids/
# 
# The file names contain the main parameters of the models:
# lte{Teff/10}-{Logg}{[M/H]}a[alpha/H].GRIDNAME.7.spec.gz/bz2/xz
# is the synthetic spectrum for the requested effective temperature
# (Teff),surface gravity (Logg), metallicity by log10 number density with
# respect to solar values ([M/H]), and alpha element enhencement relative     
# to solar values [alpha/H]. The model grid is also mentioned in the name.
# 
# Spectra are provided in an ascii format (\*.7.gz):
# 
# column1: wavelength in Angstroem
# column2: 10\*\*(F_lam + DF) to convert to Ergs/sec/cm\*\*2/A
# column3: 10\*\*(B_lam + DF) i.e. the blackbody fluxes of same Teff in same units.
# 
# Additional columns, obtained systematically when computing spectra using the
# Phoenix simulator, give the information to identify atomic and molecular
# lines. This information is used by the idl scripts lineid.pro and plotid.pro 
# which are provided in the user result package.  
#    
# With the stacked ascii format (\*.spec.gz files ) we have rather:
# 
# line1: Teff logg [M/H] of the model
# line2: number of wavelengths
# line3: F_lam(n) X 10\*\*DF , n=1,number of wavelengths
# lineX: B_lam(n) X 10\*\*DF , n=1,number of wavelengths
# 

# In[15]:


# In[ ]:


def load_phoenix(name):
    


# In[20]:



def comparison_plot(name, teff, logg, feh):
    
    pars = teff, logg, feh


# In[26]:


Artucus = [4300, 1.50, -0.5]
HD30501 = [5200, 4.5, 0.0]
ACES_bottom = [2300, 4.5, 0.0]
Sun = [5800, 4.5, 0.0]



# In[27]:



comparison_plot("Sun", *Sun)




# In[28]:



comparison_plot("ACES_bottom", *ACES_bottom)



# In[29]:



comparison_plot("Artucus", *Artucus)




# In[25]:



comparison_plot("HD30501", *HD30501)


