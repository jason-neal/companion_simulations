
# coding: utf-8

# # Chi sqaure analsys for the best host star model
# ## HD211847
# 
# 
# 
# 
# 

# In[ ]:


import sys

sys.path.append("/home/jneal/Phd/Codes/companion_simulations/")

sys.path.append("/home/jneal/Phd/Codes/companion_simulations/utilities/")
sys.path.append("../../")
sys.path.append("../")
import param_file
# from mingle.utilities.param_file import parse_paramfile
#from best_host_model_HD211847 import main

from param_file import parse_paramfile
from Chisqr_of_observation import select_observation, load_spectrum


# In[ ]:





# In[ ]:



handy_path = lambda x ,y : "/home/jneal/.hand_spectra/{0}-{1}-mixavg-tellcorr_{2}.fits".format(x, y, z)

star = "HD211847"
comp = ""   # set to b or c if needed
param_file = "/home/jneal/Phd/data/parameter_files/{}{}_params.dat".format(star, comp)
host_parameters = parse_paramfile(param_file, path=None)
obs_num = 1
chip = 1
obs_name_org = select_observation(star, obs_num, chip)
obs_name_new = handy_path(star, chip)
print(obs_name_org)
print(obs_name_new)


# In[ ]:


# Load observation
uncorrected_spectra = load_spectrum(obs_name)
observed_spectra = load_spectrum(obs_name)
observed_spectra = barycorr_crires_spectrum(observed_spectra, -22)
observed_spectra.flux /= 1.02


# In[ ]:





# In[ ]:


main()


# 

# In[ ]:





# In[ ]:




