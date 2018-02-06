
# coding: utf-8

# # Code to extract out the chi2 values for many different SNR values combinations.
# Like excel sheet
# 

# In[1]:


import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
home = "/home/jneal/Phd/Analysis/fake_sims_with_var_teff1"
import pandas as pd
import sqlalchemy as sa
from mingle.utilities.db_utils import load_sql_table

# In[2]:


# ls /home/jneal/Phd/Analysis/fake_sims_with_var_teff1/analysis/


# In[3]:


noises=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20,
        50, 100, 150, 250, 500, 1000, 2000, 5000, 1000000]
teffs = [2300, 3400, 4000]


# In[4]:


pd.options.display.max_colwidth = 10
def load_min_chi2(teff, noises):
    df_store = pd.DataFrame()
    for snr in noises:   
        obsnum = 1
        starname = "NOISESCRIPT{}N{}".format(teff, snr)
        directory = os.path.join(home, "analysis", starname, "iam")
        
        dbname = f"{starname}-{obsnum}_coadd_iam_chisqr_results.db"
        try:
            table = load_sql_table(os.path.join(directory,dbname), verbose=False, echo=False)
        
            chi2_val = "coadd_chi2"
            dbdf = pd.read_sql(sa.select(table.c).order_by(table.c[chi2_val].asc()).limit(1), table.metadata.bind)
            dbdf["snr"] = snr   # Add SNR column
            df_store = dbdf.append(df_store)
        except Exception as e:
            print(e)
            print(f"Didn't get Database for {teff}-{snr}")
    df_store["median_alpha"] = df_store.apply(lambda row: np.median([row.alpha_1, row.alpha_2, row.alpha_3, row.alpha_4]), axis=1)
    return df_store


# In[5]:


df_teff = []
for teff in teffs:
            df_teff.append(load_min_chi2(teff, noises))


# In[6]:


def analyse_min_chi(df, teff):
    print("\nHost Temperature = 5200 K, Companion Temperature = {}".format(teff))
    print(df[["snr", "coadd_chi2", "teff_1", "teff_2", "median_alpha"]])
    print()
    ax = df.plot(x="snr", y="teff_1", style="o-", logx=True)
    plt.axhline(y=5200, color="k", linestyle="--")
    ax.set_xlabel("SNR")
    ax.set_ylabel("Teff [K]")
    plt.title("Host Temperature")

    ax2 = df.plot(x="snr", y="teff_2", style="o-", logx=True )
    plt.axhline(y=teff, color="k", linestyle="--")
    ax2.set_xlabel("SNR")
    ax2.set_ylabel("Teff [K]")
    plt.title("Companion Temperature")

    ax3=df.plot(x="snr", y="coadd_chi2", style="o-", logx=True)
    plt.title("Chi squared")
    ax3.set_xlabel("SNR")
    ax3.set_ylabel("$\chi^2$")
  
    plt.show()
    


# In[7]:


for i, teff in enumerate(teffs):
          analyse_min_chi(df_teff[i], teff)


# In[8]:


# Single model simulations
#/home/jneal/Phd/Analysis/sims_variable_params_same_snr/analysis/BHMNOISESCRIPT5200N0
#/home/jneal/Phd/Analysis/sims_variable_params_same_snr/analysis/BHMNOISESCRIPT5200N20
#/home/jneal/Phd/Analysis/sims_variable_params_same_snr/analysis/BHMNOISESCRIPT5200N50
#/home/jneal/Phd/Analysis/sims_variable_params_same_snr/analysis/BHMNOISESCRIPT5200N100
#/home/jneal/Phd/Analysis/sims_variable_params_same_snr/analysis/BHMNOISESCRIPT5200N1000


def load_min_bhm_chi2(teff, noises):
    df_store = pd.DataFrame()
    for snr in noises:   
        obsnum = 1
        starname = "BHMNOISESCRIPT{}N{}".format(teff, snr)
        directory = os.path.join(home, "analysis", starname, "bhm")
        
        dbname = f"{starname}-{obsnum}_coadd_bhm_chisqr_results.db"
        try:
            table = load_sql_table(os.path.join(directory,dbname), verbose=False, echo=False)
        
            chi2_val = "coadd_chi2"
            dbdf = pd.read_sql(sa.select(table.c).order_by(table.c[chi2_val].asc()).limit(1), table.metadata.bind)
            dbdf["snr"] = snr   # Add SNR column
            df_store = dbdf.append(df_store)
        except Exception as e:
            print(e)
            print(f"Didn't get Database for {teff}-{snr}")
    #df_store["median_alpha"] = df_store.apply(lambda row: np.median([row.alpha_1, row.alpha_2, row.alpha_3, row.alpha_4]), axis=1)
    return df_store


# In[9]:


noises=[0,  20, 50, 100,  1000]
bhm_teffs = [5200]
df_bhm_teff = []
for teff in bhm_teffs:
            df_bhm_teff.append(load_min_bhm_chi2(teff, noises))


# In[10]:


#/home/jneal/Phd/Analysis/sims_variable_params_same_snr/analysis/BHMNOISESCRIPT5200N50/bhm/BHMNOISESCRIPT520050-7_coadd_bhm_chisqr_results.db
def analyse_min_chi(df, teff):
    print("\nHost Temperature = {} K".format(teff))
    print(df[["snr", "coadd_chi2", "teff_1", "teff_2"]])#, "median_alpha"]])

    ax = df.plot(x="snr", y="teff_1", style="o-", logx=True )
    plt.axhline(y=teff, color="k", linestyle="--")
    ax.set_xlabel("SNR")
    ax.set_ylabel("Teff [K]")
    plt.title("Companion Temperature")

    ax3=df.plot(x="snr", y="coadd_chi2", style="o-", logx=True)
    plt.title("Chi squared")
    ax3.set_xlabel("SNR")
    ax3.set_ylabel("$\chi^2$")

    plt.show()
    


# In[11]:


for i, teff in enumerate(bhm_teffs):
          analyse_min_chi(df_teff[i], teff)

