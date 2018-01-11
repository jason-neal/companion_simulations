
# coding: utf-8

# ## Apply Beta-sigma SNR estimates on the CRIRES Spectra
# 
# Using the berved spectra
# 
# Should compare weith berved masked but assume it will be small differences.
# 
# 

# In[42]:


from astropy.io import fits
import os
from PyAstronomy import pyasl
import matplotlib.pyplot as plt
import numpy as np


# In[43]:


file = "/home/jneal/.handy_spectra/HD30501-1-mixavg-tellcorr_1_bervcorr.fits"

data = fits.getdata(file)

plt.plot(data["wavelength"], data["flux"])

plt.show()


# In[61]:


def betasigma_spectra(file, Nmax=5, j=1, arb=False, returnMAD=True):
    data = fits.getdata(file)
    xi, yi = data["wavelength"], data["flux"]

    mdiff = np.max(np.abs(np.diff(flux)))
    print("""Maximum absolute difference between consecutive
        values of flux: """, mdiff)

    nd = len(flux)
    print("Number of 'data points': ", nd)
    print()
    print("Very Rough std = {}".format(np.std(flux)))

    # Create class instance for equidistant sampling
    if arb:
        bsarb = pyasl.BSArbSamp()
    else:
        bseq = pyasl.BSEqSamp()
    
    # Specify jump parameter (j) for construction of beta sample
    j = j

    # Order of approximation to use
    Ns = range(Nmax+1)

    # Use to store noise estimates
    smads, dsmads = [], []

    # Loop over orders of approximation between 0 and 3
    for N in Ns:
        print("Order of approximation (N): ", N)

        # Get estimates of standard deviation based on robust (MAD-based) estimator
        if arb:
            smad, dsmad = bsarb.betaSigma(xi, yi, N, j, returnMAD=returnMAD)
            print("    Size of beta sample: ", len(bsarb.betaSample))
        else:
            smad, dsmad = bseq.betaSigma(yi, N, j, returnMAD=returnMAD)
            print("    Size of beta sample: ", len(bseq.betaSample))
        print("    Robust estimate of noise std: %6.5f +/- %6.5f" % (smad, dsmad))
        # Save result
        smads.append(smad)
        dsmads.append(dsmad)

    # Plot g(t) and the synthetic data
    plt.subplot(2,1,1)
    plt.title("Data (top) and noise estimates (bottom)")
    plt.plot(xi, yi, 'b.-', label="flux")
    #plt.errorbar(ti, yi, yerr=np.ones(nd)*istd, fmt='r+', label="$y_i$")
    plt.legend()
    plt.subplot(2,1,2)
    plt.title("N=0 is insufficient")
    plt.errorbar(Ns, smads, yerr=dsmads, fmt='k+', label="Noise estimates")
    #plt.plot([min(Ns)-0.5, max(Ns)+0.5], [np.std(flux)]*2, 'k--', label="Rough value")
    plt.legend()
    plt.xlabel("Order of approximation (N)")
    plt.ylabel("Noise STD")
    plt.tight_layout()
    plt.show()


# In[64]:


files = ["/home/jneal/.handy_spectra/HD30501-1-mixavg-tellcorr_1_bervcorr.fits",
         "/home/jneal/.handy_spectra/HD30501-1-mixavg-tellcorr_2_bervcorr.fits",
         "/home/jneal/.handy_spectra/HD30501-1-mixavg-tellcorr_3_bervcorr.fits",
         "/home/jneal/.handy_spectra/HD30501-1-mixavg-tellcorr_4_bervcorr.fits"]


for file in files:
    print(os.path.split(file)[-1])
    print("j=1")
    betasigma_spectra(file)
    print("j=1")
    betasigma_spectra(file, j=2)
    print("j=3")
    betasigma_spectra(file, j=2)


# In[63]:


files = ["/home/jneal/.handy_spectra/HD30501-1-mixavg-tellcorr_3_bervcorr.fits",
         "/home/jneal/.handy_spectra/HD211847-1-mixavg-tellcorr_3_bervcorr.fits",
         "/home/jneal/.handy_spectra/HD202206-1-mixavg-tellcorr_3_bervcorr.fits",
         "/home/jneal/.handy_spectra/HD4747-1-mixavg-tellcorr_3_bervcorr.fits",
         "/home/jneal/.handy_spectra/HD162020-1-mixavg-tellcorr_3_bervcorr.fits",
         "/home/jneal/.handy_spectra/HD168443-1-mixavg-tellcorr_3_bervcorr.fits",
         "/home/jneal/.handy_spectra/HD167665-1a-mixavg-tellcorr_3_bervcorr.fits"]


for file in files:
    print(os.path.split(file)[-1])
    betasigma_spectra(file, j=2)
    


# In[65]:



These spectra seem to have SNR ~ 300-900 in the continuum from Beta simga estimates.

