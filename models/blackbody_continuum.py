"""Black body fit to phoenix spectra.

Sometimes works, but spectral lines and depressions are a problem.
"""
import numpy as np
from astropy.io import fits
from astropy import units as u
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
from astropy.analytic_functions import blackbody_lambda
from spectrum_overload.Spectrum import Spectrum


def blackbody_residual(params, x, data):
    """Using lmfit to fit black body.

    Blackbody_lambda function parameters are:
    Parameters
    ----------
    in_x : number, array-like, or Quantity
        Frequency, wavelength, or wave number. If not a Quantity, it is assumed to be in Angstrom.
    temperature : number, array-like, or Quantity
        Blackbody temperature. If not a Quantity, it is assumed to be in Kelvin.

    Returns
    -------
    flux: Quantity
        Blackbody monochromatic flux in ergcm−2s−1A˚−1sr−1.
    """
    temp = params["temp"]
    scale = params["scale"]
    model = blackbody_lambda(x * u_nm, temp * u_k) * scale
    return data - model.value


u_nm = u.nanometer
u_k = u.K

host_temp = 5400
comp_temp = 2400
# Load in some phoenix data and make a simulation
phoenix_wl = fits.getdata("/home/jneal/Phd/data/fullphoenix/phoenix.astro.physik.uni-goettingen.de/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits") / 10

host_phoenix = "/home/jneal/Phd/data/fullphoenix/phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte{:05d}-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(host_temp)

comp_phoenix = "/home/jneal/Phd/data/fullphoenix/phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte{:05d}-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(comp_temp)

unnorm_host_spec = Spectrum(flux=fits.getdata(host_phoenix), xaxis=phoenix_wl)
unnorm_comp_spec = Spectrum(flux=fits.getdata(comp_phoenix), xaxis=phoenix_wl)


# Waveleght limits. The result is sensitive to these limits.
min_wav = 500
max_wav = 3500

unnorm_host_spec.wav_select(min_wav, max_wav)
unnorm_comp_spec.wav_select(min_wav, max_wav)

# Black body from spectrum temp.
norm_host_spec = Spectrum(flux=unnorm_host_spec.flux / blackbody_lambda(phoenix_wl * u_nm, host_temp), xaxis=phoenix_wl)
norm_comp_spec = Spectrum(flux=unnorm_comp_spec.flux / blackbody_lambda(phoenix_wl * u_nm, host_temp), xaxis=phoenix_wl)

norm_host_spec.wav_select(min_wav, max_wav)
norm_comp_spec.wav_select(min_wav, max_wav)

plt.subplot(311)
plt.plot(norm_host_spec.xaxis, norm_host_spec.flux, label="Host")
plt.plot(norm_comp_spec.xaxis, norm_comp_spec.flux, label="Comp")
plt.title("Unnormalized")
plt.legend()

plt.subplot(312)
plt.plot(norm_host_spec.xaxis, unnorm_host_spec.flux - norm_host_spec.flux, label="Host")
plt.plot(norm_comp_spec.xaxis, unnorm_comp_spec.flux - norm_comp_spec.flux, label="Comp")
plt.title("/ blackbody")
plt.legend()

plt.subplot(313)
plt.plot(phoenix_wl, blackbody_lambda(phoenix_wl * u_nm, host_temp * u_k), label="Host")
plt.plot(phoenix_wl, blackbody_lambda(phoenix_wl * u_nm, comp_temp * u_k), label="Comp")
plt.title("Blackbody")
plt.xlim([min_wav, max_wav])
plt.legend()
plt.show()

# Fit to models
host_params = Parameters()
host_params.add('temp', value=host_temp, min=2000, max=10000)
host_params.add('scale', value=1e9)

comp_params = Parameters()
comp_params.add('temp', value=comp_temp, min=2000, max=10000)
comp_params.add('scale', value=1e9)

host_out = minimize(blackbody_residual, host_params, args=(unnorm_host_spec.xaxis, unnorm_host_spec.flux))
comp_out = minimize(blackbody_residual, comp_params, args=(unnorm_comp_spec.xaxis, unnorm_comp_spec.flux))

print("Host temp", host_temp)
host_out.params.pretty_print()
print("Comp temp", comp_temp)
comp_out.params.pretty_print()

data = unnorm_host_spec.flux
data_wl = unnorm_host_spec.xaxis
bins = 2000
# Trying to get the continum with binning
bin_means = (np.histogram(data, bins, weights=data)[0] /
             np.histogram(data, bins)[0])
bin_wl = (np.histogram(data_wl, bins, weights=data_wl)[0] /
          np.histogram(data_wl, bins)[0])

plt.plot(unnorm_host_spec.xaxis, unnorm_host_spec.flux, label="Host {}K".format(host_temp))
plt.plot(unnorm_comp_spec.xaxis, unnorm_comp_spec.flux, label="Comp {}K".format(comp_temp))

plt.plot(unnorm_host_spec.xaxis, blackbody_lambda(unnorm_host_spec.xaxis * u_nm, host_out.params["temp"] * u_k) * host_out.params["scale"], label="Host bb fit {}K".format(int(host_out.params["temp"].value)))

plt.plot(unnorm_host_spec.xaxis, blackbody_lambda(unnorm_host_spec.xaxis * u_nm, comp_out.params["temp"] * u_k) * comp_out.params["scale"], label="Comp bb fit {}K".format(int(comp_out.params["temp"].value)))
# plt.plot(bin_wl, bin_means, "+--", label="bin mean")
plt.legend()
plt.show()
