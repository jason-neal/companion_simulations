"""Try fitting with lmfit."""
import lmfit
import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt
from lmfit import minimize, Parameters

from spectrum_overload.Spectrum import Spectrum
from models.alpha_model import double_shifted_alpha_model
from Planet_spectral_simulations import simple_normalization
from utilities.phoenix_utils import local_normalization, spec_local_norm


def alpha_model_residual(params, x, data, eps_data, host_models, companion_models):
    """Residaul function to use with lmfit Minimizer."""
    alpha = params['alpha'].value
    rv1 = params['rv1'].value
    rv2 = params['rv2'].value
    host_index = params['host_index'].value
    companion_index = params['companion_index'].value
    limits = [params['min_limit'].value, params['max_limit'].value]

    host = host_models[host_index]
    companion = companion_models[companion_index]

    print(host_models)

    print("x", x, "len x", len(x))
    print("alpha", alpha)
    print("rv1, rv2 ", rv1, rv2)
    print("host", host.xaxis, host.flux)
    print("companion", companion.xaxis, companion.flux)
    print(len(host), len(companion))
    print(limits)
    model = double_shifted_alpha_model(alpha, rv1, rv2, host, companion, limits, new_x=x)

    print(data)
    print(model)
    print(model.xaxis)
    print(model.flux)
    print(eps_data)
    return (data - model.flux) / eps_data


host_temp = 5300
comp_temp = 2300
# Load in some phoenix data and make a simulation
phoenix_wl = fits.getdata("/home/jneal/Phd/data/fullphoenix/phoenix.astro.physik.uni-goettingen.de/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits") / 10

host_phoenix = "/home/jneal/Phd/data/fullphoenix/phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte{:05d}-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(host_temp)

comp_phoenix = "/home/jneal/Phd/data/fullphoenix/phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte{:05d}-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(comp_temp)

unnorm_host_spec = Spectrum(flux=fits.getdata(host_phoenix), xaxis=phoenix_wl)
unnorm_comp_spec = Spectrum(flux=fits.getdata(comp_phoenix), xaxis=phoenix_wl)

min_wav = 2050
max_wav = 2250

unnorm_host_spec.wav_select(min_wav, max_wav)
unnorm_comp_spec.wav_select(min_wav, max_wav)

# local normalization
norm_host_flux = local_normalization(unnorm_host_spec.xaxis, unnorm_host_spec.flux, method="exponential", plot=False)
norm_comp_flux = local_normalization(unnorm_comp_spec.xaxis, unnorm_comp_spec.flux, method="exponential", plot=False)

norm_host_spec = spec_local_norm(unnorm_host_spec, method="exponential")
norm_comp_spec = spec_local_norm(unnorm_comp_spec, method="exponential")

double_norm_host = spec_local_norm(norm_host_spec, method="quadratic")
host_spec = simple_normalization(unnorm_host_spec)
comp_spec = simple_normalization(unnorm_comp_spec)

plot = 0
if plot:
    plt.subplot(311)
    plt.plot(unnorm_host_spec.xaxis, unnorm_host_spec.flux, label="Host")
    plt.plot(unnorm_comp_spec.xaxis, unnorm_comp_spec.flux, label="Comp")
    plt.title("Unnormalized")
    plt.legend()

    plt.subplot(312)
    plt.plot(norm_comp_spec.xaxis, norm_comp_spec.flux, label="Comp")
    plt.plot(norm_host_spec.xaxis, norm_host_spec.flux, label="Host")
    plt.title("Local normalization.")
    plt.legend()

    plt.subplot(313)
    plt.plot(comp_spec.xaxis, comp_spec.flux, label="Comp")
    plt.plot(host_spec.xaxis, host_spec.flux, label="Host")
    plt.title("Simple normalization")
    plt.legend()
    plt.show()

host_models = [norm_host_spec]
comp_models = [norm_comp_spec]


def vel_vect(wav, ref_wav=None):
    """Convert wavelength to velocity vector."""
    if ref_wav is None:
        ref_wav = np.median(wav)
    v = (wav - ref_wav) * 299792.458 / ref_wav   # km/s
    return v


alpha_val = 0.1
rv1_val = 0
rv2_val = -50
host_val = norm_host_spec
companion_val = norm_comp_spec
limits_val = [2100, 2180]

data_spec = double_shifted_alpha_model(alpha_val, rv1_val, rv2_val, host_val, companion_val, limits_val)
if plot:
    plt.plot(vel_vect(comp_spec.xaxis), comp_spec.flux, label="Comp")
    plt.plot(vel_vect(host_val.xaxis), host_val.flux, label="Host")
    plt.plot(vel_vect(data_spec.xaxis), data_spec.flux, label=" CRIRES Simulation")
    plt.xlim(limits_val)
    plt.legend()
    plt.show()

data_spec.add_noise(snr=350)

params = Parameters()
params.add('alpha', value=0.4, min=0, max=0.5)
params.add('rv1', value=0., min=-100, max=100, brute_step=0.5, vary=False)
params.add('rv2', value=-10., min=-200, max=200, brute_step=0.5, vary=True)
params.add('host_index', value=0, vary=False, brute_step=1)
params.add('companion_index', value=0, vary=False, brute_step=1)
params.add('min_limit', value=2100, vary=False)
params.add('max_limit', value=2180, vary=False)

out = minimize(alpha_model_residual, params, args=(data_spec.xaxis, data_spec.flux,
               np.ones_like(data_spec.flux), host_models, comp_models))

print(out)
out.params.pretty_print()
print(lmfit.report_fit(out))

fit_params = out.params

result = double_shifted_alpha_model(fit_params['alpha'].value, fit_params['rv1'].value,
                                    fit_params['rv2'].value, host_models[fit_params['host_index'].value],
                                    comp_models[fit_params['companion_index'].value], [fit_params['min_limit'].value,
                                    fit_params['max_limit'].value])


plt.plot(data_spec.xaxis, data_spec.flux, label="Simulation")
plt.plot(phoenix_wl, )
plt.plot(result.xaxis, result.flux, label="Returned fit")
plt.legend()
plt.show()

# Need to try a DIFFERENT OPTIMIZER
