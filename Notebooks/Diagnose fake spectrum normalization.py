# coding: utf-8

# # Diagnose fake spectrum normalization
#
# 7 Novemeber 2017

# In[1]:


import matplotlib.pyplot as plt
import numpy as np

from mingle.models.broadcasted_models import inherent_alpha_model
from mingle.utilities.norm import continuum
from mingle.utilities.simulation_utilities import spec_max_delta
from simulators.iam_module import prepare_iam_model_spectra, continuum_alpha


# In[2]:



def fake_simulation(wav, params1, params2, gamma, rv, chip=None, limits=(2070, 2180), noise=None):
    """Make a fake spectrum with binary params and radial velocities."""
    mod1_spec, mod2_spec = prepare_iam_model_spectra(params1, params2, limits)

    mod1_spec.plot()
    mod2_spec.plot()
    plt.show()
    # Estimated flux ratio from models
    if chip is not None:
        inherent_alpha = continuum_alpha(mod1_spec, mod2_spec, chip)
        print("inherent flux ratio = {}, chip={}".format(inherent_alpha, chip))

    # Combine model spectra with iam model
    iam_grid_func = inherent_alpha_model(mod1_spec.xaxis, mod1_spec.flux, mod2_spec.flux,
                                             rvs=rv, gammas=gamma)
    if wav is None:
        delta = spec_max_delta(mod1_spec, rv, gamma)
        assert np.all(np.isfinite(mod1_spec.xaxis))
        mask = (mod1_spec.xaxis > mod1_spec.xaxis[0] + delta) * (mod1_spec.xaxis < mod1_spec.xaxis[-1] - delta)
        wav = mod1_spec.xaxis[mask]
        print("wav masked", wav)

    iam_grid_models = iam_grid_func(wav).squeeze()

    print(iam_grid_models)
    assert np.all(np.isfinite(iam_grid_models))
    if isinstance(noise, (int, float)):
        snr = noise
    else:
        snr = None

    # Continuum normalize all iam_gird_models
    def axis_continuum(flux):
        """Continuum to apply along axis with predefined variables parameters."""
        return continuum(wav, flux, splits=50, method="exponential", top=5)

    iam_grid_continuum = np.apply_along_axis(axis_continuum, 0, iam_grid_models)

    iam_grid_models = iam_grid_models / iam_grid_continuum

    # grid_spectrum = Spectrum(xaxis=wav, flux=iam_grid_models)
    # iam_grid_models = grid_spectrum.normalize("exponential")
    # Add the noise
    from mingle.utilities.simulation_utilities import add_noise
    if snr is not None:
        iam_grid_models = add_noise(iam_grid_models, snr)

    if np.any(np.isnan(iam_grid_models)):
        print("there was some nans")
        pass
    return wav, iam_grid_models


def main(star, sim_num, params1, params2, gamma, rv, noise=None, suffix=None):
    params1 = [float(par) for par in params1.split(",")]
    params2 = [float(par) for par in params2.split(",")]

    x_wav, y_wav = fake_simulation(np.linspace(2090, 2150, 2000), params1, params2, gamma, rv, chip=1, noise=noise)

    x, y = fake_simulation(None, params1, params2, gamma, rv, chip=1, noise=noise)
    print(x)

    plt.plot(x, y, label="Fake simulation")
    plt.plot(x_wav, y_wav, ".", label="2k")

    plt.xlim([2070, 2170])
    plt.legend()
    plt.title("{0} simnum={1}, noise={2}\n host={3}, companion={4}".format(star, sim_num, noise, params1, params2))
    plt.legend()
    plt.show()

    # NEED to normalize at full wavelenght and then resample

    y_reinterp = np.interp(x_wav, x, y)
    # y_500_reinterp = np.interp(x_wav, x_500, y_500)
    # y_10k_reinterp = np.interp(x_wav, x_10k, y_10k)
    #  y_30k_reinterp = np.interp(x_wav, x_30k, y_30k)

    plt.plot(x_wav, y_wav, ".", label="x_wav")
    plt.plot(x_wav, y_reinterp, ".", label="org sampling.")
    #  plt.plot(x_wav, y_500_reinterp, ".", label="500")
    #  plt.plot(x_wav, y_10k_reinterp, ".", label="10k.")
    #   plt.plot(x_wav, y_30k_reinterp, ".", label="30k.")
    plt.title("Accessing renormaliation")
    plt.legend()
    plt.show()

    plt.plot(y_reinterp - y_wav, label="diff")
    # plt.plot(y_500_reinterp - y_wav, label="500 diff")
    # plt.plot(y_10k_reinterp - y_wav, label="10k diff")
    # plt.plot(y_30k_reinterp - y_wav, label="30k diff")
    plt.legend()
    plt.show()


# In[3]:


params1 = "5000,4.5, 0.0"
params2 = "3000,4.5, 0.0"
gamma = 20
rv = -7

main("test", "1", params1, params2, gamma, rv, noise=None, suffix=None)
