# coding: utf-8

# In[ ]:


# Monitor fake detection processing etc.


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
from spectrum_overload import Spectrum

from mingle.models.broadcasted_models import inherent_alpha_model, independent_inherent_alpha_model
from mingle.utilities.chisqr import chi_squared
from mingle.utilities.phoenix_utils import load_starfish_spectrum

# In[3]:


snr = 300
sim_num = 3
starname = "HDSIM4"
params1 = [6000, 4.5, 0.0]
params2 = [5800, 4.5, 0.0]
gamma = 45
rv = -30

normalization_limits = [2000, 2300]

mod1_spec_scaled = load_starfish_spectrum(params1, limits=normalization_limits,
                                          hdr=True, normalize=False, area_scale=True,
                                          flux_rescale=True)
mod1_spec_unscaled = load_starfish_spectrum(params1, limits=normalization_limits,
                                            hdr=True, normalize=False, area_scale=False,
                                            flux_rescale=True)

mod2_spec_scaled = load_starfish_spectrum(params2, limits=normalization_limits,
                                          hdr=True, normalize=False, area_scale=True,
                                          flux_rescale=True)
mod2_spec_unscaled = load_starfish_spectrum(params2, limits=normalization_limits,
                                            hdr=True, normalize=False, area_scale=False,
                                            flux_rescale=True)

for name, mod1_spec, mod2_spec in zip(["area scaled", "area unscaled"],
                                      [mod1_spec_scaled, mod1_spec_unscaled],
                                      [mod2_spec_scaled, mod2_spec_unscaled]):
    mod1_spec = mod1_spec.remove_nans()
    mod2_spec = mod2_spec.remove_nans()
    mod1_spec.wav_select(2000, 2200)
    mod2_spec.wav_select(2000, 2200)

    plt.plot(mod1_spec.xaxis, mod1_spec.flux, label="mod1")
    plt.plot(mod2_spec.xaxis, mod2_spec.flux, label="mod2")
    plt.title(name)
plt.show()

# In[4]:


# Need to sample the spectrum away from the ends so that you don't get nans at the ends
# The chisqr does not like nans
sample_x = np.linspace(2112, 2145, 1024)


# In[5]:


def join_with_broadcast_spectrum(mod1, mod2, rv, gamma, new_x, independent=False):
    if independent:
        broadcast_result = independent_inherent_alpha_model(mod1.xaxis, mod1.flux, mod2.flux,
                                                            rvs=rv, gammas=gamma, independent_rv=True)
    else:
        broadcast_result = inherent_alpha_model(mod1.xaxis, mod1.flux, mod2.flux,
                                                rvs=rv, gammas=gamma, independent_rv=True)

    broadcast_values = broadcast_result(new_x)
    return Spectrum(flux=broadcast_values.squeeze(), xaxis=new_x)


def join_with_broadcast(mod1, mod2, rv, gamma, new_x, independent=False):
    if independent:
        broadcast_result = independent_inherent_alpha_model(mod1.xaxis, mod1.flux, mod2.flux,
                                                            rvs=rv, gammas=gamma, independent_rv=True)
    else:
        broadcast_result = inherent_alpha_model(mod1.xaxis, mod1.flux, mod2.flux,
                                                rvs=rv, gammas=gamma, independent_rv=True)
    broadcast_values = broadcast_result(new_x)
    return broadcast_values.squeeze()


gammas = np.linspace(-100, 100, 50)
rvs = np.linspace(-100, 100, 60)

for independent in (True, False):
    print("independant ", independent)

    print("fake data")
    fake_data = join_with_broadcast_spectrum(mod1_spec_scaled, mod2_spec_scaled,
                                             rv, gamma, sample_x, independent=independent)

    gamma_grid_data = join_with_broadcast(mod1_spec_scaled, mod2_spec_scaled,
                                          [-6], gammas, sample_x, independent=independent)
    rv_grid_data = join_with_broadcast(mod1_spec_scaled, mod2_spec_scaled, rvs,
                                       [10], sample_x, independent=independent)
    dual_grid_data = join_with_broadcast(mod1_spec_scaled, mod2_spec_scaled,
                                         rvs, gammas, sample_x, independent=independent)

    for normalize in (True, False):
        print("normalizing", normalize)

        fake_data.remove_nans()

        print(fake_data.flux.shape)
        print(gamma_grid_data.shape)
        print(rv_grid_data.shape)
        print(dual_grid_data.shape)

        gamma_chi2 = chi_squared(fake_data.flux[:, np.newaxis], gamma_grid_data)
        rv_chi2 = chi_squared(fake_data.flux[:, np.newaxis], rv_grid_data)
        dual_chi2 = chi_squared(fake_data.flux[:, np.newaxis, np.newaxis], dual_grid_data)

        plt.plot(gammas, gamma_chi2)
        plt.title("gamma chi2")
        plt.show()

        plt.plot(rvs, rv_chi2)
        plt.title("rv chi2")
        plt.show()

        gam, rv_grid = np.meshgrid(gammas, rvs)
        plt.contourf(gam, rv_grid, dual_chi2)
        plt.title("dual chi2 - gamma {}, rv {}".format(gamma, rv))
        plt.xlabel("gamma")
        plt.ylabel("rv")
        plt.colorbar()
        if independent:
            plt.plot(gamma, rv, "rx")
        else:
            plt.plot(gamma, gamma + rv, "rx")
        plt.show()

        dof = len(fake_data.xaxis) - 1

# In[ ]:


plt.plot(fake_data.flux)
plt.show()


# In[ ]:


def plt_1d_grid(grid):
    assert len(grid.shape) == 2
    for i in range(grid.shape[1]):
        plt.plot(grid[:, i], label=i)
    # plt.legend()
    plt.show()


def plt_2d_grid(grid):
    assert len(grid.shape) == 3

    for i in range(grid.shape[1]):
        for j in range(grid.shape[2]):
            plt.plot(grid[:, i, j], label="{}, {}".format(i, j))
    # plt.legend()
    plt.show()


plt_1d_grid(gamma_grid_data)

plt_1d_grid(rv_grid_data)

plt_2d_grid(dual_grid_data)

# In[ ]:






# In[6]:


# Reduced chi_2
fake_data2 = fake_data.copy()
fake_data2.add_noise(200)

gamma_chi2_2 = chi_squared(fake_data2.flux[:, np.newaxis], gamma_grid_data, 1 / 100)
rv_chi2_2 = chi_squared(fake_data2.flux[:, np.newaxis], rv_grid_data, 1 / 100)
dual_chi2_2 = chi_squared(fake_data2.flux[:, np.newaxis, np.newaxis], dual_grid_data, 1 / 100)

gamma_reduced_chi2 = gamma_chi2_2 / dof
rv_reduced_chi2 = gamma_chi2_2 / dof
gamma_reduced_chi2 = dual_chi2_2 / (dof - 1)

print(np.min(gamma_reduced_chi2))
print(np.min(rv_reduced_chi2))
print(np.min(gamma_reduced_chi2))


# In[ ]:
