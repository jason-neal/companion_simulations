import matplotlib.pyplot as plt

from obsolete.models.alpha_model import alpha_model2
from obsolete.models.alpha_model import no_alpha
from mingle.utilities.phoenix_utils import load_starfish_spectrum

"""Try out the models and how they look.
"""

# Parameters of models
# host_params = [6000, 4.5, 0]
# companion_params = [3100, 4.5, 0.5]
host_params = [5000, 4.5, 0]
companion_params = [2300, 4.5, 0.5]

host_unnorm = load_starfish_spectrum(host_params)
comp_unnorm = load_starfish_spectrum(companion_params)

plt.plot(host_unnorm.xaxis, host_unnorm.flux, label="host un-norm")
plt.plot(comp_unnorm.xaxis, comp_unnorm.flux, label="companion un-norm")
plt.legend()
plt.title("Unormalized spectra")
plt.show()

host_norm = load_starfish_spectrum(host_params, normalize=True)
comp_norm = load_starfish_spectrum(companion_params, normalize=True)

plt.plot(host_norm.xaxis, host_norm.flux, label="host norm")
plt.plot(comp_norm.xaxis, comp_norm.flux, label="companion norm")
plt.legend()
plt.title("Normalized spectra")
plt.show()

limits = [2100, 2170]

no_alpha_spec = no_alpha(100, host_unnorm, comp_unnorm, limits=limits, normalize=True)

# alpha_join_50 = alpha_model2(0.5, 100, host_norm, comp_norm, limits=limits)
# alpha_join_30 = alpha_model2(0.3, 100, host_norm, comp_norm, limits=limits)
# alpha_join_20 = alpha_model2(0.2, 100, host_norm, comp_norm, limits=limits)
alpha_join_10 = alpha_model2(0.1, 100, host_norm, comp_norm, limits=limits)
alpha_join_5 = alpha_model2(0.05, 100, host_norm, comp_norm, limits=limits)
alpha_join_0 = alpha_model2(0.00, 100, host_norm, comp_norm, limits=limits)

plt.plot(no_alpha_spec.xaxis, no_alpha_spec.flux, label="NO-ALPHA")
plt.plot(alpha_join_0.xaxis, alpha_join_0.flux, label="Alpha 0%")
plt.plot(alpha_join_5.xaxis, alpha_join_5.flux, label="Alpha 5%")
plt.plot(alpha_join_10.xaxis, alpha_join_10.flux, label="Alpha 10%")
# plt.plot(alpha_join_20.xaxis, alpha_join_20.flux, label="Alpha 20%")
# plt.plot(alpha_join_30.xaxis, alpha_join_30.flux, label="Alpha 30%")
# plt.plot(alpha_join_50.xaxis, alpha_join_50.flux, label="Alpha 50%")
plt.legend()
plt.title("Combined")
plt.show()
