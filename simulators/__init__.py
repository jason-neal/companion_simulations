__version__ = '0.2'
__all__ = ["best_host_model_old", "bhm_HD211847", "bhm_module", "bhm_script",
           "Chisqr_of_HD211847_starfish", "Chisqr_of_observation",
           "Chisqr_of_observation_HD211847", "iam_module", "iam_script",
           "Planet_spectral_simulations", "tcm_HD211847", "tcm_HD211847_test_",
           "tcm_module.py"]

# Read the users config.yaml file.
# If it doesn't exist, print a useful help message

import yaml
import matplotlib
matplotlib.use('Agg')

try:
    f = open("config.yaml")
    config = yaml.load(f)
    f.close()
except FileNotFoundError as e:
    default = __file__[:-11] + "config.yaml"
    import warnings
    warnings.warn("Using the default config.yaml file located at {0}."
                  "This is likely NOT what you want. "
                  "Please create a similar 'config.yaml' file in your "
                  "current working directory.".format(default), UserWarning)
    f = open(default)
    config = yaml.load(f)
    f.close()


# Read the YAML variables into package-level dictionaries to be used by the other programs.
sim_grid = config["sim_grid"]
paths = config["paths"]
name = config["name"]
starfish_grid = config["grid"]
# use as gammas = simulators.sim_grid["gammas"]
data = config["data"]
outdir = config["outdir"]
plotdir = config["plotdir"]
