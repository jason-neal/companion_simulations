import os

import matplotlib
import numpy as np
import yaml

matplotlib.use('Agg')

__version__ = '0.2'
__all__ = ["bhm_module", "bhm_script",
           "iam_module", "iam_script",
           "tcm_script", "tcm_module.py",
           "common_setup", "fake_simulator",]

# Read the users config.yaml file.
# If it doesn't exist, print a useful help message
try:
    f = open("config.yaml")
    config = yaml.load(f)
    f.close()
except FileNotFoundError as e:
    default = os.path.join(__file__[:-11], "..", "config.yaml")
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
spec_version = config.get("spec_version", None)
betasigma = config.get("betasigma", {"N":5, "j":2})

# Check the sim_grid parameters are not empty
for key in sim_grid:
    if "None" in sim_grid[key]:
        pass
    else:
        assert len(np.arange(*sim_grid[key])) > 0, "Config.yaml parameters not correct for {}".format(key)
