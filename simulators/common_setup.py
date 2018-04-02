import logging
import os
import warnings
from typing import Dict, Tuple, Union, List

from logutils import BraceMessage as __

import simulators
from mingle.utilities import parse_paramfile


def setup_dirs(star: str, mode: str = "iam") -> str:
    mode = mode.lower()
    assert mode in ["iam", "tcm", "bhm"]

    basedir = os.path.join(simulators.paths["output_dir"], star.upper(), mode)
    os.makedirs(basedir, exist_ok=True)
    os.makedirs(os.path.join(basedir, "plots"), exist_ok=True)
    return basedir


def sim_helper_function(star: str, obsnum: Union[int, str], chip: int, skip_params: bool = False, mode: str = "iam") -> Tuple[
    str, Dict[str, Union[str, float, List[Union[str, float]]]], str]:
    """Help simulations by getting parameters, and observation name, and prefix for any output files."""
    mode = mode.lower()
    if mode not in ["iam", "tcm", "bhm"]:
        raise ValueError("Mode {} for sim_helper_function not in 'iam, tcm, bhm'".format(mode))

    if not skip_params:
        param_file = os.path.join(simulators.paths["parameters"], "{}_params.dat".format(star))
        params = parse_paramfile(param_file, path=None)
    else:
        params = {}

    obs_name = os.path.join(
        simulators.paths["spectra"], obs_name_template().format(star, obsnum, chip))

    output_prefix = os.path.join(
        simulators.paths["output_dir"], star.upper(), mode,
        "{0}-{1}_{2}_{3}_chisqr_results".format(star.upper(), obsnum, chip, mode))
    return obs_name, params, output_prefix


def obs_name_template() -> str:
    """Make spectrum name based on config file.

    Valid values:
        [tell_corr, h2o_tell_corr, berv_mask, berv_corr, h2o_berv_corr, h2o_berv_mask]
    """
    spec_version = simulators.spec_version
    valid_keys = ["tell_corr", "h2o_tell_corr", "berv_mask", "berv_corr", "h2o_berv_corr", "h2o_berv_mask"]
    if spec_version is None:
        warnings.warn("No spec_version specified in config.yaml. Defaulting to berv_mask template.")
        spec_version = "berv_mask"

    if spec_version not in valid_keys:
        raise ValueError("spec_version {} is not valid.".format(spec_version))
    if spec_version == "h2o_berv_mask":
        fname = "{0}-{1}-mixavg-h2otellcorr_{2}_bervcorr_masked.fits"
    elif spec_version == "h2o_berv_corr":
        fname = "{0}-{1}-mixavg-h2otellcorr_{2}_bervcorr.fits"
    elif spec_version == "h2o_tell_corr":
        fname = "{0}-{1}-mixavg-h2otellcorr_{2}.fits"
    elif spec_version == "berv_mask":
        fname = "{0}-{1}-mixavg-tellcorr_{2}_bervcorr_masked.fits"
    elif spec_version == "berv_corr":
        fname = "{0}-{1}-mixavg-tellcorr_{2}_bervcorr.fits"
    else:  # spec_version == "tell_corr":
        fname = "{0}-{1}-mixavg-tellcorr_{2}.fits"

    return fname
