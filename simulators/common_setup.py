import logging
import os
import warnings
from typing import Dict, Tuple, Union, List

import simulators
from joblib import Memory
from logutils import BraceMessage as __
from mingle.utilities import parse_paramfile, load_spectrum, spectrum_masking, barycorr_crires_spectrum, betasigma_error

joblib_dir = os.path.join(os.path.expanduser("~"), ".tmp", "joblib")

os.makedirs(joblib_dir, exist_ok=True)
memory = Memory(cachedir=joblib_dir, verbose=0)


def setup_dirs(star: str, mode: str = "iam") -> str:
    mode = mode.lower()
    assert mode in ["iam", "tcm", "bhm"]

    basedir = os.path.join(simulators.paths["output_dir"], star.upper(), mode)
    os.makedirs(basedir, exist_ok=True)
    os.makedirs(os.path.join(basedir, "plots"), exist_ok=True)
    return basedir


def sim_helper_function(star: str, obsnum: Union[int, str], chip: int, skip_params: bool, mode: str = "iam") -> Tuple[
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

    assert spec_version in valid_keys, "spec_versions {} is not valid.".format(spec_version
                                                                               )
    if spec_version == "berv_mask":
        fname = "{0}-{1}-mixavg-tellcorr_{2}_bervcorr_masked.fits"

    elif spec_version == "h2o_berv_mask":
        fname = "{0}-{1}-mixavg-h2otellcorr_{2}_bervcorr_masked.fits"

    elif spec_version == "berv_corr":
        fname = "{0}-{1}-mixavg-tellcorr_{2}_bervcorr.fits"

    elif spec_version == "h2o_berv_corr":
        fname = "{0}-{1}-mixavg-h2otellcorr_{2}_bervcorr.fits"

    elif spec_version == "tell_corr":
        fname = "{0}-{1}-mixavg-tellcorr_{2}.fits"

    elif spec_version == "h2o_tell_corr":
        fname = "{0}-{1}-mixavg-h2otellcorr_{2}.fits"

    else:
        raise ValueError("")
    logging.debug(__("Filename template from obs_name_template = '{0}'", fname))

    return fname


@memory.cache
def load_observation_with_errors(star, obsnum, chip, mode="iam", strict_mask=False, verbose=False, **kwargs):
    obs_name, params, output_prefix = sim_helper_function(star, obsnum, chip, skip_params=False, mode=mode)
    if verbose:
        print("The observation used is ", obs_name, "\n")
    assert not isinstance(chip, list)
    # Load observation
    obs_spec = load_spectrum(obs_name)
    # Mask out bad portion of observed spectra
    obs_spec = spectrum_masking(obs_spec, star, obsnum, chip, stricter=strict_mask)
    # Barycentric correct spectrum
    #_obs_spec = barycorr_crires_spectrum(obs_spec, extra_offset=None)

    # Determine Spectrum Errors
    N = simulators.betasigma.get("N", 5)
    j = simulators.betasigma.get("j", 2)
    errors, derrors = betasigma_error(obs_spec, N=N, j=j)
    if verbose:
        print("Beta-Sigma error value = {:6.5f}+/-{:6.5f}".format(errors, derrors))
    return obs_spec, errors, params
