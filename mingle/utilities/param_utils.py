import itertools
import logging
import warnings
from typing import Dict, Any, Optional, Tuple, List, Union

import numpy as np
from logutils import BraceMessage as __
from numpy import int64, float64

import simulators
from mingle.utilities import check_inputs

from mingle.utilities.limits import get_phoenix_limits, set_model_limits


def target_params(params: Dict[str, Any], mode: Optional[str] = "iam") -> Tuple[
    List[Union[int, float]], List[Union[int, float]]]:
    """Extract parameters from dict for each target.

    Includes logic for handling missing companion logg/fe_h.
    """
    host_params = [params["temp"], params["logg"], params["fe_h"]]

    # Specify the companion logg and metallicity in the parameter files.
    if params.get("comp_logg", None) is None:
        logging.warning(__("Logg for companion 'comp_logg' is not set for {0}", params.get("name", params)))
    print("mode in target params", mode)
    if mode == "iam":
        comp_logg = params.get("comp_logg", params["logg"])  # Set equal to host if not given
        comp_fe_h = params.get("comp_fe_h", params["fe_h"])  # Set equal to host if not given
        comp_temp = params.get("comp_temp", 999999)  # Will go to largest grid
        comp_params = [comp_temp, comp_logg, comp_fe_h]
    elif mode == "bhm":
        comp_params = []
    else:
        raise ValueError("Mode={} is invalid".format(mode))
    return host_params, comp_params


def closest_obs_params(params, mode: str = "iam"):
    """Return the closest gird values to the params values"""

    host_params, comp_params = target_params(params, mode=mode)

    if mode == "iam":
        closest_host_model = closest_model_params(*host_params)  # unpack temp, logg, fe_h with *
        closest_comp_model = closest_model_params(*comp_params)
        return closest_host_model, closest_comp_model
    elif mode == "bhm":
        closest_host_model = closest_model_params(*host_params)  # unpack temp, logg, fe_h with *
        return closest_host_model
    else:
        raise ValueError("Mode for closest_obs_params is not correct. Needs to be 'iam' or 'bhm'")


def closest_model_params(teff: Union[float, int], logg: Union[float, int], feh: Union[float, int],
                         alpha: Optional[Union[float, int]] = None) -> List[Union[int64, float64]]:
    """Find the closest PHOENIX-ACES model parameters to the stellar parameters given.

    Parameters
    ----------
    teff: float
    logg: float
    feh: float
    alpha: float (optional)

    Returns
    -------
    params: list of floats
        Parameters for the closest matching model.

    """
    teffs = np.concatenate((np.arange(2300, 7000, 100),
                            np.arange(7000, 12100, 200)))
    loggs = np.arange(0, 6.1, 0.5)
    fehs = np.concatenate((np.arange(-4, -2, 1), np.arange(-2, 1.1, 0.5)))
    alphas = np.arange(-0.2, 0.3, 0.2)  # use only these alpha values if necessary

    closest_teff = teffs[np.abs(teffs - teff).argmin()]
    closest_logg = loggs[np.abs(loggs - logg).argmin()]
    closest_feh = fehs[np.abs(fehs - feh).argmin()]

    if alpha is not None:
        if abs(float(alpha)) > 0.2:
            logging.warning("Alpha is outside acceptable range -0.2->0.2")
        closest_alpha = alphas[np.abs(alphas - alpha).argmin()]

        return [closest_teff, closest_logg, closest_feh, closest_alpha]
    else:
        return [closest_teff, closest_logg, closest_feh]


def get_bhm_model_pars(params: Dict[str, Union[int, float]], method: str = "close") -> List[
    List[Union[int64, float64]]]:
    method = method.lower()
    closest_host_model = closest_obs_params(params, mode="bhm")
    if method == "config":
        model_pars = list(generate_bhm_config_params(closest_host_model))
    elif method == "close":
        # Model parameters to try iterate over.
        model_pars = list(generate_close_params(closest_host_model, small=True))
    else:
        raise ValueError("The method '{0}' is not valid".format(method))

    return model_pars


def generate_bhm_config_params(params, limits="phoenix"):
    """Generate teff, logg, Z values given star params and config values.

    Version of "generate_close_params_with_simulator" for bhm.
    """

    temp, logg, metals = params[0], params[1], params[2]
    # This is the backup if not specified in config file.
    bk_temps, bk_loggs, bk_metals = gen_new_param_values(temp, logg, metals, small=True)

    teff_values = simulators.sim_grid.get("teff_1")
    logg_values = simulators.sim_grid.get("logg_1")
    feh_values = simulators.sim_grid.get("feh_1")
    new_temps = make_grid_parameter(temp, teff_values, bk_temps)
    new_loggs = make_grid_parameter(logg, logg_values, bk_loggs)
    new_metals = make_grid_parameter(metals, feh_values, bk_metals)

    phoenix_limits = get_phoenix_limits(limits)

    new_temps, new_loggs, new_metals = set_model_limits(new_temps, new_loggs, new_metals, phoenix_limits)

    new_temps, new_loggs, new_metals = set_model_limits(new_temps, new_loggs, new_metals,
                                                        simulators.starfish_grid["parrange"])

    new_temps = check_inputs(new_temps)
    new_loggs = check_inputs(new_loggs)
    new_metals = check_inputs(new_metals)

    for t, l, m in itertools.product(new_temps, new_loggs, new_metals):
        yield [t, l, m]


def generate_close_params(params, small=True, limits="phoenix"):
    """teff, logg, Z.

    "small" is a mode selector basically.
    """
    temp, logg, metals = params[0], params[1], params[2]

    new_temps, new_loggs, new_metals = gen_new_param_values(temp, logg, metals, small=small)

    phoenix_limits = get_phoenix_limits(limits)

    new_temps, new_loggs, new_metals = set_model_limits(new_temps, new_loggs, new_metals, phoenix_limits)

    for t, l, m in itertools.product(new_temps, new_loggs, new_metals):
        yield [t, l, m]


def generate_close_params_with_simulator(params, target, limits="phoenix"):
    """teff, logg, Z.

    "Target" is required to make sure this is used correctly..."""
    if target not in ["host", "companion"]:
        raise ValueError("Target must be 'host' or 'companion', not '{}'".format(target))

    temp, logg, metals = params[0], params[1], params[2]
    # This is the backup if not specified in config file.
    bk_temps, bk_loggs, bk_metals = gen_new_param_values(temp, logg, metals, small=target)
    # print("params", params, target, small, limits)

    teff_key = "teff_1" if target == "host" else "teff_2"
    logg_key = "logg_1" if target == "host" else "logg_2"
    feh_key = "feh_1" if target == "host" else "feh_2"

    teff_values = simulators.sim_grid.get(teff_key)
    logg_values = simulators.sim_grid.get(logg_key)
    feh_values = simulators.sim_grid.get(feh_key)

    new_temps = make_grid_parameter(temp, teff_values, bk_temps)
    new_loggs = make_grid_parameter(logg, logg_values, bk_loggs)
    new_metals = make_grid_parameter(metals, feh_values, bk_metals)

    phoenix_limits = get_phoenix_limits(limits)

    new_temps, new_loggs, new_metals = set_model_limits(new_temps, new_loggs, new_metals, phoenix_limits)

    dim = len(new_temps) * len(new_loggs) * len(new_metals)
    new_temps, new_loggs, new_metals = set_model_limits(new_temps, new_loggs, new_metals,
                                                        simulators.starfish_grid["parrange"])
    dim_2 = len(new_temps) * len(new_loggs) * len(new_metals)
    if dim_2 < dim:
        # Warning in-case you do not remember about parrange limits.
        logging.warning("Some models were cut out using the 'parrange' limits.")

    new_temps = check_inputs(new_temps)
    new_loggs = check_inputs(new_loggs)
    new_metals = check_inputs(new_metals)

    for t, l, m in itertools.product(new_temps, new_loggs, new_metals):
        yield [t, l, m]


def gen_new_param_values(temp, logg, metals, small=True):
    if small == "host":
        # only include error bounds.
        new_temps = np.array([-100, 0, 100]) + temp
        new_metals = np.array([-0.5, 0.0, 0.5]) + metals
        new_loggs = np.array([-0.5, 0.0, 0.5]) + logg
    elif small:
        new_temps = np.arange(-600, 601, 100) + temp
        new_metals = np.array([-0.5, 0.5, 0.5]) + metals
        new_loggs = np.array([-0.5, 0.5, 0.5]) + logg
    else:
        new_temps = np.arange(-500, 501, 100) + temp
        new_metals = np.arange(-1, 1.1, 0.5) + metals
        new_loggs = np.arange(-1, 1.1, 0.5) + logg
    return new_temps, new_loggs, new_metals


def make_grid_parameter(param, step_config, backup):
    """Extend parameter grid about param. Using step_config=[start, stop, step].

    param:
        Value of the parameter to increment from.
    step_config:
        [Start, stop, step] or can be None.
    backup:
        Pre-calculated values if the step_config is not given in config.yaml.
        """
    if step_config is None or step_config == "None":
        return backup
    else:
        values = np.arange(*step_config)
        if len(values) == 1 and values[0] != 0:
            print("The configured parameter range is {}".format(values))
            raise ValueError("Invalid parameter configuration. No single model grid with offset !=0 allowed!")
        else:
            if 0 not in values:
                warnings.warn("The grids do not span the closest parameters. Values={}. Check config".format(values))
            return param + values