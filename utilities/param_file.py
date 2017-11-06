"""param_file.py."""
import logging
import os
from typing import Dict, List, Union

import simulators


def parse_paramfile(param_file: str, path: str = None) -> Dict[str, Union[str, float, List[float]]]:
    """Extract orbit and stellar parameters from parameter file.

    Parameters
    ----------
    param_file: str
        Filename of parameter file.
    path: str [optional]
        Path to directory of filename.

    Returns
    --------
    parameters: dict
        Parameters as a {param: value} dictionary.
    """
    if path is not None:
        param_file = os.path.join(path, param_file)
    parameters = dict()
    if not os.path.exists(param_file):
        raise Exception("Invalid Arguments, expected a file that exists not. {0}".format(param_file))

    with open(param_file, 'r') as f:
        for line in f:
            if line.startswith("#"):
                pass
            else:
                if '#' in line:  # Remove comment from end of line
                    line = line.split("#")[0]
                if line.endswith("="):
                    logging.warning(("Parameter missing value in {0}.\nLine = {1}."
                                     " Value set to None.").format(param_file, line))
                    line = line + " None"  # Add None value when parameter is missing
                par, val = line.lower().split('=')
                par, val = par.strip(), val.strip()
                if (val.startswith("[") and val.endswith("]")) or ("," in val):  # Val is a list
                    parameters[par] = parse_list_string(val)
                else:
                    try:
                        parameters[par] = float(val)  # Turn parameters to floats if possible.
                    except ValueError:
                        parameters[par] = val

    return parameters


def parse_list_string(string: str) -> List[Union[str, float]]:
    """Parse list of floats out of a string."""
    string = string.replace("[", "").replace("]", "").strip()
    list_str = string.split(",")
    try:
        return [float(val) for val in list_str]
    except ValueError as e:
        # Can't turn into floats.
        return [val.strip() for val in list_str]


def get_host_params(star):
    """Find host star parameters from param file."""
    params = load_paramfile(star)
    return params["temp"], params["logg"], params["fe_h"]


def load_paramfile(star):
    """Load parameter file with config path."""
    # test assert for now
    param_file = "{0}_params.dat".format(star)

    return parse_paramfile(param_file, simulators.paths["parameters"])
