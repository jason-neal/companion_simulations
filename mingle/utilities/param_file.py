"""param_file.py."""
import logging
import os
from logutils import BraceMessage as __

import simulators

from typing import Dict, List, Optional, Tuple, Union


def parse_paramfile(param_file: str, path: Optional[str] = None) -> Dict[
    str, Union[str, float, List[Union[str, float]]]]:
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
    parameters = dict()  # Dict[str, Union[str, float, List[Union[str, float]]]]
    if not os.path.exists(param_file):
        raise Exception("Invalid Arguments, expected a file that exists not. {0}".format(param_file))

    with open(param_file, 'r') as f:
        for line in f:
            if line.startswith("#") or line.isspace() or not line:
                pass
            else:
                if '#' in line:  # Remove comment from end of line
                    line = line.split("#")[0]
                if line.endswith("="):
                    logging.warning(__(("Parameter missing value in {0}.\nLine = {1}."
                                        " Value set to None."), param_file, line))
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


def get_host_params(star: str) -> Tuple[float, float, float]:
    """Find host star parameters from param file."""
    params = load_paramfile(star)
    temp, logg, fe_h = params["temp"], params["logg"], params["fe_h"]
    if isinstance(temp, list):
        temp = temp[0]
    if isinstance(logg, list):
        temp = temp[0]
    if isinstance(fe_h, list):
        temp = temp[0]
    return float(temp), float(logg), float(fe_h)


def load_paramfile(star: str) -> Dict[str, Union[str, float, List[Union[str, float]]]]:
    """Load parameter file with config path."""
    # test assert for now
    param_file = "{0}_params.dat".format(star)

    return parse_paramfile(param_file, simulators.paths["parameters"])


def parse_obslist(fname: str, path: str = None) -> List[str]:
    """Parse Obslist file containing list of dates/times.

    Parameters
    ----------
    fname: str
        Filename of obs_list file.
    path: str [optional]
        Path to directory of filename.

    Returns
    --------
    times: list of strings
        Observation times in a list.
    """
    if path is not None:
        fname = os.path.join(path, fname)
    if not os.path.exists(fname):
        logging.warning(__("Obs_list file given does not exist. {0}", fname))

    obstimes = list()
    with open(fname, 'r') as f:
        for line in f:
            if line.startswith("#") or line.isspace() or not line:  # Ignores comments and blank/empty lines.
                continue
            else:
                if "#" in line:  # Remove comment from end of line
                    line = line.split("#")[0]
                if "." in line:
                    line = line.split(".")[0]  # remove fractions of seconds.
                obstimes.append(line.strip())
        logging.debug(__("obstimes = {}", obstimes))
    return obstimes
