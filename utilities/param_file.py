import os
import logging


def parse_paramfile(param_file, path=None):
    """Extract orbit and stellar parameters from parameter file.

    parameters
    ----------
    param_file: str
        Filename of parameter file.
    path: str [optional]
        Path to directory of filename.

    Reuturns
    --------
    """

    if path is not None:
        param_file = os.path.join(path, param_file)
    parameters = dict()
    if not os.path.exists(param_file):
        logging.warning("Parameter file given does not exist. {}".format(param_file))

    with open(param_file, 'r') as f:
        for line in f:
            if line.startswith("#"):
                pass
            else:
                par, val = line.lower().split('=')
                try:
                    # Turn parameters to floats if possible.
                    parameters[par.strip()] = float(val.strip())
                except ValueError:
                    parameters[par.strip()] = val.strip()

    return parameters
