import json
import os
import warnings

import numpy as np

import simulators

"""Calculate Errors on the Spectrum.

   For a first go using an fixed SNR of 200 for all observations.
"""


def get_snrinfo(star, obsnum, chip):
    """Load SNR info from json file."""
    snr_file = os.path.join(simulators.paths["spectra"], "detector_snrs.json")
    with open(snr_file, "r") as f:
        snr_data = json.load(f)
    try:
        return snr_data[str(star)][str(obsnum)][str(chip)]
    except KeyError as e:
        warnings.warn("No snr data present for {0}-{1}_{2}. "
                      "Setting error to None instead".format(star, obsnum, chip))
        return None


def spectrum_error(star, obsnum, chip, error_off=False):
    """Return the spectrum error.

    errors = None will perform a normal chi**2 statistic.
    """
    if error_off:
        errors = None
    else:
        snr = get_snrinfo(star, obsnum, chip)
        if snr is None:
            errors = None
        elif len(snr) == 1:
            errors = 1 / np.float(snr[0])
        else:
            raise NotImplementedError("Haven't checked if an error array can be handled yet.")
    return errors
