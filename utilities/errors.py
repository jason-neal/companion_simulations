import os
import simulators
import numpy as np
import json

"""Calculate Errors on the Spectrum.

   For a first go using an fixed SNR of 200 for all observations.
"""

def get_snrinfo(star, obs_num, chip):
    snr_file = os.path.join(simulators.paths["spectra"], "detector_snrs.json")
    with open(snr_file, "r") as f:
        snr_data = json.load(f)
    try:
        return snr_data[str(star)][str(obs_num)][str(chip)]
    except KeyError as e:
        print("No snr data present for {0}-{1}_{2}".format(star, obs_num, chip))
        raise e


def spectrum_error(star, obs_num, chip, error_off=False):
    """Return the specturm error.

    errors = None will perform a normal chi**2 statistic.
    """
    if error_off:
        errors = None
    else:
        snr = get_snrinfo(star, obs_num, chip)
        if len(snr) == 1:
            errors = 1 / np.float(snr[0])
        else:
            raise NotImplementedError("Haven't checked if an error array can be handled yet.")
    return errors
