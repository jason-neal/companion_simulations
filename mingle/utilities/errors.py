"""Calculate Errors on the Spectrum.

   For a first go using an fixed SNR of 200 for all observations.
"""

import json
import os

import numpy as np
from jsmin import jsmin

import simulators


def get_snrinfo(star, obsnum, chip):
    """Load SNR info from json file."""
    snr_file = os.path.join(simulators.paths["spectra"], "detector_snrs.json")
    try:
        with open(snr_file, "r") as f:
            snr_data = json.loads(jsmin(f.read()))

        return snr_data[str(star)][str(obsnum)][str(chip)]
    except (KeyError, FileNotFoundError) as e:
        print(e)
        print("No snr file/data present for {0}-{1}_{2}. "
              "Setting error to None instead".format(star, obsnum, chip))
        return None


def spectrum_error(star, obsnum, chip, error_off=False):
    """Return the spectrum error.

    if errors is None it will perform a normal chi**2 statistic.
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


def betasigma_error(spectrum, N=5, j=2, returnMAD=True, **kwargs):
    """Calculated std using the BetaSigma technique.

    N=5, j=2 is suitable for the CRIRES High resolution spectra used here.
    Check if this is valid for the spectra you are using it for following
    the guidelines in Czesla et al. 2017.

    Extra beta sigma kwargs passed in.
    Uses the MAD robust estimator.
    """
    from PyAstronomy import pyasl

    # Arbitrary returns segfaults and linear algebra faults.
    # bsarb = pyasl.BSArbSamp()
    # sigma, delta_sigma = bsarb.betaSigma(spectrum.xaxis, spectrum.flux, N, j, returnMAD=returnMAD, **kwargs)
    bseq = pyasl.BSEqSamp()
    sigma, delta_sigma = bseq.betaSigma(spectrum.flux, N, j, returnMAD=returnMAD, **kwargs)

    return sigma, delta_sigma
