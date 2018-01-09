"""Adding wl masking.
"""
import json
import os

from jsmin import jsmin

import simulators


def get_maskinfo(star, obsnum, chip):
    mask_file = os.path.join(simulators.paths["spectra"], "detector_masks.json")
    with open(mask_file, "r") as f:
        mask_data = json.loads(jsmin(f.read()))
    try:
        this_mask = mask_data[str(star)][str(obsnum)][str(chip)]
        return this_mask
    except KeyError:
        print("No Masking data present for {0}-{1}_{2}".format(star, obsnum, chip))
        return []


def spectrum_masking(spec, star, obsnum, chip):
    chip_masks = get_maskinfo(star, obsnum, chip)
    if int(chip) == 4:
        # Ignore first 50 pixels of detector 4
        dw = 0.0000001  # small offset to mask inclusive
        spec.wav_select(spec.xaxis[50] - dw, spec.xaxis[-1] + dw)
    for mask_limits in chip_masks:
        # If empty they do not go in here
        if len(mask_limits) is not 2:
            raise ValueError("Mask limits in mask file are incorrect for {0}-{1}_{2}".format(star, obsnum, chip))
        spec.wav_select(*mask_limits)  # Wavelengths to include
    return spec
