"""Adding wl masking.
"""
import json
import os

from jsmin import jsmin

import simulators
from spectrum_overload import Spectrum


def get_maskinfo(star, obsnum, chip):
    mask_file = os.path.join(simulators.paths["spectra"], "detector_masks.json")
    try:
        with open(mask_file, "r") as f:
            mask_data = json.loads(jsmin(f.read()))

        this_mask = mask_data[str(star)][str(obsnum)][str(chip)]
        return this_mask
    except (KeyError, FileNotFoundError) as e:
        print(e)
        print("No Masking file/data present for {0}-{1}_{2}".format(star, obsnum, chip))
        return []


def spectrum_masking(spec, star, obsnum, chip, stricter=False):
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

    if stricter:
        # Add manual masks to reduce mismatch
        new_spec = stricter_spectrum_masking(spec)

    return new_spec


# Nanometers Masks to HD211847
strict_masks = [(2114, 2115),
         (2127.5, 2128.8),
         (2132.4, 2132.8),
         (2137.8, 2138.4),
         (2154, 2166),
         (2119.1, 2119.6),
         (2122.9, 2123.2),
         (2117.8, 2118.2),
         (2120.3, 2120.5),
         (2147, 2153),
         (2118.5, 2118.8)
         ]

def stricter_spectrum_masking(spec):
    """Apply rigorious cuts in wavelength where largest mismatch occurs.

    This should change the sigma contours.
    """
    xaxis = spec.xaxis
    flux = spec.flux
    for mask in strict_masks:
        mask_bool = [(xaxis > mask[0]) & (xaxis < mask[-1])]
        xaxis = xaxis[mask_bool]
        flux = flux[mask_bool]
        # Slicing of a Spectrum should take out having to handle the other kwargs here.
    return Spectrum(xaxis=xaxis, flux=flux, header=spec.header, calibrated=spec.calibrated, interp_method=interp_method)