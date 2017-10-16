"""Adding wl masking.
"""
import json

def get_maskinfo(star, obs_num, chip):
    with open("/home/jneal/.handy_spectra/detector_masks.json", "r") as f:
        mask_data = json.load(f)
    try:
        this_mask = mask_data[str(star)][str(obs_num)][str(chip)]
        print("This mask", this_mask)
        return this_mask
    except KeyError:
        print("No Masking data present for {0}-{1}_{2}".format(star, obs_num, chip))
        return []
