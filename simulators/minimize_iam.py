import logging

from logutils import BraceMessage as __

import simulators
from mingle.utilities.crires_utilities import barycorr_crires_spectrum
from mingle.utilities.errors import betasigma_error
from mingle.utilities.masking import spectrum_masking
from mingle.utilities.phoenix_utils import (closest_model_params)
from mingle.utilities.spectrum_utils import load_spectrum
from simulators.iam_module import (iam_helper_function,
                                   setup_iam_dirs, target_params)

logging.basicConfig(level=logging.WARNING,
                    format='%(levelname)s %(message)s')


def main(star, obsnum, chip):
    star = star.upper()
    setup_iam_dirs(star)

    # Setup comparision spectra
    obs_name, params, output_prefix = iam_helper_function(star, obsnum, chip)
    # if suffix is not None:
    #    output_prefix = output_prefix + str(suffix)

    print("The observation used is ", obs_name, "\n")

    host_params, comp_params = target_params(params, mode="iam")

    closest_host_model = closest_model_params(*host_params)
    closest_comp_model = closest_model_params(*comp_params)

    # Load observation
    obs_spec = load_spectrum(obs_name)
    # Mask out bad portion of observed spectra
    obs_spec = spectrum_masking(obs_spec, star, obsnum, chip)
    # Barycentric correct spectrum
    _obs_spec = barycorr_crires_spectrum(obs_spec, extra_offset=None)

    # Determine Spectrum Errors
    N = simulators.betasigma.get("N", 5)
    j = simulators.betasigma.get("j", 2)
    errors, derrors = betasigma_error(obs_spec, N=N, j=j)
    print("Beta-Sigma error value = {:6.5f}+/-{:6.5f}".format(errors, derrors))

    params = Parameters()
    params.add()
    params.add('teff_1', value=closest_host_model[0], min=2300, max=7000, vary=True, brute_step=100)
    params.add('teff_2', value=closest_comp_model[0], min=2300, max=7000, vary=True, brute_step=100)
    params.add('logg_1', value=closest_host_model[1], min=-0, max=6, vary=False, brute_step=0.5)
    params.add('logg_2', value=closest_comp_model[1], min=-0, max=6, vary=False, brute_step=0.5)
    params.add('feh_1', value=closest_host_model[2], min=-2, max=1, vary=False, brute_step=0.5)
    params.add('feh_2', value=closest_comp_model[2], min=-2, max=1, vary=False, brute_step=0.5)
    params.add('rv_1', value=0, min=-50, max=50, vary=True)
    params.add('rv_2', value=0, min=-50, max=50, vary=True)

    minimize(iam_chi2, params)


if __name__ == "__main__":
    star = "HD211847"
    obsnum = 1
    chip = 1
    main(star, obsnum, chip)

    print("Done")
