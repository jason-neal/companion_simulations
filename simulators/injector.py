import argparse
from argparse import Namespace
import numpy as np
from typing import List, Union, Any
import sys, os
from mingle.utilities.debug_utils import timeit
from lmfit import Parameters
from spectrum_overload import Spectrum
from simulators.iam_module import observation_rv_limits
from mingle.utilities.phoenix_utils import load_starfish_spectrum
from simulators.minimize_iam import brute_solve_iam
from simulators.common_setup import load_observation_with_errors
from mingle.utilities.param_utils import closest_obs_params
from mingle.models.broadcasted_models import inherent_alpha_model


def parse_args(args: List[str]) -> Namespace:
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Injection script.')
    parser.add_argument("-s", "--star", help='Star name.', type=str, default="HD211847")
    parser.add_argument("-o", "--obsnum", help='Star observation number.', type=str, default="2")
    # parser.add_argument("-n", "--re_normalize", help="Scalar re-normalize flux to models. Default=False",
    #                     action="store_true")
    # parser.add_argument("-m", "--norm_method", help="Re-normalization method flux to models. Default=scalar",
    #                     choices=["scalar", "linear"], default="scalar")
    # parser.add_argument("--error_off", help="Turn snr value errors off.",
    #                     action="store_true")
    # parser.add_argument('-a', '--area_scale', action="store_false",
    #                     help='Scaling by stellar area. (raise to disable)')
    # parser.add_argument('--disable_wav_scale', action="store_true",
    #                     help='Disable scaling by wavelength.')
    # parser.add_argument('--suffix', help='Suffix for file.', type=str)

    parser.add_argument('-v', '--verbose', action="store_true",
                        help='Turn on Verbose.')
    return parser.parse_args(args)


def injector_wrapper(star, obsnum, chip, Ns=20):
    try:
        iter(chip)
    except:
        # Make iterable
        chip = [chip]

    spec_list, error_list = [], []
    for c in chip:
        obs_spec, errors, obs_params = load_observation_with_errors(star, obsnum, c)
        spec_list.append(obs_spec)
        error_list.append(errors)
    obs_spec, errors = spec_list, error_list
    print("len(obs_spec)", len(obs_spec), "len(chip)", len(chip))
    assert len(obs_spec) == len(chip)

    closest_host_model, closest_comp_model = closest_obs_params(obs_params, mode="iam")

    # setup parameters
    params = Parameters()
    params.add('teff_1', value=closest_host_model[0], min=5600, max=5800, vary=False, brute_step=100)
    params.add('logg_1', value=closest_host_model[1], min=0, max=6, vary=False, brute_step=0.5)
    params.add('logg_2', value=closest_comp_model[1], min=0, max=6, vary=False, brute_step=0.5)
    params.add('feh_1', value=closest_host_model[2], min=-2, max=1, vary=False, brute_step=0.5)
    params.add('feh_2', value=closest_comp_model[2], min=-2, max=1, vary=False, brute_step=0.5)
    params.add('rv_1', value=7, min=-20, max=20, vary=True, brute_step=0)
    params.add('rv_2', value=1.5, min=-10, max=10, vary=True, brute_step=0)


    rv_limits = [observation_rv_limits(obs, params["rv_1"].value,
                                       params["rv_2"].value) for obs in obs_spec]

    mod1_spec = [load_starfish_spectrum(closest_host_model, limits=lim,
                                           hdr=True, normalize=False, area_scale=True,
                                           flux_rescale=True, wav_scale=True) for lim in rv_limits]

    def inject(teff_2):
        params.add('teff_2', value=teff_2, min=teff_2 - 1000, max=teff_2 + 1000, vary=True, brute_step=100)

        # Add companion to observation
        print("Have not added the companion here")

        # Load in the models
        injected_spec = []
        for ii, c in enumerate(chip):
            mod2_spec = load_starfish_spectrum([teff_2, params["logg_2"].value, params["feh_2"].value],
                                               limits=rv_limits[ii], hdr=True, normalize=False,
                                               area_scale=True, flux_rescale=True, wav_scale=True)

            iam_grid_func = inherent_alpha_model(mod1_spec[ii].xaxis, mod1_spec[ii].flux, mod2_spec.flux,
                                             rvs=params["rv_2"].value, gammas=params["rv_1"].value)
            synthetic_model = iam_grid_func(obs_spec[ii].xaxis)
            continuum = Spectrum(xaxis=obs_spec[ii].xaxis, flux=synthetic_model).continuum()

            mod2_spec_norm = mod2_spec / continuum

            # This is doing the injection
            # This currently does not renomalize the host due to the addition.
            injected_chip = obs_spec + mod2_spec_norm
            injected_spec.append(injected_chip)

        return brute_solve_iam(params, injected_spec, errors, chip, Ns=Ns)

    return inject


@timeit
def main(star, obsnum, **kwargs):
    """Main function."""
    chip = [1, 2, 3]
    injector = injector_wrapper(star, obsnum, chip, Ns=20)

    injection_temps = np.arange(2300, 5801, 100)

    first = injection_temps[0]
    first_injector_result = injector(first)
    last = injection_temps[1]
    last_injector_result = injector(last)
    assert is_recovered(last, first_injector_result)
    assert not is_recovered(first, last_injector_result)

    # Try binary search

    while len(injection_temps > 2):
        middle_value = int(injection_temps[np.floor(len(injection_temps) / 2)])
        print(middle_value)
        injector_result = injector(middle_value)
        if is_recovered(middle_value, injector_result):
            # remove upper values to search lower half for cut off
            injected_temps = injection_temps[injection_temps >= middle_value]
            first_injector_result = injector_result
        else:
            # Remove upper half as the middle value is the new lower limit
            injection_temps = injection_temps[injection_temps <= middle_value]
            last_injector_result = injector_result

    print("Boundary of recovery", injection_temps)
    print(first_injector_result.pretty_print())
    print(last_injector_result.pretty_print())


def is_recovered(injected_value: Union[float, int], injector_result: Any, dof: int = 1) -> bool:
    """Is the recovered chi^2 min value within 1-sigma for injected_value."""

    one_sigma = 1
    indicies = np.nonzero(recovered_chi2 < min(recovered_chi2) + one_sigma)
    values_inside_one_sigma = chi2_value[indicies]

    return injected_value in values_inside_one_sigma


if __name__ == "__main__":
    args = vars(parse_args(sys.argv[1:]))
    opts = {k: args[k] for k in args}
    main(**opts)
