#!/usr/bin/env python
import argparse
import sys
from argparse import Namespace
from typing import List, Union, Any
import warnings
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameters, fit_report
from mingle.models.broadcasted_models import inherent_alpha_model
from mingle.utilities.debug_utils import timeit
from mingle.utilities.param_utils import closest_obs_params
from mingle.utilities.phoenix_utils import load_starfish_spectrum
from simulators.common_setup import load_observation_with_errors
from simulators.iam_module import iam_magic_sauce
from simulators.iam_module import observation_rv_limits
from simulators.minimize_iam import brute_solve_iam
from spectrum_overload import Spectrum

error_fudge = 1
binary_search = False


def parse_args(args: List[str]) -> Namespace:
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Injection script.')
    parser.add_argument("-s", "--star", help='Star name.', type=str, default="HD211847")
    parser.add_argument("-o", "--obsnum", help='Star observation number.', type=str, default="2")
    parser.add_argument("-m", "--strict_mask", help="Use strict masking", action="store_true")
    parser.add_argument('-v', '--verbose', action="store_true",
                        help='Turn on Verbose.')
    parser.add_argument('-p', '--plot', action="store_true",
                        help='Add plots of each injection.')
    parser.add_argument('-b', '--binary', action="store_true",
                        help='Perform Binary search.')
    parser.add_argument("-g", '--grid_bound', action="store_true",
                        help='Grid bound search limit')

    return parser.parse_args(args)


def injector_wrapper(star, obsnum, chip, Ns=20, strict_mask=False, comp_logg=None, plot=False):
    try:
        iter(chip)
    except:
        # Make iterable
        chip = [chip]

    spec_list, error_list = [], []
    for c in chip:
        obs_spec, errors, obs_params = load_observation_with_errors(star, obsnum, c, strict_mask=strict_mask)
        spec_list.append(obs_spec)
        error_list.append(errors * error_fudge)
    obs_spec, errors = spec_list, error_list
    print("len(obs_spec)", len(obs_spec), "len(chip)", len(chip))
    assert len(obs_spec) == len(chip)

    # Linearlly normalize observation.
    obs_spec = [obs.normalize(method="linear") for obs in obs_spec]
    closest_host_model, closest_comp_model = closest_obs_params(obs_params, mode="iam")

    rv_1 = 7
    rv_2 = 40
    deltarv_1 = 3
    deltarv_2 = 3
    rv1_step = 0.5
    rv2_step = 0.5

    # Setup Fixed injection grid parameters
    params = Parameters()
    params.add('teff_1', value=closest_host_model[0], min=5000, max=6000, vary=False, brute_step=100)
    params.add('logg_1', value=closest_host_model[1], min=0, max=6, vary=False, brute_step=0.5)
    params.add('feh_1', value=closest_host_model[2], min=-2, max=1, vary=False, brute_step=0.5)
    params.add('feh_2', value=closest_comp_model[2], min=-2, max=1, vary=False, brute_step=0.5)
    params.add('rv_1', value=rv_1, min=rv_1 - deltarv_1, max=rv_1 + deltarv_1, vary=True, brute_step=rv1_step)
    params.add('rv_2', value=rv_2, min=rv_2 - deltarv_2, max=rv_2 + deltarv_2, vary=True, brute_step=rv2_step)
    if comp_logg is None:
        params.add('logg_2', value=closest_comp_model[1], min=0, max=6, vary=False, brute_step=0.5)
    else:
        params.add('logg_2', value=comp_logg, min=0, max=6, vary=False, brute_step=0.5)

    rv_limits = [observation_rv_limits(obs, params["rv_1"].value,
                                       params["rv_2"].value) for obs in obs_spec]

    mod1_spec = [load_starfish_spectrum(closest_host_model, limits=lim,
                                        hdr=True, normalize=False, area_scale=True,
                                        flux_rescale=True, wav_scale=True) for lim in rv_limits]

    def inject(teff_2):
        """Incjector function that just takes a temperature."""
        params.add('teff_2', value=teff_2, min=max([teff_2 - 800, 2300]), max=min([teff_2 + 801, 7001]), vary=True,
                   brute_step=100)
        if plot:
            plt.figure()
        # Add companion to observation
        # print("Inital params")
        # params.pretty_print()
        # Load in the models
        injected_spec = []
        print("Injected Teff value", teff_2)

        for ii, c in enumerate(chip):
            if plot:
                plt.subplot(len(chip), 1, ii + 1)
            mod2_spec = load_starfish_spectrum([teff_2, params["logg_2"].value, params["feh_2"].value],
                                               limits=rv_limits[ii], hdr=True, normalize=False,
                                               area_scale=True, flux_rescale=True, wav_scale=True)

            iam_grid_func = inherent_alpha_model(mod1_spec[ii].xaxis, mod1_spec[ii].flux, mod2_spec.flux,
                                                 rvs=params["rv_2"].value, gammas=params["rv_1"].value)
            synthetic_model = iam_grid_func(obs_spec[ii].xaxis)
            continuum = Spectrum(xaxis=obs_spec[ii].xaxis, flux=synthetic_model).continuum(method="exponential")

            # Doppler shift companion
            injection = mod2_spec.copy()
            injection.doppler_shift(params["rv_2"].value + params["rv_1"].value)

            # Normalize by synthetic continuum
            injection.spline_interpolate_to(continuum)
            injection = injection / continuum

            # Inject the companion
            injected_chip = obs_spec[ii] + injection
            shifted_injection = injection + 1.01 - np.mean(injection.flux)

            # Renormalzie
            injected_chip = injected_chip.normalize(method="linear")
            assert not np.any(np.isnan(injection.flux))

            injected_spec.append(injected_chip)
            if plot:
                obs_spec[ii].plot(label="Observation")
                injected_chip.plot(label="Injected_chip", lw=1, linestyle="--")
                shifted_injection.plot(label="injected part", lw=1)
        if plot:
            plt.suptitle("Host= {0}, Injected Temperature = {1}".format(closest_host_model[0], teff_2))
            plt.show(block=False)
        return brute_solve_iam(params, injected_spec, errors, chip, Ns=Ns)

    print("injector", inject)

    return inject


@timeit
def main(star, obsnum, **kwargs):
    """Main function."""
    chip = [1, 2, 3]
    loop_injection_temp = []
    loop_recovered_temp = []
    print("before injector")
    strict_mask = kwargs.get("strict_mask", False)
    grid_recovered = kwargs.get("grid_bound", False)
    comp_logg = kwargs.get("comp_logg", None)
    plot = kwargs.get("plot", False)
    injector = injector_wrapper(star, obsnum, chip, Ns=20, strict_mask=strict_mask, comp_logg=comp_logg, plot=plot)

    injection_temps = np.arange(2300, 5001, 100)
    binary_search = kwargs.get("binary", False)
    if binary_search:
        print("before first")
        first = injection_temps[0]
        first_injector_result = injector(first)
        loop_injection_temp.append(first)
        loop_recovered_temp.append(first_injector_result.params["teff_2"].value)
        # show_brute_solution(first_injector_result, star, obsnum, chip, strict_mask=strict_mask)
        print("done first")
        last = injection_temps[-1]
        last_injector_result = injector(last)
        loop_injection_temp.append(last)
        loop_recovered_temp.append(last_injector_result.params["teff_2"].value)
        # show_brute_solution(last_injector_result, star, obsnum, chip, strict_mask=strict_mask)
        print("Done last")
        assert not is_recovered(first, first_injector_result)
        # assert is_recovered(last, last_injector_result)

        # Try binary search
        print("Starting Binary search")
        while len(injection_temps) > 2:
            print("in while loop")
            print("temp left = ", injection_temps)
            middle_value = int(injection_temps[int(np.floor(len(injection_temps) / 2))])
            print("doing middle value = ", middle_value)
            injector_result = injector(middle_value)
            recovered = is_recovered(middle_value, injector_result, grid_recovered)
            if recovered:
                # remove upper values to search lower half for cut off
                injection_temps = injection_temps[injection_temps <= middle_value]
                last_injector_result = injector_result
            else:
                # Remove lower half as the middle value is the new lower limit
                injection_temps = injection_temps[injection_temps >= middle_value]
                first_injector_result = injector_result
            loop_injection_temp.append(middle_value)
            # loop_recovered_temp.append(recovered)
            loop_recovered_temp.append(injector_result.params["teff_2"].value)
            print(injector_result.fit_report())
        else:
            print("Exiting while loop due to len(temps)={}".format(len(injection_temps)))

        print("Boundary of recovery", injection_temps)
        print(first_injector_result.params.__dict__)
        print(first_injector_result.params.pretty_print())
        print(last_injector_result.params.pretty_print())

        show_brute_solution(first_injector_result, star, obsnum, chip, strict_mask=strict_mask)
        show_brute_solution(last_injector_result, star, obsnum, chip, strict_mask=strict_mask)

        print("showing candidates")
        print(first_injector_result.show_candidates(0))
        print(first_injector_result.show_candidates(1))
        print("TODO: Adjust the grid.")
        injector_result = first_injector_result
    else:
        for teff2 in injection_temps:
            injector_result = injector(teff2)
            loop_injection_temp.append(teff2)
            loop_recovered_temp.append(injector_result.params["teff_2"])
        first_injector_result = injector_result

    plt.figure()
    plt.errorbar(loop_injection_temp, loop_recovered_temp, 100*np.ones_like(loop_recovered_temp),  "*")
    plt.plot(loop_injection_temp, loop_injection_temp, "r")
    plt.xlabel("Injected Companion Temp")
    plt.ylabel("Recovered Companion Temp")

    plt.title("logg_2 = {0} comp_logg = {1}".format(injector_result.params["logg_2"].value, comp_logg))
    plt.show()
    return first_injector_result


def is_recovered(injected_value: Union[float, int], injector_result: Any, grid_recovered=False, param="teff_2") -> bool:
    """Is the recovered chi^2 min value within 1-sigma for injected_value.

    Inputs:
    -------
    injected_value: float
        The input parameter value
    injector_result: lmfit result
        Lmnift brute result object.
    grid_recovered: bool
        Flag to set the "criteria 100K" instead of "1sigma". Default False
    param: str
        The prameter name to check. Default "teff_2"
    Outputs
    -------
    Recovered: bool
      True /False if recovered"""
    if grid_recovered:
        # Is the recovered temperature within +-100K
        min_chi2_temp = injector_result.params[param]
        if (min_chi2_temp <= (injected_value +100)) and (min_chi2_temp <= (injected_value +100)):
            recovered =True
        else:
            recovered = False
        if recovered:
            print("Something was recovered within 100K")
    else:
        from mingle.utilities.chisqr import chi2_at_sigma
        print("Injected_temp", injected_value)
        injector_result.redchi
        print("Reduced chi2", injector_result.redchi)
        dof = len(injector_result.var_names)
        one_sigma = chi2_at_sigma(1, dof=dof)
        values_inside_one_sigma = [injector_result.candidates[num].params[param].value
                                   for num in range(50)
                                   if injector_result.candidates[num].score < (
                                           injector_result.candidates[0].score + one_sigma)]
        print("recovered_teffs", values_inside_one_sigma)
        print("one sigma chi^2 = {0}, dof={1}".format(one_sigma, dof))
        print("Candidate1\n", injector_result.show_candidates(0))
        print("Candidate2\n", injector_result.show_candidates(1), "\nabove^^injector_result.show_candidates(1) ^^")

        # Incorrect scaling
        bad_values_inside_one_sigma = [injector_result.candidates[num].params[param].value
                                       for num in range(50)
                                       if injector_result.candidates[num].score < (
                                               injector_result.candidates[0].score + one_sigma * injector_result.redchi)]

        recovered = injected_value in values_inside_one_sigma
        print("recovered = ", recovered)
        if recovered:
            print("Something was recovered")
        elif injected_value in bad_values_inside_one_sigma:
            warnings.warn("The temp value {} was inside the incorrectly scaled sigma range.".format(injected_value))
    return recovered


def show_brute_solution(result, star, obsnum, chip, strict_mask=False):
    parvals = result.params.valuesdict()
    teff_1 = round(parvals['teff_1'] / 100) * 100
    teff_2 = round(parvals['teff_2'] / 100) * 100
    logg_1 = round(parvals['logg_1'] * 2) / 2
    logg_2 = round(parvals['logg_2'] * 2) / 2
    feh_1 = round(parvals['feh_1'] * 2) / 2
    feh_2 = round(parvals['feh_2'] * 2) / 2
    rv_1 = np.asarray([parvals['rv_1']])
    rv_2 = np.asarray([parvals['rv_2']])

    try:
        iter(chip)
    except:
        # Make iterable
        chip = [chip]

    spec_list, error_list = [], []
    for c in chip:
        obs_spec, errors, obs_params = load_observation_with_errors(star, obsnum, c, strict_mask=strict_mask)
        spec_list.append(obs_spec)
        error_list.append(errors * error_fudge)  # Errors larger
    obs_spec, errors = spec_list, error_list
    print("len(obs_spec)", len(obs_spec), "len(chip)", len(chip))
    assert len(obs_spec) == len(chip)

    # Linearlly normalize observation.
    obs_spec = [obs.normalize(method="linear") for obs in obs_spec]
    # closest_host_model, closest_comp_model = closest_obs_params(obs_params, mode="iam")
    plt.figure(figsize=(10, 15))
    for ii, c in enumerate(chip):
        flux_ii, model_ii = iam_magic_sauce(obs_spec[ii],
                                            [teff_1, logg_1, feh_1],
                                            [teff_2, logg_2, feh_2],
                                            rv_1, rv_2,
                                            chip=c, norm_method="linear",
                                            area_scale=True, norm=True,
                                            wav_scale=True, fudge=None)
        plt.subplot(611 + 2 * (ii))
        obs_spec[ii].plot(label="obs")
        plt.plot(obs_spec[ii].xaxis, flux_ii.squeeze(), label="result")
        plt.legend()
        plt.ylim([0.95, 1.02])

        plt.subplot(611 + 2 * (ii) + 1)
        plt.plot(np.concatenate((obs_spec[0].xaxis, obs_spec[1].xaxis, obs_spec[2].xaxis)), result.residual, "*",
                 label="result_residual")
        # plt.plot(obs_spec[ii].xaxis, obs_spec[ii].flux-flux_ii.squeeze(), label="residual")
        plt.plot(obs_spec[ii].xaxis, (obs_spec[ii].flux - flux_ii.squeeze()), label="unscaled residual")
        plt.plot(obs_spec[ii].xaxis, (obs_spec[ii].flux - flux_ii.squeeze()) / error_list[ii], label="scaled residual")
        plt.xlim([np.min(obs_spec[ii].xaxis), np.max(obs_spec[ii].xaxis)])
        plt.title("chip {}".format(c))
        plt.legend()
    plt.show()


if __name__ == "__main__":
    args = vars(parse_args(sys.argv[1:]))
    opts = {k: args[k] for k in args}
    opts.update(comp_logg=4.5)
    answer4p5 = main(**opts)
    opts.update(comp_logg=5.0)
    answer5 = main(**opts)

    print("Found solution for logg 4.5")
    print(fit_report(answer4p5))
    print("Found solution for logg 5")
    print(fit_report(answer5))
