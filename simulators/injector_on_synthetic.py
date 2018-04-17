#!/usr/bin/env python
import argparse
import sys
import warnings
from argparse import Namespace
from typing import List, Union, Any

import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameters, fit_report
from spectrum_overload import Spectrum

from mingle.models.broadcasted_models import inherent_alpha_model
from mingle.utilities.debug_utils import timeit
from mingle.utilities.param_utils import closest_obs_params
from mingle.utilities.phoenix_utils import load_starfish_spectrum
from simulators.common_setup import load_observation_with_errors
from simulators.iam_module import iam_magic_sauce
from simulators.iam_module import observation_rv_limits
from simulators.minimize_iam import brute_solve_iam

error_fudge = 1
binary_search = False

# RV Grid parameters
rv_1, deltarv_1, rv1_step = 0, 2, 0.25
rv_2, deltarv_2, rv2_step = 10, 12, 2


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
    parser.add_argument('-l', '--preloaded', action="store_true",
                        help='Try preloading spectra.')
    parser.add_argument("-g", '--grid_bound', action="store_true",
                        help='Grid bound search limit')
    parser.add_argument("--error", default=None, type=int,
                        help='SNR level to add')
    return parser.parse_args(args)


def synthetic_injector_wrapper(star, obsnum, chip, Ns=20, strict_mask=False, comp_logg=None, plot=False,
                               preloaded=False, error=None):
    """Inject onto a synthetic host spectra. Add noise level of star though."""
    try:
        iter(chip)
    except:
        # Make iterable
        assert chip < 4, "Only chips 1 2 and 3"
        chip = [chip]

    chip_bounds, error_list = [], []

    for c in chip:
        chip_spec, errors, obs_params = load_observation_with_errors(star, obsnum, c, strict_mask=strict_mask)
        error_list.append(errors * error_fudge)
        chip_bounds.append((chip_spec.xaxis[0], chip_spec.xaxis[-1]))
    try:
        del (chip_spec)
    except:
        pass

    if error is None:
        errors = error_list
    else:
        print("Manually setting error = {}!".format(error))
        errors = [1. / error for _ in error_list]  # overwrite with given error value

    print("Error values = ", errors)
    print("len(errors)", len(errors), "len(chip)", len(chip))
    assert len(errors) == len(chip)

    closest_host_model, closest_comp_model = closest_obs_params(obs_params, mode="iam")
    print("\nclosest host model", closest_host_model)
    print("closest comp model", closest_comp_model)
    print()

    # Setup Fixed injection grid parameters
    params = Parameters()
    params.add('teff_1', value=closest_host_model[0], min=4800, max=6600, vary=False, brute_step=100)
    # params.add('logg_1', value=closest_host_model[1], min=0, max=6, vary=False, brute_step=0.5)
    # params.add('feh_1', value=closest_host_model[2], min=-2, max=1, vary=False, brute_step=0.5)
    # params.add('feh_2', value=closest_comp_model[2], min=-2, max=1, vary=False, brute_step=0.5)
    params.add('logg_1', value=4.5, min=0, max=6, vary=False, brute_step=0.5)
    params.add('feh_1', value=0.0, min=-2, max=1, vary=False, brute_step=0.5)
    params.add('feh_2', value=0.0, min=-2, max=1, vary=False, brute_step=0.5)
    params.add('rv_1', value=rv_1, min=rv_1 - deltarv_1, max=rv_1 + deltarv_1, vary=True, brute_step=rv1_step)
    params.add('rv_2', value=rv_2, min=rv_2 - deltarv_2, max=rv_2 + deltarv_2, vary=True, brute_step=rv2_step)
    if comp_logg is None:
        # params.add('logg_2', value=closest_comp_model[1], min=0, max=6, vary=False, brute_step=0.5)
        params.add('logg_2', value=5.0, min=0, max=6, vary=False, brute_step=0.5)
    else:
        params.add('logg_2', value=comp_logg, min=0, max=6, vary=False, brute_step=0.5)

    rv_limits = [observation_rv_limits(obs, params["rv_1"].value,
                                       params["rv_2"].value) for obs in obs_spec]

    mod1_spec = [load_starfish_spectrum(closest_host_model, limits=lim,
                                        hdr=True, normalize=False, area_scale=True,
                                        flux_rescale=True, wav_scale=True) for lim in rv_limits]

    # Currying a function two only take 1 parameter.
    def inject(teff_2):
        """Injector function that just takes a temperature."""
        if teff_2 < 3500:
            upper_limit = 1401
        else:
            upper_limit = 601
        params.add('teff_2', value=teff_2, min=max([teff_2 - 400, 2300]), max=min([teff_2 + upper_limit, 7001]),
                   vary=True,
                   brute_step=100)
        if plot:
            plt.figure()

        injected_spec = []
        print("Injected Teff = ", teff_2)

        for ii, c in enumerate(chip):
            if plot:
                plt.subplot(len(chip), 1, ii + 1)
            mod2_spec = load_starfish_spectrum([teff_2, params["logg_2"].value, params["feh_2"].value],
                                               limits=rv_limits[ii], hdr=True, normalize=False,
                                               area_scale=True, flux_rescale=True, wav_scale=True)

            iam_grid_func = inherent_alpha_model(mod1_spec[ii].xaxis, mod1_spec[ii].flux, mod2_spec.flux,
                                                 rvs=params["rv_2"].value, gammas=params["rv_1"].value)
            synthetic_model_flux = iam_grid_func(chip_waves[ii]).squeeze()

            assert not np.any(np.isnan(
                synthetic_model_flux)), "There are nans in synthetic model flux. Check wavelengths for interpolation"
            synthetic_model = Spectrum(xaxis=chip_waves[ii], flux=synthetic_model_flux)

            continuum = synthetic_model.continuum(method="exponential")

            synthetic_model = synthetic_model / continuum

            synthetic_model.add_noise(1 / error_list[ii])

            injected_spec.append(synthetic_model)
            if plot:
                # obs_spec[ii].plot(label="Observation")
                (mod1_spec[ii] / continuum).plot(label="Host contribution")
                synthetic_model.plot(label="Synthetic binary")
                # injected_chip.plot(label="Would be Injected_chip", lw=1, linestyle="--")
                # shifted_injection.plot(label="injected part", lw=1)
                plt.legend()
        if plot:
            plt.suptitle("Host= {0}, Injected Temperature = {1}".format(closest_host_model[0], teff_2))
            plt.show(block=False)
        return brute_solve_iam(params, injected_spec, errors, chip, Ns=Ns, preloaded=preloaded)

    print("injector", inject)

    return inject


@timeit
def main(star, obsnum, **kwargs):
    """Main function."""
    chip = [1, 2, 3]
    loop_injection_temp = []
    loop_recovered_temp = []
    print("before injector")
    # strict_mask = kwargs.get("strict_mask", False)
    strict_mask = True
    grid_recovered = kwargs.get("grid_bound", False)
    comp_logg = kwargs.get("comp_logg", None)
    plot = kwargs.get("plot", False)
    preloaded = kwargs.get("preloaded", False)
    # injector = injector_wrapper(star, obsnum, chip, Ns=20, strict_mask=strict_mask, comp_logg=comp_logg, plot=plot)
    injector = synthetic_injector_wrapper(star, obsnum, chip, Ns=20, strict_mask=strict_mask, comp_logg=comp_logg,
                                          plot=plot, preloaded=preloaded)

    injection_temps = np.arange(2300, 5001, 100)
    binary_search = kwargs.get("binary", False)
    if binary_search:
        print("before first")
        first = injection_temps[0]
        first_injector_result = injector(first)
        loop_injection_temp.append(first)
        loop_recovered_temp2.append(first_injector_result.params["teff_2"].value)
        loop_recovered_rv2.append(first_injector_result.params["rv_2"].value)
        loop_recovered_rv1.append(first_injector_result.params["rv_1"].value)
        # show_synth_brute_solution(first_injector_result, star, obsnum, chip, strict_mask=strict_mask)
        print("done first")
        last = injection_temps[-1]
        last_injector_result = injector(last)
        loop_injection_temp.append(last)
        loop_recovered_temp2.append(last_injector_result.params["teff_2"].value)
        loop_recovered_rv2.append(last_injector_result.params["rv_2"].value)
        loop_recovered_rv1.append(last_injector_result.params["rv_1"].value)
        # show_synth_brute_solution(last_injector_result, star, obsnum, chip, strict_mask=strict_mask)
        print("Done last")
        first_recovered = is_recovered(first, first_injector_result)

        # The first one should not be recovered.
        if first_recovered:
            show_synth_brute_solution(first_injector_result, star, obsnum, chip, strict_mask=False, preloaded=preloaded)

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
            loop_recovered_temp2.append(injector_result.params["teff_2"].value)
            loop_recovered_rv2.append(injector_result.params["rv_2"].value)
            loop_recovered_rv1.append(injector_result.params["rv_2"].value)
        else:
            print("Exiting while loop due to len(temps)={}".format(len(injection_temps)))

        print("Boundary of recovery", injection_temps)
        print(first_injector_result.params.__dict__)
        print(first_injector_result.params.pretty_print())
        print(last_injector_result.params.pretty_print())

        show_synth_brute_solution(first_injector_result, star, obsnum, chip, strict_mask=strict_mask,
                                  preloaded=preloaded)
        show_synth_brute_solution(last_injector_result, star, obsnum, chip, strict_mask=strict_mask,
                                  preloaded=preloaded)

        print("showing candidates")
        print(first_injector_result.show_candidates(0))

        injector_result = first_injector_result
    else:
        for teff2 in injection_temps:
            injector_result = injector(teff2)
            loop_injection_temp.append(teff2)
            loop_recovered_temp2.append(injector_result.params["teff_2"].value)
            loop_recovered_rv2.append(injector_result.params["rv_2"].value)
            loop_recovered_rv1.append(injector_result.params["rv_1"].value)
        first_injector_result = injector_result

    fname = f"{star}_injector_results_logg={comp_logg}_error={error}_chip_{chip}_rv2{rv_2}.txt"
    with open(fname, "w") as f:
        f.write("# Injection - recovery results")
        if error is None:
            f.write(f"# Noise level = beta-sigma observed\n")
        else:
            f.write(f"# Noise level = {error}\n")
        f.write("input\t output\t rv1\t rv2")
        for input_, output, rv1, rv2 in zip(loop_injection_temp, loop_recovered_temp2, loop_recovered_rv1,
                                            loop_recovered_rv2):
            f.write(f"{input_}\t{output}\t{rv1}\t{rv2}\n")

    # plot the injection-recovery
    plt.figure()
    plt.errorbar(loop_injection_temp, loop_recovered_temp, yerr=temp_err, fmt="r*")
    temp_err = 100 * np.ones_like(loop_recovered_temp2)
    plt.plot(loop_injection_temp, loop_injection_temp, "r")
    plt.xlabel("Injected Companion Temp")
    plt.ylabel("Recovered Companion Temp")

    plt.title(
        "synthetic injector! logg_2 = {0} comp_logg = {1}".format(injector_result.params["logg_2"].value, comp_logg))
    plt.show()
    return first_injector_result


def is_recovered(injected_value: Union[float, int], injector_result: Any, grid_recovered=False, param="teff_2") -> bool:
    """Is the recovered chi^2 min value within 1-sigma for injected_value.

    Inputs:
    -------
    injected_value: float
        The input parameter value
    injector_result: lmfit result
        Lmfit brute result object.
    grid_recovered: bool
        Flag to set the "criteria 100K" instead of "1sigma". Default False
    param: str
        The parameter name to check. Default "teff_2"
    Outputs
    -------
    Recovered: bool
      True /False if recovered"""
    if grid_recovered:
        # Is the recovered temperature within +-100K
        min_chi2_temp = injector_result.params[param]
        if (min_chi2_temp <= (injected_value + 100)) and (min_chi2_temp <= (injected_value + 100)):
            recovered = True
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

        # Incorrect scaling
        bad_values_inside_one_sigma = [injector_result.candidates[num].params[param].value
                                       for num in range(50)
                                       if injector_result.candidates[num].score < (
                                               injector_result.candidates[
                                                   0].score + one_sigma * injector_result.redchi)]

        recovered = injected_value in values_inside_one_sigma
        print("recovered = ", recovered)
        if recovered:
            print("Something was recovered", injected_value, "==", injector_result.params["teff_2"])
        elif injected_value in bad_values_inside_one_sigma:
            warnings.warn("The temp value {} was inside the incorrectly scaled sigma range.".format(injected_value))
    return recovered


def show_synth_brute_solution(result, star, obsnum, chip, strict_mask=False, preloaded=False):
    parvals = result.params.valuesdict()
    teff_1 = parvals['teff_1']
    teff_2 = parvals['teff_2']
    logg_1 = parvals['logg_1']
    logg_2 = parvals['logg_2']
    feh_1 = parvals['feh_1']
    feh_2 = parvals['feh_2']
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

    closest_host_model, closest_comp_model = closest_obs_params(obs_params, mode="iam")
    rv_limits = [observation_rv_limits(obs, rv_1, rv_2) for obs in obs_spec]

    mod1_spec = [load_starfish_spectrum(closest_host_model, limits=lim,
                                        hdr=True, normalize=False, area_scale=True,
                                        flux_rescale=True, wav_scale=True) for lim in rv_limits]
    synth_obs = mod1_spec

    plt.figure(figsize=(10, 15))
    for ii, c in enumerate(chip):
        flux_ii, model_ii = iam_magic_sauce(synth_obs[ii],
                                            [teff_1, logg_1, feh_1],
                                            [teff_2, logg_2, feh_2],
                                            rv_1, rv_2,
                                            chip=c, norm_method="linear",
                                            area_scale=True, norm=True,
                                            wav_scale=True, fudge=None, preloaded=preloaded)
        plt.subplot(611 + 2 * ii)
        obs_spec[ii].plot(label="obs")
        plt.plot(synth_obs[ii].xaxis, flux_ii.squeeze(), label="result")
        plt.legend()
        plt.ylim([0.95, 1.02])
        # TODO: Fix up this plotting
        plt.subplot(611 + 2 * ii + 1)
        print(result.residual.shape)
        print(synth_obs[0].xaxis.shape)
        # plt.plot(np.concatenate((synth_obs[0].xaxis, synth_obs[1].xaxis, synth_obs[2].xaxis)), result.residual, "*",
        #         label="result_residual")
        # plt.plot(obs_spec[ii].xaxis, obs_spec[ii].flux-flux_ii.squeeze(), label="residual")
        # plt.plot(synth_obs[ii].xaxis, (synth_obs[ii].flux - flux_ii.squeeze()), label="unscaled residual")
        # plt.plot(synth_obs[ii].xaxis, (synth_obs[ii].flux - flux_ii.squeeze()) / error_list[ii], label="scaled residual")
        plt.xlim([np.min(synth_obs[ii].xaxis), np.max(synth_obs[ii].xaxis)])
        plt.title("chip {}".format(c))
        plt.legend()
    plt.show()


if __name__ == "__main__":
    args = vars(parse_args(sys.argv[1:]))
    opts = {k: args[k] for k in args}

    if opts["preloaded"]:
        from mingle.utilities.phoenix_utils import preload_spectra

        preload_spectra()
        print("finished preloading")

    opts.update(comp_logg=4.5)
    answer4p5 = main(**opts)

    opts.update(comp_logg=5.0)
    answer5 = main(**opts)

    print("Found solution for logg 4.5")
    print(fit_report(answer4p5))

    print("Found solution for logg 5")
    print(fit_report(answer5))
