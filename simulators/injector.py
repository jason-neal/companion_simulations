#!/usr/bin/env python
import argparse
import sys
from argparse import Namespace
from typing import List

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
    parser.add_argument('-l', '--preloaded', action="store_true",
                        help='Try preloading spectra.')
    parser.add_argument("-g", '--grid_bound', action="store_true",
                        help='Grid bound search limit')

    return parser.parse_args(args)


def injector_wrapper(star, obsnum, chip, Ns=20, teff_1=None, rv_1=None, strict_mask=False, comp_logg=None, plot=False,
                     preloaded=False):
    """Take the Observation and prepare to inject different temperature companions."""
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

    # Linearly normalize observation.
    obs_spec = [obs.normalize(method="linear") for obs in obs_spec]
    closest_host_model, closest_comp_model = closest_obs_params(obs_params, mode="iam")

    if rv_1 is None:
        rv_1 = 0
        rv1_vary = False
    else:
        rv1_vary = True
    rv_2 = 25
    deltarv_1 = 2
    deltarv_2 = 10
    rv1_step = 0.25
    rv2_step = 1

    if teff_1 is None:
        teff_1 = closest_host_model[0]

    # Setup Fixed injection grid parameters
    params = Parameters()
    params.add('teff_1', value=teff_1, min=5000, max=6000, vary=False, brute_step=100)
    params.add('logg_1', value=closest_host_model[1], min=0, max=6, vary=False, brute_step=0.5)
    params.add('feh_1', value=closest_host_model[2], min=-2, max=1, vary=False, brute_step=0.5)
    params.add('feh_2', value=closest_comp_model[2], min=-2, max=1, vary=False, brute_step=0.5)
    params.add('rv_1', value=rv_1, min=rv_1 - deltarv_1, max=rv_1 + deltarv_1, vary=rv1_vary, brute_step=rv1_step)
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

    # Currying a function two only take 1 parameter.
    def inject(teff_2):
        """Injector function that just takes a temperature."""

        if teff_2 < 3500:
            upper_limit = 1401
        else:
            upper_limit = 601

        params.add('teff_2', value=teff_2, min=max([teff_2 - 600, 2300]), max=min([teff_2 + upper_limit, 7001]),
                   vary=True,
                   brute_step=100)
        if plot:
            plt.figure()
        # Add companion to observation

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
            injected_chip = obs_spec[ii].copy() + injection
            shifted_injection = injection + 1.01 - np.mean(injection.flux)

            # Re-normalzie
            injected_chip2 = injected_chip.normalize(method="linear")
            assert not np.any(np.isnan(injection.flux))

            injected_spec.append(injected_chip2)
            if plot:
                obs_spec[ii].plot(label="Observation")
                injected_chip.plot(label="Un-normalized Injected_chip", lw=1, linestyle="--")
                injected_chip2.plot(label="Injected_chip", lw=1, linestyle="--")
                shifted_injection.plot(label="injected part + 1.01", lw=1)
                plt.legend()
                from bin.radius_ratio import flux_ratio
                f_ratio = flux_ratio(params["teff_1"].value, params["logg_1"].value, params["feh_1"].value,
                                     params["teff_2"].value, params["logg_2"].value, params["feh_2"].value)
                plt.annotate("Flux ratio F2/F1 = {0:5.03}".format(1/f_ratio), (0.01, 0.2), xycoords="axes fraction")
        if plot:
            plt.suptitle("Host= {0}, Injected Temperature = {1}".format(closest_host_model[0], teff_2))
            plt.show(block=False)
        return brute_solve_iam(params, injected_spec, errors, chip, Ns=Ns, preloaded=preloaded)

    print("injector ", inject)

    return inject


@timeit
def main(star, obsnum, **kwargs):
    """Main function."""
    chip = [1, 2, 3]
    loop_injection_temp = []
    loop_recovered_temp = []
    loop_recovered_rv1 = []
    loop_recovered_rv2 = []
    print("before injector")
    strict_mask = kwargs.get("strict_mask", False)
    # grid_recovered = kwargs.get("grid_bound", False)
    comp_logg = kwargs.get("comp_logg", None)
    plot = kwargs.get("plot", False)
    preloaded = kwargs.get("preloaded", False)

    # FIT BEST HOST MODEL TO OBSERVATIONS
    from simulators.minimize_bhm import main as minimize_bhm
    result = minimize_bhm(star, obsnum, 1)
    print(result)
    print("Teff_1 =", result.params["teff_1"].value)
    print("rv_1 =", result.params["rv_1"].value)

    # Adding teff_1 and rv_1 to fix those parameters.
    injector = injector_wrapper(star, obsnum, chip,
                                Ns=20, strict_mask=strict_mask,
                                comp_logg=comp_logg, plot=plot,
                                preloaded=preloaded, teff_1=result.params["teff_1"].value,
                                rv_1=result.params["rv_1"].value)

    injection_temps = np.arange(2300, 5001, 100)
    for teff2 in injection_temps:
        injector_result = injector(teff2)
        loop_injection_temp.append(teff2)
        loop_recovered_temp.append(injector_result.params["teff_2"].value)
        loop_recovered_rv1.append(injector_result.params["rv_1"].value)
        loop_recovered_rv2.append(injector_result.params["rv_2"].value)
        first_injector_result = injector_result

    filename = f"{star}_real_injector_results_logg={comp_logg}_obs{obsnum}.txt"
    with open(filename, "w") as f:
        f.write("# Real Injection - recovery results")
        f.write("# ")
        f.write("input\t output\t rv1\t rv2")
        for input_, output, rv1, rv2 in zip(loop_injection_temp, loop_recovered_temp, loop_recovered_rv1,
                                            loop_recovered_rv2):
            f.write(f"{input_}\t{output}\t{rv1}\t{rv2}\n")

    plt.figure()
    temp_err = 100 * np.ones_like(loop_recovered_temp)
    plt.errorbar(loop_injection_temp, loop_recovered_temp, yerr=temp_err, fmt="r*")
    plt.plot(loop_injection_temp, loop_injection_temp, "r")
    plt.xlabel("Injected Companion Temp")
    plt.ylabel("Recovered Companion Temp")

    plt.title("Injector: logg_1 = {0} logg_2 = {1}".format(injector_result.params["logg_1"].value,
                                                           injector_result.params["logg_2"].value))
    plt.savefig(kwargs.get("plot_name", "Test_recovery_plot.pdf"))
    plt.tight_layout()
    plt.show()

    return first_injector_result


def show_brute_solution(result, star, obsnum, chip, strict_mask=False, preloaded=False):
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
    # closest_host_model, closest_comp_model = closest_obs_params(obs_params, mode="iam")
    plt.figure(figsize=(10, 15))
    for ii, c in enumerate(chip):
        flux_ii, model_ii = iam_magic_sauce(obs_spec[ii],
                                            [teff_1, logg_1, feh_1],
                                            [teff_2, logg_2, feh_2],
                                            rv_1, rv_2,
                                            chip=c, norm_method="linear",
                                            area_scale=True, norm=True,
                                            wav_scale=True, fudge=None, preloaded=preloaded)
        plt.subplot(611 + 2 * ii)
        obs_spec[ii].plot(label="obs")
        plt.plot(obs_spec[ii].xaxis, flux_ii.squeeze(), label="result")
        plt.legend()
        plt.ylim([0.95, 1.02])

        plt.subplot(611 + 2 * ii + 1)
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

    # preload = False
    # opts["preload"] = preload

    if opts["preloaded"]:
        from mingle.utilities.phoenix_utils import preload_spectra

        preload_spectra()
        print("Finished preloading")

    opts.update(comp_logg=4.5)
    answer4p5 = main(**opts)

    opts.update(comp_logg=5.0)
    answer5 = main(**opts)

    print("Found solution for logg 4.5")
    print(fit_report(answer4p5))
    print("Found solution for logg 5")
    print(fit_report(answer5))
