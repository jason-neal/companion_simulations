#!/usr/bin/env python
import argparse
import copy
import sys
from argparse import Namespace
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameters
from spectrum_overload import Spectrum

from mingle.models.broadcasted_models import inherent_alpha_model
from mingle.utilities.debug_utils import timeit
from mingle.utilities.param_utils import closest_obs_params
from mingle.utilities.phoenix_utils import load_starfish_spectrum
from simulators.common_setup import load_observation_with_errors
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
    parser.add_argument('-l', '--preloaded', action="store_true",
                        help='Try preloading spectra.')
    parser.add_argument("-g", '--grid_bound', action="store_true",
                        help='Grid bound search limit')
    parser.add_argument("--error", default=None, type=int,
                        help='SNR level to add')
    parser.add_argument("-c", "--chip", default=None, type=str,
                        help='Chips 2 use e.g. "1, 2, 3"')
    return parser.parse_args(args)


def synthetic_injector_wrapper(star, obsnum, chip, strict_mask=False, comp_logg=None, plot=False, error=None):
    """Inject onto a synthetic host spectra. Add noise level of star though.

    if error is not None it is the SNR level to experiment with"""
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
        del chip_spec

    if error is None:
        snr = [1. / err for err in error_list]
        errors = error_list
    else:
        print("Manually setting error = {}!".format(error))
        snr = [error for _ in error_list]  # overwrite with given error value
        errors = [1. / error for _ in error_list]  # overwrite with given error value
    print("Error values = ", errors)
    print("len(snr)", len(snr), "len(chip)", len(chip))
    assert len(snr) == len(chip), "Should be same lenght: len(snr)={}, len(chip)={}".format(
        len(snr), len(chip))

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
    params.add('rv_1', value=rv_1, min=rv_1 - deltarv_1, max=rv_1 + deltarv_1, vary=False, brute_step=rv1_step)
    params.add('rv_2', value=rv_2, min=rv_2 - deltarv_2, max=rv_2 + deltarv_2, vary=True, brute_step=rv2_step)
    if comp_logg is None:
        # params.add('logg_2', value=closest_comp_model[1], min=0, max=6, vary=False, brute_step=0.5)
        params.add('logg_2', value=5.0, min=0, max=6, vary=False, brute_step=0.5)
    else:
        params.add('logg_2', value=comp_logg, min=0, max=6, vary=False, brute_step=0.5)

    rv_limits = [(2111, 2125), (2127, 2138), (2141, 2153)]

    mod1_spec = [load_starfish_spectrum(closest_host_model, limits=lim,
                                        hdr=True, normalize=False, area_scale=True,
                                        flux_rescale=True, wav_scale=True) for lim in rv_limits]
    # Resample mod1_spec to 1024 pixels only
    npix = 1024
    assert len(mod1_spec[0]) != npix
    chip_waves = [np.linspace(bound[0], bound[1], npix) for bound in chip_bounds]
    for chip_wave in chip_waves:
        assert len(chip_wave) == npix

    # Currying a function two only take 1 parameter.
    def inject(teff_2):
        """Injector function that just takes a temperature."""
        if teff_2 < 3500:
            upper_limit = 1401
        else:
            upper_limit = 601
        inject_params = copy.deepcopy(params)
        inject_params.add('teff_2', value=teff_2, min=max([teff_2 - 400, 2300]), max=min([teff_2 + upper_limit, 7001]),
                          vary=True,
                          brute_step=100)
        if plot:
            plt.figure()

        injected_spec = []
        print("Injected Teff = ", teff_2)

        for ii, c in enumerate(chip):
            if plot:
                plt.subplot(len(chip), 1, ii + 1)
            mod2_spec = load_starfish_spectrum([teff_2, inject_params["logg_2"].value, inject_params["feh_2"].value],
                                               limits=rv_limits[ii], hdr=True, normalize=False,
                                               area_scale=True, flux_rescale=True, wav_scale=True)

            iam_grid_func = inherent_alpha_model(mod1_spec[ii].xaxis, mod1_spec[ii].flux, mod2_spec.flux,
                                                 rvs=inject_params["rv_2"].value, gammas=inject_params["rv_1"].value)
            synthetic_model_flux = iam_grid_func(chip_waves[ii]).squeeze()

            assert not np.any(np.isnan(
                synthetic_model_flux)), "There are nans in synthetic model flux. Check wavelengths for interpolation"
            synthetic_model = Spectrum(xaxis=chip_waves[ii], flux=synthetic_model_flux)

            continuum = synthetic_model.continuum(method="exponential")

            synthetic_model = synthetic_model / continuum

            synthetic_model.add_noise(snr[ii])

            injected_spec.append(synthetic_model)
            if plot:
                # obs_spec[ii].plot(label="Observation")
                (mod1_spec[ii] / continuum).plot(label="Host contribution")
                synthetic_model.plot(label="Synthetic binary")
                # injected_chip.plot(label="Would be Injected_chip", lw=1, linestyle="--")
                # shifted_injection.plot(label="injected part", lw=1)
                plt.legend()

        if plot:
            plt.suptitle("Host = {0}, Injected Temperature = {1}".format(closest_host_model[0], teff_2))
            plt.show(block=False)

        assert len(injected_spec) == len(chip), "Should be same lenght: len(injected_spec)={}, len(chip)={}".format(
        len(injected_spec), len(chip))
        # return brute_solve_iam(params, injected_spec, errors, chip, Ns=Ns, preloaded=preloaded)
        return inject_params, injected_spec, errors, chip

    print("injector", inject)

    return inject, params


@timeit
def main(star, obsnum, **kwargs):
    """Main function."""
    comp_logg = kwargs.get("comp_logg", None)
    plot = kwargs.get("plot", False)
    preloaded = kwargs.get("preloaded", False)
    error = kwargs.get("error", None)  # None means use from the observations

    chip = kwargs.get("chip", [1, 2, 3])
    loop_injection_temp = []
    loop_recovered_temp2 = []
    loop_recovered_rv1 = []
    loop_recovered_rv2 = []

    print("Before injector")

    strict_mask = kwargs.get("strict_mask", False)

    # injector = injector_wrapper(star, obsnum, chip, Ns=20, strict_mask=strict_mask, comp_logg=comp_logg, plot=plot)
    injector, initial_params = synthetic_injector_wrapper(star, obsnum, chip, strict_mask=strict_mask,
                                                          comp_logg=comp_logg,
                                                          plot=plot, error=error)
    print("inital params set:")
    initial_params.pretty_print()

    injection_temps = np.arange(2300, 5001, 100)

    fname = f"{star}_injector_results_logg={comp_logg}_error={error}_chip_{chip}_rv2{rv_2}.txt"
    with open(fname, "w") as f:
        f.write("# Injection - recovery results\n")
        f.write("# Initial_vals:\n")
        f.write("# teff_1={}, logg_2={}, rv_1={}, rv_2={}\n".format(
            *(initial_params[key].value for key in ["teff_1", "logg_2", "rv_1", "rv_2"])))
        if error is None:
            f.write(f"# Noise level = beta-sigma observed\n")
        else:
            f.write(f"# Noise level = {error}\n")
        f.write("input\t output\t rv1\t rv2\n")

        for teff2 in injection_temps[::-1]:
            injected_values = injector(teff2)
            injector_result = brute_solve_iam(*injected_values, Ns=20, preloaded=preloaded)
            print("Recovered temp = {} K".format(injector_result.params["teff_2"].value))
            loop_injection_temp.append(teff2)
            loop_recovered_temp2.append(injector_result.params["teff_2"].value)
            loop_recovered_rv2.append(injector_result.params["rv_2"].value)
            loop_recovered_rv1.append(injector_result.params["rv_1"].value)

            f.write(f"{teff2}\t{loop_recovered_temp2[-1]}\t{loop_recovered_rv1[-1]}\t{loop_recovered_rv2[-1]}\n")

    # fname = f"{star}_injector_results_logg={comp_logg}_error={error}_chip_{chip}_rv2{rv_2}.txt"
    # with open(fname, "w") as f:
    #     f.write("# Injection - recovery results\n")
    #     f.write("# Initial_vals:\n")
    #     f.write("# teff_1={}, logg_2={}, rv_1={}, rv_2={}\n".format(
    #         initial_params[key].value for key in ["teff_1", "logg_2", "rv_1", "rv_2"]))
    #     if error is None:
    #         f.write(f"# Noise level = beta-sigma observed\n")
    #     else:
    #         f.write(f"# Noise level = {error}\n")
    #     f.write("input\t output\t rv1\t rv2\n")
    #     for input_, output, rv1, rv2 in zip(loop_injection_temp, loop_recovered_temp2, loop_recovered_rv1,
    #                                         loop_recovered_rv2):
    #         f.write(f"{input_}\t{output}\t{rv1}\t{rv2}\n")

    # plot the injection-recovery
    plt.figure()
    # ax1 = plt.subplot(211)
    ax1 = plt.subplot(111)
    temp_err = 100 * np.ones_like(loop_recovered_temp2)
    ax1.errorbar(loop_injection_temp, loop_recovered_temp2, yerr=temp_err, fmt=".", color="C1")
    ax1.plot(loop_injection_temp, loop_injection_temp, "r")
    plt.xlabel("Injected Temperature (K)")
    plt.ylabel("Recovered Temperature (km/s)")

    plt.title(
        "synthetic injector: logg_1 = {0} logg_2 = {1}".format(injector_result.params["logg_1"].value,
                                                               injector_result.params["logg_2"].value))

    # ax2 = plt.subplot(212, sharex=ax1)
    # plt.plot(loop_injection_temp, loop_recovered_rv1, "C2*", label="rv_1")
    # plt.axhline(rv_1, color="C1", alpha=0.6)
    # plt.plot(loop_injection_temp, loop_recovered_rv2, "C3o", label="rv_2")
    # plt.axhline(rv_2, color="C2", alpha=0.6)
    # plt.ylabel("Recovered RV (km/s")
    # plt.xlabel("Injected Temperature (K)")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    return None


if __name__ == "__main__":
    args = vars(parse_args(sys.argv[1:]))
    opts = {k: args[k] for k in args}
    chip = opts.get("chip")
    if chip is not None:
        chip = [int(c) for c in opts["chip"].split(",")]
        opts["chip"] = chip
    else:
        opts["chip"] = [1, 2, 3]

    if opts["preloaded"]:
        from mingle.utilities.phoenix_utils import preload_spectra

        preload_spectra()
        print("finished preloading")

    opts.update(comp_logg=4.5)
    answer4p5 = main(**opts)

    opts.update(comp_logg=5.0)
    answer5 = main(**opts)
