#!/usr/bin/env python
import argparse
import logging
import sys
from argparse import Namespace
from typing import List

# Minimize function
import numpy as np
from lmfit import Parameters, fit_report, Minimizer
from mingle.utilities.debug_utils import timeit2
from mingle.utilities.param_utils import closest_obs_params
from simulators.common_setup import load_observation_with_errors
from simulators.iam_module import iam_chi2_magic_sauce, iam_magic_sauce
from simulators.iam_module import (setup_iam_dirs)
from simulators.minimize_bhm import brute_solve_bhm
from spectrum_overload import Spectrum

logging.basicConfig(level=logging.WARNING,
                    format='%(levelname)s %(message)s')


@timeit2
def main(star, obsnum, chip):
    star = star.upper()
    setup_iam_dirs(star)

    # Setup comparision spectra
    obs_spec, errors, obs_params = load_observation_with_errors(star, obsnum, chip)

    closest_host_model, closest_comp_model = closest_obs_params(obs_params, mode="iam")

    params = Parameters()

    params.add('teff_1', value=closest_host_model[0], min=closest_host_model[0]-100, max=closest_host_model[0]+100, vary=False, brute_step=100)
    params.add('teff_2', value=closest_comp_model[0], min=closest_comp_model[0]-600, max=closest_comp_model[0]+600, vary=True, brute_step=100)
    params.add('logg_1', value=closest_host_model[1], min=0, max=6, vary=False, brute_step=0.5)
    params.add('logg_2', value=closest_comp_model[1], min=0, max=6, vary=False, brute_step=0.5)
    params.add('feh_1', value=closest_host_model[2], min=-2, max=1, vary=False, brute_step=0.5)
    params.add('feh_2', value=closest_comp_model[2], min=-2, max=1, vary=False, brute_step=0.5)
    params.add('rv_1', value=7, min=-20, max=20, vary=True, brute_step=0)
    params.add('rv_2', value=1.5, min=-10, max=10, vary=True, brute_step=0)

    result = brute_solve_iam(params, obs_spec, errors, chip, Ns=20)
    result_bhm = brute_solve_bhm(params, obs_spec, errors, chip, Ns=20)

    print("Results")
    result.params.pretty_print()

    print("Chi2", result.chisqr)
    print("Reduced Chi2", result.redchi)

    print("Fit report", fit_report(result.params))

    # All chip version
    spec_list, error_list = [], []
    chips = [1, 2, 3, 4]
    for chip in [1, 2, 3]:
        obs_spec, errors, obs_params = load_observation_with_errors(star, obsnum, chip)
        spec_list.append(obs_spec)
        error_list.append(errors)

    brute_solve_iam(params, spec_list, error_list, chips)


def brute_solve_iam(params, obs_spec, errors, chip, Ns=20):
    if isinstance(obs_spec, list):
        minimize_wav = [obs.xaxis for obs in obs_spec]
        minimize_flux = [obs.flux for obs in obs_spec]
        assert len(minimize_flux) == len(errors)
        assert len(minimize_wav) == len(chip)
    else:
        minimize_wav = obs_spec.xaxis
        minimize_flux = obs_spec.flux

    kws = {"chip": chip, "norm": True, "norm_method": "linear",
           "area_scale": True, "wav_scale": True, "fudge": None}

    # Least-squares fit to the spectrum.
    mini = Minimizer(func_array, params, fcn_args=(minimize_wav, minimize_flux, errors), fcn_kws=kws)
    # Evaluate 20 points on each axis and keep all points candidates
    result = mini.minimize(method="brute", Ns=Ns, keep="all")

    print("Results")
    # result.params.pretty_print()

    print("Chi2", result.chisqr)
    print("Reduced Chi2", result.redchi)
    # conf_interval does not work with brute force method
    # ci = lmfit.conf_interval(mini, result)
    # lmfit.printfuncs.report_ci(ci)
    print("Fit report", fit_report(result.params))
    return result


def func_chi2(pars, obs_wav, obs_flux, chip=None, norm=True, norm_method="scalar",
              area_scale=True, wav_scale=True, fudge=None, arb_norm=False, errors=None):
    """Calculate binary model chi^2 for given parameters and observation"""
    # unpack parameters: extract .value attribute for each parameter
    parvals = pars.valuesdict()
    teff_1 = parvals['teff_1']
    teff_2 = parvals['teff_2']
    logg_1 = parvals['logg_1']
    logg_2 = parvals['logg_2']
    feh_1 = parvals['feh_1']
    feh_2 = parvals['feh_2']
    rv_1 = np.asarray([parvals['rv_1']])
    rv_2 = np.asarray([parvals['rv_2']])

    chi2_value = iam_chi2_magic_sauce(Spectrum(xaxis=obs_wav, flux=obs_flux),
                                      [teff_1, logg_1, feh_1],
                                      [teff_2, logg_2, feh_2],
                                      rv_1, rv_2,
                                      chip=chip, norm_method=norm_method,
                                      area_scale=area_scale, norm=norm,
                                      wav_scale=wav_scale, fudge=fudge,
                                      arb_norm=arb_norm, errors=errors)
    return chi2_value


def func_array(pars, obs_wav, obs_flux, errors, chip=None, norm=True, norm_method="scalar",
               area_scale=True, wav_scale=True, fudge=None):
    """Calculate binary model chi^2 for given parameters and observation"""
    # unpack parameters: extract .value attribute for each parameter
    parvals = pars.valuesdict()
    teff_1 = round(parvals['teff_1'] / 100) * 100
    teff_2 = round(parvals['teff_2'] / 100) * 100
    logg_1 = round(parvals['logg_1'] * 2) / 2
    logg_2 = round(parvals['logg_2'] * 2) / 2
    feh_1 = round(parvals['feh_1'] * 2) / 2
    feh_2 = round(parvals['feh_2'] * 2) / 2
    rv_1 = np.asarray([parvals['rv_1']])
    rv_2 = np.asarray([parvals['rv_2']])

    arb_norm = parvals.get("arb_norm", 1)  # 1 if not provided

    if isinstance(chip, list):
        # Do multiple chips at once, append them together
        assert len(chip) == len(obs_wav)
        assert len(obs_flux) == len(obs_wav)

        flux = np.empty((0,))
        model = np.empty((0,))
        error_array = np.empty((0,))

        for ii, c in enumerate(chip):
            flux_ii, model_ii = iam_magic_sauce(Spectrum(xaxis=obs_wav[ii], flux=obs_flux[ii]),
                                                [teff_1, logg_1, feh_1],
                                                [teff_2, logg_2, feh_2],
                                                rv_1, rv_2,
                                                chip=c, norm_method=norm_method,
                                                area_scale=area_scale, norm=norm,
                                                wav_scale=wav_scale, fudge=fudge)

            flux = np.concatenate((flux, flux_ii.squeeze()))
            model = np.concatenate((model, model_ii.squeeze()))
            error_array = np.concatenate((error_array, errors[ii] * np.ones_like(flux_ii.squeeze())))
        errors = error_array
    else:
        flux, model = iam_magic_sauce(Spectrum(xaxis=obs_wav, flux=obs_flux),
                                      [teff_1, logg_1, feh_1],
                                      [teff_2, logg_2, feh_2],
                                      rv_1, rv_2,
                                      chip=chip, norm_method=norm_method,
                                      area_scale=area_scale, norm=norm,
                                      wav_scale=wav_scale, fudge=fudge)
        flux = flux.squeeze()
        model = model.squeeze()
    residual = model - (flux * arb_norm)

    # Scale to make chi-square sensible.
    scaled_residual = residual / errors
    return scaled_residual


def parse_args(args: List[str]) -> Namespace:
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Minimize iam script.')
    parser.add_argument("-s", "--star", help='Star name.', type=str, default="HD211847")
    parser.add_argument("-o", "--obsnum", help='Star observation number.', type=str, default="2")
    parser.add_argument("-c", "--chip", help='Star chip number.', default=1)
    # parser.add_argument("-m", "--strict_mask", help="Use strict masking", action="store_true")
    # parser.add_argument('-v', '--verbose', action="store_true",
    #                    help='Turn on Verbose.')
    return parser.parse_args(args)


if __name__ == "__main__":
    args = vars(parse_args(sys.argv[1:]))
    opts = {k: args[k] for k in args}
    main(**opts)

    # star = "HD211847"
    # obsnum = 1
    # chip = 1
    # main(star, obsnum, chip)

    print("Done")
