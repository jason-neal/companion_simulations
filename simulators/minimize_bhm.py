import logging

# Minimize function
import lmfit
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameters, fit_report, Minimizer
from logutils import BraceMessage as __
from spectrum_overload import Spectrum

import simulators
from mingle.utilities.crires_utilities import barycorr_crires_spectrum
from mingle.utilities.debug_utils import timeit2
from mingle.utilities.errors import betasigma_error
from mingle.utilities.masking import spectrum_masking
from mingle.utilities.spectrum_utils import load_spectrum
from simulators.iam_module import iam_magic_sauce
from simulators.bhm_module import bhm_magic_sauce
from simulators.bhm_module import (bhm_helper_function, setup_bhm_dirs)

from mingle.utilities.param_utils import closest_obs_params, closest_model_params

logging.basicConfig(level=logging.WARNING,
                    format='%(levelname)s %(message)s')
from simulators.minimize_iam import load_observation


@timeit2
def main(star, obsnum, chip):
    star = star.upper()
    setup_bhm_dirs(star)

    # Setup comparision spectra
    obs_spec, errors, obs_params = load_observation(star, obsnum, chip)

    closest_host_model = closest_obs_params(obs_params, mode="bhm")

    params = Parameters()

    params.add('teff_1', value=closest_host_model[0], min=5600, max=5800, vary=True, brute_step=100)
    #  params.add('teff_2', value=closest_comp_model[0], min=3000, max=3400, vary=False, brute_step=100)
    params.add('logg_1', value=closest_host_model[1], min=0, max=6, vary=False, brute_step=0.5)
    # params.add('logg_2', value=closest_comp_model[1], min=0, max=6, vary=False, brute_step=0.5)
    params.add('feh_1', value=closest_host_model[2], min=-2, max=1, vary=False, brute_step=0.5)
    # params.add('feh_2', value=closest_comp_model[2], min=-2, max=1, vary=False, brute_step=0.5)
    params.add('rv_1', value=7, min=-20, max=20, vary=True, brute_step=0)
    # params.add('rv_2', value=1.5, min=-10, max=10, vary=True, brute_step=0)

    # result = brute_solve_iam(params, obs_spec, errors, chip, Ns=20)
    result = brute_solve_bhm(params, obs_spec, errors, chip, Ns=20)

    print("Results")
    result.params.pretty_print()

    print("Chi2", result.chisqr)
    print("Reduced Chi2", result.redchi)

    print("Fit report", fit_report(result.params))


def brute_solve_bhm(params, obs_spec, errors, chip, Ns=20):
    kws = {"chip": chip, "norm": True, "norm_method": "linear",
           "area_scale": True, "wav_scale": True, "fudge": None}

    # Least-squares fit to the spectrum.
    mini = Minimizer(bhm_func_array, params, fcn_args=(obs_spec.xaxis, obs_spec.flux, errors), fcn_kws=kws)
    # Evaluate 20 points on each axis and keep all points candidates
    result = mini.minimize(method="brute", Ns=Ns, keep="all")

    return result


def bhm_func_array(pars, obs_wav, obs_flux, errors, chip=None, norm=True, norm_method="scalar",
                   wav_scale=True, fudge=None):
    """Calculate binary model chi^2 for given parameters and observation"""
    # unpack parameters: extract .value attribute for each parameter
    parvals = pars.valuesdict()
    teff_1 = round(parvals['teff_1'] / 100) * 100
    # teff_2 = round(parvals['teff_2'] / 100) * 100
    logg_1 = round(parvals['logg_1'] * 2) / 2
    # logg_2 = round(parvals['logg_2'] * 2) / 2
    feh_1 = round(parvals['feh_1'] * 2) / 2
    # feh_2 = round(parvals['feh_2'] * 2) / 2
    rv_1 = np.asarray([parvals['rv_1']])
    rv_2 = np.asarray([parvals['rv_2']])
    arb_norm = parvals.get("arb_norm", 1)  # 1 if not provided

    obs_flux2, bhm_flux = bhm_magic_sauce(Spectrum(xaxis=obs_wav, flux=obs_flux),
                                          [teff_1, logg_1, feh_1],
                                          rv_1, rv_2,
                                          chip=chip, norm_method=norm_method,
                                          norm=norm,
                                          wav_scale=wav_scale, fudge=fudge)
    residual = bhm_flux - obs_flux2
    # Scale to make chi-square sensible.
    return residual / errors


def func_all_chips_array(pars, obs_wav, obs_flux, errors, norm=True, norm_method="scalar",
                         area_scale=True, wav_scale=True, fudge=None, arb_norm=False):
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

    # Detector 1
    obs_flux2, iam_flux = iam_magic_sauce(Spectrum(xaxis=obs_wav, flux=obs_flux),
                                          [teff_1, logg_1, feh_1],
                                          [teff_2, logg_2, feh_2],
                                          rv_1, rv_2,
                                          chip=chip, norm_method=norm_method,
                                          area_scale=area_scale, norm=norm,
                                          wav_scale=wav_scale, fudge=fudge)
    residual = iam_flux - obs_flux2
    # Scale to make chi-square sensible.
    return residual / errors


if __name__ == "__main__":
    star = "HD211847"
    obsnum = 1
    chip = 1
    main(star, obsnum, chip)

    print("Done")
