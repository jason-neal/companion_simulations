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
from mingle.utilities.phoenix_utils import (closest_model_params)
from mingle.utilities.spectrum_utils import load_spectrum
from simulators.iam_module import iam_chi2_magic_sauce, iam_magic_sauce
from simulators.minimize_bhm import brute_solve_bhm
from simulators.iam_module import (iam_helper_function,
                                   setup_iam_dirs, target_params)

logging.basicConfig(level=logging.WARNING,
                    format='%(levelname)s %(message)s')


def load_observation(star, obsnum, chip):
    obs_name, params, output_prefix = iam_helper_function(star, obsnum, chip)

    print("The observation used is ", obs_name, "\n")

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
    return obs_spec, errors, params


@timeit2
def main(star, obsnum, chip):
    star = star.upper()
    setup_iam_dirs(star)

    # Setup comparision spectra
    obs_spec, errors, obs_params = load_observation(star, obsnum, chip)

    host_params, comp_params = target_params(obs_params, mode="iam")

    closest_host_model = closest_model_params(*host_params)
    closest_comp_model = closest_model_params(*comp_params)

    params = Parameters()

    params.add('teff_1', value=closest_host_model[0], min=5600, max=5800, vary=False, brute_step=100)
    params.add('teff_2', value=closest_comp_model[0], min=3000, max=3400, vary=True, brute_step=100)
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
        obs_spec, errors, obs_params = load_observation(star, obsnum, chip)
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
           "area_scale": True, "wav_scale": True, "fudge": None, "arb_norm": False}

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
        assert obs_flux.shape == obs_wav.shape

        flux = np.empty()
        model = np.empty()
        error_array = np.empty()

        for ii, c in enumerate(chip):
            flux_ii, model_ii = iam_magic_sauce(Spectrum(xaxis=obs_wav[ii], flux=obs_flux[ii]),
                                                [teff_1, logg_1, feh_1],
                                                [teff_2, logg_2, feh_2],
                                                rv_1, rv_2,
                                                chip=c, norm_method=norm_method,
                                                area_scale=area_scale, norm=norm,
                                                wav_scale=wav_scale, fudge=fudge)
            flux = np.concatenate((flux, flux_ii))
            model = np.concatenate((model, model_ii))
            error_array = np.concatenate((error_array, errors[ii] * np.ones_like(flux_ii)))
        errors = error_array
    else:
        flux, model = iam_magic_sauce(Spectrum(xaxis=obs_wav, flux=obs_flux),
                                      [teff_1, logg_1, feh_1],
                                      [teff_2, logg_2, feh_2],
                                      rv_1, rv_2,
                                      chip=chip, norm_method=norm_method,
                                      area_scale=area_scale, norm=norm,
                                      wav_scale=wav_scale, fudge=fudge)

    residual = model - (flux * arb_norm)

    # Scale to make chi-square sensible.
    scaled_residual = residual / errors
    return scaled_residual


def func_all_chips_array(pars, obs_wav, obs_flux, errors, norm=True, norm_method="scalar",
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
