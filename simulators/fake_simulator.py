import argparse
import logging
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import simulators
from astropy.io import fits
from mingle.models.broadcasted_models import inherent_alpha_model, independent_inherent_alpha_model
from mingle.utilities.norm import continuum
from mingle.utilities.simulation_utilities import spec_max_delta
from simulators.common_setup import obs_name_template
from simulators.iam_module import prepare_iam_model_spectra
from spectrum_overload import Spectrum

from mingle.models.broadcasted_models import one_comp_model
from mingle.utilities.phoenix_utils import load_starfish_spectrum
from mingle.utilities.simulation_utilities import add_noise


def parse_args(args):
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Fake observation simulator.')
    parser.add_argument("star", help='Star name.', type=str)
    parser.add_argument("sim_num", help='Star observation number.', type=str)
    parser.add_argument("-p", '--params1', help='Host parameters. "teff, logg, feh"', type=str)
    parser.add_argument("-q", '--params2', help='Companion parameters. "teff, logg, feh"', type=str)
    parser.add_argument("-g", "--gamma", help='RV of host.', type=float)
    parser.add_argument('-v', "--rv", help='RV of Companion.', type=float)
    parser.add_argument("-i", "--independent", help='Independent rv value."', action="store_true")
    parser.add_argument('-s', '--noise',
                        help='SNR value. int', type=float, default=None)
    parser.add_argument('-r', '--replace',
                        help='Replace old fake spectra.', action="store_true")
    parser.add_argument('-n', '--noplots',
                        help='Turn plots off.', action="store_true")
    parser.add_argument('-t', '--test',
                        help='Run testing only.', action="store_true")
    parser.add_argument('--suffix', help='Suffix for file.', type=str)
    parser.add_argument("-m", "--mode", help="Combination mode", choices=["tcm", "bhm", "iam"],
                        default="iam")
    parser.add_argument("-f", "--fudge", help="Fudge value to add", default=None)
    return parser.parse_args(args)


def fake_iam_simulation(wav, params1, params2, gamma, rv, limits=(2070, 2180),
                        independent=False, noise=None, header=False, fudge=None, area_scale=True):
    """Make a fake spectrum with binary params and radial velocities."""
    mod1_spec, mod2_spec = prepare_iam_model_spectra(params1, params2, limits, area_scale=area_scale)

    if fudge is not None:
        mod2_spec.flux = mod2_spec.flux * fudge
        warnings.warn("Fudging fake companion by '*{0}'".format(fudge))
    # Combine model spectra with iam model
    if independent:
        iam_grid_func = independent_inherent_alpha_model(mod1_spec.xaxis, mod1_spec.flux, mod2_spec.flux,
                                                         rvs=rv, gammas=gamma)
    else:
        iam_grid_func = inherent_alpha_model(mod1_spec.xaxis, mod1_spec.flux, mod2_spec.flux,
                                             rvs=rv, gammas=gamma)
    if wav is None:
        delta = spec_max_delta(mod1_spec, rv, gamma)
        assert np.all(np.isfinite(mod1_spec.xaxis))
        mask = (mod1_spec.xaxis > mod1_spec.xaxis[0] + 2 * delta) * (mod1_spec.xaxis < mod1_spec.xaxis[-1] - 2 * delta)
        wav = mod1_spec.xaxis[mask]

    iam_grid_models = iam_grid_func(wav).squeeze()
    logging.debug("iam_grid_func(wav).squeeze()", iam_grid_models)
    logging.debug("number of nans", np.sum(~np.isfinite(iam_grid_models)))
    logging.debug("iam_grid_models", iam_grid_models)

    logging.debug("Continuum normalizing")

    # Continuum normalize all iam_gird_models
    def axis_continuum(flux):
        """Continuum to apply along axis with predefined variables parameters."""
        return continuum(wav, flux, splits=50, method="exponential", top=5)

    iam_grid_continuum = np.apply_along_axis(axis_continuum, 0, iam_grid_models)

    iam_grid_models = iam_grid_models / iam_grid_continuum

    # This noise is added after continuum normalization.
    if noise is not None or noise is not 0:
        # Add 1 / snr noise to continuum normalized spectra
        iam_grid_models = add_noise(iam_grid_models, noise, use_mu=False)
    else:
        logging.warning("\n!!!\n\nNot adding any noise to fake simulation!!\n\n!!!!!\n")
        print("\n!!!\n\nNot adding any noise to fake simulation!!\n\n!!!!!\n")

    if header:
        return wav, iam_grid_models.squeeze(), mod1_spec.header
    else:
        return wav, iam_grid_models.squeeze()


def fake_bhm_simulation(wav, params, gamma, limits=(2070, 2180), noise=None, header=False):
    """Make a fake spectrum with binary params and radial velocities."""

    mod_spec = load_starfish_spectrum(params, limits=limits, hdr=True,
                                      normalize=True, wav_scale=True)

    bhm_grid_func = one_comp_model(mod_spec.xaxis, mod_spec.flux, gammas=gamma)

    if wav is None:
        delta = spec_max_delta(mod_spec, 0, gamma)
        assert np.all(np.isfinite(mod_spec.xaxis))
        mask = (mod_spec.xaxis > mod_spec.xaxis[0] + 2 * delta) * (mod_spec.xaxis < mod_spec.xaxis[-1] - 2 * delta)
        wav = mod_spec.xaxis[mask]

    bhm_grid_values = bhm_grid_func(wav).squeeze()

    logging.debug("number of bhm nans", np.sum(~np.isfinite(bhm_grid_values)))

    if noise is not None or noise is not 0:
        bhm_grid_values = add_noise(bhm_grid_values, noise)
    else:
        logging.warning("\n!!!\n\nNot adding any noise to bhm fake simulation!!!!!\n\n!!!!!\n")
        print("\n!!!!\n\nNot adding any noise to fake simulation!!!!\n\n!\n")

    if header:
        return wav, bhm_grid_values.squeeze(), mod_spec.header
    else:
        return wav, bhm_grid_values.squeeze()


def main(star, sim_num, params1=None, params2=None, gamma=None, rv=None,
         independent=False, noise=None, test=False, replace=False,
         noplots=False, mode="iam", fudge=None, area_scale=True, suffix=""):
    star = star.upper()

    if gamma is None:
        gamma = 0
    if rv is None:
        rv = 0

    if params1 is not None:
        params_1 = [float(par) for par in params1.split(",")]
    else:
        raise ValueError("No host parameter given. Use '-p'")

    if mode == "iam":
        if params2 is not None:
            params_2 = [float(par) for par in params2.split(",")]
        else:
            raise ValueError("No companion parameter given. Use '-q', or set '--mode bhm'")

        if test:
            testing_noise(star, sim_num, params_1, params_2, gamma, rv,
                          independent=False)
            testing_fake_spectrum(star, sim_num, params_1, params_2, gamma, rv,
                                  independent=False, noise=None)
        else:
            x_wav, y_wav, header = fake_iam_simulation(None, params_1, params_2, gamma=gamma, rv=rv,
                                                       independent=independent, noise=noise,
                                                       header=True, fudge=fudge, area_scale=area_scale)
            fake_spec = Spectrum(xaxis=x_wav, flux=y_wav, header=header)

            # save to file
            save_fake_observation(fake_spec, star, sim_num, params1, params2=params2, gamma=gamma, rv=rv,
                                  independent=False, noise=None, replace=replace, noplots=noplots)
    elif mode == "bhm":
        # Do a bhm simulation
        x_wav, y_wav, header = fake_bhm_simulation(None, params_1, gamma, noise=noise, header=True)

        fake_spec = Spectrum(xaxis=x_wav, flux=y_wav, header=header)

        # save to file
        save_fake_observation(fake_spec, star, sim_num, params1, gamma=gamma,
                              noise=None, replace=replace, noplots=noplots)

    return None

def save_fake_observation(spectrum, star, sim_num, params1, params2=None, gamma=None, rv=None,
                          independent=False, noise=None, suffix=None, replace=False, noplots=False):
    # Detector limits
    detector_limits = [(2112, 2123), (2127, 2137), (2141, 2151), (2155, 2165)]
    npix = 1024

    header = fits.Header.fromkeys({})
    for ii, detector in enumerate(detector_limits):
        spec = spectrum.copy()
        spec.wav_select(*detector)
        spec.interpolate1d_to(np.linspace(spec.xaxis[0], spec.xaxis[-1], npix))

        if not noplots:
            plt.plot(spec.xaxis, spec.flux)
            plt.title("Fake spectrum {0} {1} detector {2}".format(star, sim_num, ii + 1))
            plt.show()
        name = obs_name_template().format(star, sim_num, ii + 1)
        # name = "{0}-{1}-mixavg-tellcorr_{2}.fits".format(star, sim_num, ii + 1)
        name = os.path.join(simulators.paths["spectra"], name)
        # spec.save...
        hdrkeys = ["OBJECT", "Id_sim", "num", "chip", "snr", "ind_rv", "c_gamma", "cor_rv", "host", "compan"]
        hdrvals = [star, "Fake simulation data", sim_num, ii + 1, noise, independent, gamma, rv, params1, params2]
        if os.path.exists(name) and not replace:
            print(name, "Already exists")
        else:
            if os.path.exists(name):
                print("Replacing {0}".format(name))
                os.remove(name)
            export_fits(name, spec.xaxis, spec.flux, header, hdrkeys, hdrvals)
            print("Saved fits to {0}".format(name))


def testing_noise(star, sim_num, params1, params2, gamma, rv,
                  independent=False):
    x_wav, y_wav = fake_iam_simulation(None, params1, params2, gamma, rv, limits=[2070, 2180],
                                       independent=independent, noise=None)

    x_wav_1000, y_wav_1000 = fake_iam_simulation(None, params1, params2, gamma, rv, limits=[2070, 2180],
                                                 independent=independent, noise=1000)

    x_wav_200, y_wav_200 = fake_iam_simulation(None, params1, params2, gamma, rv, limits=[2070, 2180],
                                               independent=independent, noise=200)

    fig, axis = plt.subplots(2, 1, sharex=True)
    ax1 = axis[0]
    ax1.plot(x_wav_200, y_wav_200, label="snr =200")
    ax1.plot(x_wav_1000, y_wav_1000, label="snr =1000")
    ax1.plot(x_wav, y_wav, label="None")
    ax1.set_title("{0} simnum={1}, noise test\n host={2}, companion={3}".format(star, sim_num, params1, params2))
    ax1.legend()

    ax2 = axis[1]
    ax2.plot(x_wav_200, y_wav_200 - y_wav, label="snr=200")
    ax2.plot(x_wav_1000, y_wav_1000 - y_wav, label="snr=1000")
    ax2.set_title("Difference from no noise model")
    ax2.legend()
    plt.tight_layout()
    plt.show()


def testing_fake_spectrum(star, sim_num, params1, params2, gamma, rv,
                          independent=False, noise=None):
    x_wav, y_wav = fake_iam_simulation(None, params1, params2, gamma, rv,
                                       independent=independent, noise=noise)

    x_2k, y_2k = fake_iam_simulation(np.linspace(2100, 2140, 2000), params1,
                                     params2, gamma, rv, independent=independent, noise=noise)

    x_1k, y_1k = fake_iam_simulation(np.linspace(2090, 2150, 1000), params1, params2, gamma, rv,
                                     independent=independent, noise=noise)

    x_30k, y_30k = fake_iam_simulation(np.linspace(2090, 2150, 30000), params1, params2, gamma, rv,
                                       independent=independent, noise=noise)

    x_5k, y_5k = fake_iam_simulation(np.linspace(2090, 2150, 5000), params1, params2, gamma, rv,
                                     independent=independent, noise=noise)

    print("x", x_wav)
    print("y", y_wav)
    print("y_1k", y_1k)

    plt.plot(x_wav, y_wav, "b*", label="org Fake simulation")
    plt.plot(x_2k, y_2k, ".", label="2k")
    plt.plot(x_1k, y_1k, "s", label="1k")
    plt.plot(x_5k, y_5k, "o", label="5k")
    plt.plot(x_30k, y_30k, "h", label="30k")

    plt.xlim([2070, 2170])
    plt.legend()
    plt.title("{0} simnum={1}, noise={2}\n host={3}, companion={4}".format(star, sim_num, noise, params1, params2))
    plt.legend()
    plt.show()

    # NEED to normalize at full wavelength and then re-sample
    y_1k_reinterp = np.interp(x_2k, x_1k, y_1k)
    y_wav_reinterp = np.interp(x_2k, x_wav, y_wav)
    y_5k_reinterp = np.interp(x_2k, x_5k, y_5k)
    y_30k_reinterp = np.interp(x_2k, x_30k, y_30k)

    plt.plot(x_2k, y_2k, ".-", label="2k")
    plt.plot(x_2k, y_wav_reinterp, ".-", label="org sampling.")
    plt.plot(x_2k, y_5k_reinterp, ".-", label="5k")
    plt.plot(x_2k, y_1k_reinterp, ".-", label="1k.")
    plt.plot(x_2k, y_30k_reinterp, ".-", label="30k.")
    plt.title("Accessing re-normalization (inter to 2k")
    plt.legend()
    plt.show()

    plt.plot(y_wav_reinterp - y_2k, label="org diff")
    plt.plot(y_1k_reinterp - y_2k, label="1k diff")
    plt.plot(y_5k_reinterp - y_2k, label="5k diff")
    plt.plot(y_30k_reinterp - y_2k, label="30k diff")
    plt.title("Difference to 2k interp")
    plt.legend()
    plt.show()


def export_fits(filename, wavelength, flux, hdr, hdrkeys, hdrvals):
    """Write Telluric Corrected spectra to a fits table file."""
    col1 = fits.Column(name="wavelength", format="E", array=wavelength)  # colums of data
    col2 = fits.Column(name="flux", format="E", array=flux)
    cols = fits.ColDefs([col1, col2])

    tbhdu = fits.BinTableHDU.from_columns(cols)  # binary table hdu
    prihdr = append_hdr(hdr, hdrkeys, hdrvals)
    prihdu = fits.PrimaryHDU(header=prihdr)
    thdulist = fits.HDUList([prihdu, tbhdu])

    thdulist.writeto(filename, output_verify="silentfix")  # Fixing errors to work properly
    return None


def append_hdr(hdr, keys=None, values=None, item=0):
    """Append/change parameters to fits hdr.

    Can take list or tuple as input of keywords
    and values to change in the header
    Defaults at changing the header in the 0th item
    unless the number the index is given,
    If a key is not found it adds it to the header.
    """
    if keys is not None and values is not None:
        if isinstance(keys, str):  # To handle single value
            hdr[keys] = values
        else:
            assert len(keys) == len(values), 'Not the same number of keys as values'
            for i, key in enumerate(keys):
                hdr[key] = values[i]
                # print(repr(hdr[-2:10]))
    return hdr


if __name__ == "__main__":
    args = vars(parse_args(sys.argv[1:]))
    opts = {k: args[k] for k in args}

    # with warnings.catch_warnings():
    #    warnings.filterwarnings('error')
    main(**opts)
