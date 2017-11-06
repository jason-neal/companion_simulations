#!/usr/bin/env python
# Make min chi_2 spectral model

import argparse
import sys

import matplotlib.pyplot as plt
from spectrum_overload import Spectrum

from models.broadcasted_models import (independent_inherent_alpha_model,
                                       inherent_alpha_model)
from simulators.iam_module import iam_helper_function
from utilities.errors import spectrum_error
from utilities.masking import spectrum_masking
from utilities.phoenix_utils import load_starfish_spectrum
from utilities.spectrum_utils import load_spectrum


def _parser():
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Produce spectrum of results.')
    parser.add_argument('star', help='Star Name', type=str)
    parser.add_argument('obs_num', help="Observation label")
    parser.add_argument('teff_1', type=int,
                        help='Host Temperature')
    parser.add_argument('logg_1', type=float,
                        help='Host Tlogg')
    parser.add_argument('feh_1', type=float,
                        help='Host Fe/H')
    parser.add_argument('teff_2', type=int,
                        help='Companion Temperature')
    parser.add_argument('logg_2', type=float,
                        help='Companion logg')
    parser.add_argument('feh_2', type=float,
                        help='companion Fe/H')
    parser.add_argument('gamma', type=float,
                        help='Host rv')
    parser.add_argument("rv", type=float,
                        help='Companion rv')
    parser.add_argument("-i", "--independent", action="store_true",
                        help="Independent rv of companion")
    parser.add_argument("-p", "--plot_name", type=str,
                        help="Name of save figure.")
    return parser.parse_args()


def main(star, obs_num, teff_1, logg_1, feh_1, teff_2, logg_2, feh_2, gamma, rv, independent=False,
         plot_name=None):
    fig, axis = plt.subplots(2, 2, figsize=(15, 8), squeeze=False)

    for chip, ax in zip(range(1, 5), axis.flatten()):
        # Get observation data
        obs_name, params, output_prefix = iam_helper_function(star, obs_num, chip)

        # Load observed spectrum
        obs_spec = load_spectrum(obs_name)

        # Mask out bad portion of observed spectra
        obs_spec = spectrum_masking(obs_spec, star, obs_num, chip)

        # Barycentric correct spectrum
        # obs_spec = barycorr_crires_spectrum(obs_spec)

        error_off = False
        # Determine Spectrum Errors
        errors = spectrum_error(star, obs_num, chip, error_off=error_off)

        # Create model with given parameters
        host = load_starfish_spectrum([teff_1, logg_1, feh_1],
                                      limits=[2110, 2165], area_scale=True, hdr=True)
        companion = load_starfish_spectrum([teff_2, logg_2, feh_2],
                                           limits=[2110, 2165], area_scale=True, hdr=True)

        if independent:
            joint_model = independent_inherent_alpha_model(host.wav, host.flux,
                                                           companion.flux, gamma,
                                                           rv)
        else:
            joint_model = inherent_alpha_model(host.xaxis, host.flux,
                                               companion.flux, gamma, rv)

        model_spec = Spectrum(xaxis=host.xaxis, flux=joint_model(host.xaxis).squeeze())
        model_spec = model_spec.remove_nans()
        model_spec = model_spec.normalize("exponential")

        # plot
        obs_spec.plot(axis=ax, label="{}-{}".format(star, obs_num))
        model_spec.plot(axis=ax, label="Chi-squared model")
        # ax.plot(model_spec.xaxis, model_spec.flux, label="Mixed model")
        ax.set_xlim([obs_spec.xmin() - 0.5, obs_spec.xmax() + 0.5])

        ax.set_title("{} obs {} chip {}".format(star, obs_num, chip))
        ax.legend()
    plt.tight_layout()

    if plot_name is None:
        fig.show()
    else:
        fig.savefig(plot_name)

    return 0


if __name__ == "__main__":
    args = vars(_parser())
    opts = {k: args[k] for k in args}

    sys.exit(main(**opts))
    sys.exit(main(star, obs_num, teff_1, logg_1, feh_1, teff_2,
                  logg_2, feh_2, gamma, rv, **opts))
