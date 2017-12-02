#!/usr/bin/env python
# Make min chi_2 spectral model

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from spectrum_overload import Spectrum

from mingle.models.broadcasted_models import (independent_inherent_alpha_model,
                                              inherent_alpha_model)
from mingle.utilities.crires_utilities import barycorr_crires_spectrum
from mingle.utilities.errors import spectrum_error
from mingle.utilities.masking import spectrum_masking
from mingle.utilities.phoenix_utils import load_starfish_spectrum
from mingle.utilities.spectrum_utils import load_spectrum
from simulators.iam_module import iam_helper_function


def parse_args(args):
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Produce spectrum of results.')
    parser.add_argument('star', help='Star Name', type=str)
    parser.add_argument('obsnum', help="Observation label")
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
    return parser.parse_args(args)


def main(star, obsnum, teff_1, logg_1, feh_1, teff_2, logg_2, feh_2, gamma, rv, independent=False,
         plot_name=None):
    fig, axis = plt.subplots(2, 2, figsize=(15, 8), squeeze=False)

    for chip, ax in zip(range(1, 5), axis.flatten()):
        # Get observation data
        obs_name, params, output_prefix = iam_helper_function(star, obsnum, chip)

        # Load observed spectrum
        obs_spec = load_spectrum(obs_name)

        # Mask out bad portion of observed spectra
        obs_spec = spectrum_masking(obs_spec, star, obsnum, chip)

        # Barycentric correct spectrum
        _obs_spec = barycorr_crires_spectrum(obs_spec, extra_offset=None)

        error_off = False
        # Determine Spectrum Errors
        errors = spectrum_error(star, obsnum, chip, error_off=error_off)

        # Create model with given parameters
        host = load_starfish_spectrum([teff_1, logg_1, feh_1],
                                      limits=[2110, 2165], area_scale=True, hdr=True)
        if teff_2 is None:
            assert (logg_2 is None) and (feh_2 is None) and (rv == 0), "All must be None for bhm case."
            companion = Spectrum(xaxis=host.xaxis, flux=np.zeros_like(host.flux))
        else:
            companion = load_starfish_spectrum([teff_2, logg_2, feh_2],
                                               limits=[2110, 2165], area_scale=True, hdr=True)

        if independent:
            joint_model = independent_inherent_alpha_model(host.xaxis, host.flux,
                                                           companion.flux, gammas=gamma,
                                                           rvs=rv)
        else:
            joint_model = inherent_alpha_model(host.xaxis, host.flux,
                                               companion.flux, gammas=gamma, rvs=rv)

        model_spec = Spectrum(xaxis=host.xaxis, flux=joint_model(host.xaxis).squeeze())
        model_spec = model_spec.remove_nans()
        model_spec = model_spec.normalize("exponential")

        # plot
        obs_spec.plot(axis=ax, label="{}-{}".format(star, obsnum))
        model_spec.plot(axis=ax, linestyle="--", label="Chi-squared model")
        # ax.plot(model_spec.xaxis, model_spec.flux, label="Mixed model")
        ax.set_xlim([obs_spec.xmin() - 0.5, obs_spec.xmax() + 0.5])

        ax.set_title("{} obs {} chip {}".format(star, obsnum, chip))
        ax.legend()
    plt.tight_layout()

    if plot_name is None:
        fig.show()
    else:
        if not (plot_name.endswith(".png") or plot_name.endswith(".pdf")):
            raise ValueError("plot_name does not end with .pdf or .png")
        fig.savefig(plot_name)

    return 0


if __name__ == "__main__":
    args = vars(parse_args(sys.argv[1:]))
    opts = {k: args[k] for k in args}

    sys.exit(main(**opts))
    sys.exit(main(star, obsnum, teff_1, logg_1, feh_1, teff_2,
                  logg_2, feh_2, gamma, rv, **opts))
