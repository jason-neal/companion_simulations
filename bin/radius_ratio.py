"""Script to calculate radius ratio and flux ratio between synthetic spectra."""
import argparse
import sys
import numpy as np


def parse_args(args):
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Compute ratios to between synthetic spectra.')
    parser.add_argument('teff_1', type=int, help='Host Temperature')

    parser.add_argument('logg_1', type=float, help='Host logg')
    parser.add_argument('feh_1', type=float, help='Host Fe/H')
    parser.add_argument('teff_2', type=int, help='Companion Temperature')
    parser.add_argument('logg_2', type=float, help='Companion logg')
    parser.add_argument('feh_2', type=float, help='companion Fe/H')
    parser.add_argument('-r', "--radius", action="store_true",
                        help='Compute radius ratio')
    parser.add_argument('-f', "--abs_flux", action="store_true",
                        help='Compute absolute flux ratio')
    parser.add_argument('-e', "--relative_flux", action="store_true",
                        help='Compute relative flux ratio')

    parser.add_argument("-p", "--plot_name", type=str,
                        help="Name of save figure.")
    return parser.parse_args(args)


def main(teff_1, logg_1, feh_1, teff_2, logg_2, feh_2, **kwargs):
    from simulators.iam_module import prepare_iam_model_spectra
    from mingle.utilities.phoenix_utils import phoenix_radius
    wav_limits = [2110, 2160]
    host, companion = prepare_iam_model_spectra([teff_1, logg_1, feh_1],
                                                [teff_2, logg_2, feh_2],
                                                limits=wav_limits, area_scale=True, wav_scale=True)

    print("Calculating with:")
    print("Host = [{0} K, {1}, {2}]".format(teff_1, logg_1, feh_1))
    print("Companion = [{0} K, {1}, {2}]".format(teff_2, logg_2, feh_2))
    print("Wavelength range {0} nm".format(wav_limits))
    print("--------------------------------------")

    r_host = phoenix_radius(host.header)
    r_companion = phoenix_radius(companion.header)
    r_ratio = r_host / r_companion
    flux_ratio = host.flux / companion.flux

    if kwargs.get("radius", False):
        print("Radius ratio          R_1/R_2 = {0: 7.06}".format(r_ratio))
    # emitting_area = phoenix_area(spec.header)

    if kwargs.get("abs_flux", False):
        mean_ratio = np.mean(flux_ratio)
        print("Absolute Flux ratio   F_1/F_2 = {0: 7.06}".format(mean_ratio))

    if kwargs.get("relative_flux", False):
        # correct for area_scaling by dividing by radius ratio squared
        # This is equivalent to loading models straight without modification
        mean_ratio = np.mean(flux_ratio / r_ratio ** 2)
        print("Relative  Flux ratio  f_1/f_2 = {0: 7.06}".format(mean_ratio))


if __name__ == "__main__":
    args = vars(parse_args(sys.argv[1:]))
    opts = {k: args[k] for k in args}
    main(**opts)
