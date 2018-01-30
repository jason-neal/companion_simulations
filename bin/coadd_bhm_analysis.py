#!/usr/bin/env python
"""Script to load part files into a sql database.

For the inherent alpha model

We assume that the host temperature is well known so we will restrict the
results to the host temperature. (within error bars)
For HD30501 and HD211847 this means  +- 50K so a fixed temperature.

"""
import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt

import simulators
from bin.coadd_analysis_script import decompose_database_name, load_sql_table, get_npix_values
from bin.coadd_bhm_analysis_module import (chi2_parabola_plots, chi2_individual_parabola_plots,
                                           compare_spectra, contours,
                                           contrast_bhm_results,
                                           display_bhm_xcorr_values,
                                           gamma_plot, get_column_limits,
                                           host_parameters,
                                           host_parameters_reduced_gamma,
                                           parabola_plots,
                                           smallest_chi2_values, test_figure)
from mingle.utilities.param_file import get_host_params
from mingle.utilities.phoenix_utils import closest_model_params


def parse_args(args):
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Chisquare analysis.')
    parser.add_argument('star', help='Star Name')
    parser.add_argument('obsnum', help="Observation label")
    parser.add_argument('-s', '--suffix', default=None,
                        help='Suffix to add to database name.')
    parser.add_argument("-e", "--echo", action="store_true",
                        help="Echo the SQL queries")
    parser.add_argument('-v', '--verbose', action="store_true",
                        help='Turn on Verbose.')
    parser.add_argument('-p', '--npars', type=int, default=3,
                        help='Number of interesting parameters. (default=3)')
    parser.add_argument("-m", "--mode", default="parabola",
                        help="Analysis mode to choose",
                        choices=["parabola", "fixed_host_params", "param_limits",
                                 "smallest_chi2", "test", "contour",
                                 "all", "rvplot", "chi2_parabola", "compare_spectra",
                                 "contrast", "xcorr"])
    return parser.parse_args(args)


def main(star, obsnum, suffix=None, echo=False, mode="parabola",
         verbose=False, npars=3):
    star = star.upper()
    suffix = "" if suffix is None else suffix

    database = os.path.join(
        simulators.paths["output_dir"], star, "bhm",
        "{0}-{1}_coadd_bhm_chisqr_results{2}.db".format(star, obsnum, suffix))

    if verbose:
        print("Database name ", database)
        print("Database exists", os.path.isfile(database))
    if not os.path.isfile(database):
        raise IOError("Database '{0}' does not exist.".format(database))

    path, dbstar, db_obsnum, chip = decompose_database_name(database)
    assert dbstar == star, "{} == {}".format(dbstar, star)
    assert str(db_obsnum) == str(obsnum), "{} == {}".format(db_obsnum, obsnum)
    assert chip == "coadd", "{} == {}".format(chip, "coadd")

    os.makedirs(os.path.join(path, "plots"), exist_ok=True)  # make dir for plots

    teff, logg, fe_h = closest_model_params(*get_host_params(star))
    params = {"path": path, "star": star, "obsnum": obsnum, "chip": chip, "suffix": suffix,
              "teff": int(teff), "logg": float(logg), "fe_h": float(fe_h), "npars": npars}

    db_table = load_sql_table(database, verbose=verbose, echo=echo)

    # Put pixel counts in params
    params["npix"] = get_npix_values(db_table)

    if verbose:
        print("Mode =", mode)

    if mode == "fixed_host_params":
        host_parameters_reduced_gamma(db_table, params)
        host_parameters(db_table, params)
    elif mode == "param_limits":
        get_column_limits(db_table, params)
    elif mode == "parabola":
        parabola_plots(db_table, params)
    elif mode == "smallest_chi2":
        smallest_chi2_values(db_table, params)
    elif mode == "contour":
        contours(db_table, params)
    elif mode == "test":
        test_figure(db_table, params)
    elif mode == "rvplot":
        gamma_plot(db_table, params)
    elif mode == "xcorr":
        display_bhm_xcorr_values(db_table, params)
    elif mode == "chi2_parabola":
        chi2_parabola_plots(db_table, params)
        chi2_individual_parabola_plots(db_table, params)
    elif mode == "compare_spectra":
        compare_spectra(db_table, params)
    elif mode == "contrast":
        contrast_bhm_results(db_table, params)
    elif mode == "all":
        try:
            host_parameters_reduced_gamma(db_table, params)
        except:
            pass
        plt.close("all")
        try:
            get_column_limits(db_table, params)
        except:
            pass
        plt.close("all")
        try:
            host_parameters(db_table, params)
        except:
            pass
        plt.close("all")
        try:
            smallest_chi2_values(db_table, params)
        except:
            pass
        plt.close("all")
        try:
            gamma_plot(db_table, params)
        except:
            pass
        plt.close("all")
        try:
            parabola_plots(db_table, params)
        except:
            pass
        plt.close("all")
        try:
            contours(db_table, params)
        except:
            pass
        plt.close("all")
        try:
            test_figure(db_table, params)
        except:
            pass
        plt.close("all")
        try:
            chi2_parabola_plots(db_table, params)
        except:
            pass
        plt.close("all")
        try:
            chi2_individual_parabola_plots(db_table, params)
        except:
            pass
        plt.close("all")
        try:
            compare_spectra(db_table, params)
        except:
            pass
        plt.close("all")
        try:
            contrast_bhm_results(db_table, params)
        except:
            pass
        plt.close("all")
        try:
            display_bhm_xcorr_values(db_table, params)
        except:
            pass
        plt.close("all")
    else:
        warnings.warn("Incorrect Mode in bhm analysis")
    plt.close("all")
    print("Done")
    return 0


if __name__ == '__main__':
    args = vars(parse_args(sys.argv[1:]))
    opts = {k: args[k] for k in args}

    sys.exit(main(**opts))
