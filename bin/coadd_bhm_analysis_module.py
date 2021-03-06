#!/usr/bin/env python
import logging
import os
import warnings

import numpy as np
from logutils import BraceMessage as __
from matplotlib import pyplot as plt
from matplotlib import rc
from scipy.optimize import newton
from spectrum_overload import Spectrum

from bin.coadd_analysis_module import fit_chi2_parabola, parabola
from mingle.models.broadcasted_models import one_comp_model
from mingle.utilities import chi2_at_sigma
from mingle.utilities.chisqr import reduced_chi_squared
from mingle.utilities.crires_utilities import barycorr_crires_spectrum
from mingle.utilities.db_utils import DBExtractor
from mingle.utilities.debug_utils import timeit2
from mingle.utilities.phoenix_utils import load_starfish_spectrum
from mingle.utilities.spectrum_utils import load_spectrum
from simulators.bhm_module import bhm_helper_function

rc("image", cmap="inferno")
chi2_names = ["chi2_1", "chi2_2", "chi2_3", "chi2_4", "coadd_chi2"]
npix_names = ["npix_1", "npix_2", "npix_3", "npix_4", "coadd_npix"]


def gamma_plot(table, params):
    extractor = DBExtractor(table)
    for chi2_val, npix_val in zip(chi2_names, npix_names):
        df = extractor.simple_extraction(["gamma", chi2_val, "teff_1"])
        fig, ax = plt.subplots()
        c = ax.scatter(df["gamma"], df[chi2_val], c=df["teff_1"], alpha=0.8)
        cbar = plt.colorbar(c)
        cbar.ax.set_ylabel(r"teff_1")
        ax.set_xlabel(r'Host RV offset', fontsize=12)
        ax.set_ylabel(r"${0}$".format(chi2_val), fontsize=12)
        ax.set_title(r'$teff_1$ (color) and companion temperature.')
        ax.grid(True)
        fig.tight_layout()
        name = "{0}-{1}_{2}_temp_gamma_plot_{3}_{4}.pdf".format(
            params["star"], params["obsnum"], params["chip"], chi2_val, params["suffix"])
        plt.savefig(os.path.join(params["path"], "plots", name))
        plt.savefig(os.path.join(params["path"], "plots", name.replace(".pdf", ".png")))
        plt.close()

        red_chi2 = "red_{0}".format(chi2_val)
        df[red_chi2] = reduced_chi_squared(df[chi2_val], params["npix"][npix_val], params["npars"])
        fig, ax = plt.subplots()
        c = ax.scatter(df["gamma"], df[red_chi2], c=df["teff_1"], alpha=0.8)
        cbar = plt.colorbar(c)
        cbar.ax.set_ylabel(r"teff_1")
        ax.set_xlabel(r'Host RV offset', fontsize=12)
        ax.set_ylabel(r"${0}$".format(red_chi2), fontsize=12)
        ax.set_title(r'$teff_1$ (color) and companion temperature.')
        ax.grid(True)
        fig.tight_layout()
        name = "{0}-{1}_{2}_temp_gamma_plot_{3}_{4}.pdf".format(
            params["star"], params["obsnum"], params["chip"], red_chi2, params["suffix"])
        plt.savefig(os.path.join(params["path"], "plots", name))
        plt.savefig(os.path.join(params["path"], "plots", name.replace(".pdf", ".png")))
        plt.close()


def xshift(x, num):
    """Shift x position slightly."""
    return x + num * (x * 0.1)


def display_bhm_xcorr_values(table, params):
    extractor = DBExtractor(table)
    fig, axarr = plt.subplots(3)
    for ii, xcorr in enumerate([r"xcorr_1", r"xcorr_2", r"xcorr_3", r"xcorr_4"]):
        df = extractor.simple_extraction(["gamma", xcorr, "teff_1"])

        c = axarr[0].scatter(xshift(df["gamma"], ii), df[xcorr],
                             c=df[r"teff_1"].values, alpha=0.9, label=xcorr)

        axarr[0].set_xlabel(r'host rv offset', fontsize=12)
        axarr[0].set_ylabel(r'Arbitrary norm', fontsize=12)
        axarr[0].set_title(r'Arbitrary normalization.')

        d = axarr[1].scatter(xshift(df[r"gamma"], ii), df[xcorr],
                             c=df[r"teff_1"].values, alpha=0.9, label=xcorr)
        axarr[1].set_xlabel(r'$\gamma$ offset', fontsize=12)
        axarr[1].set_ylabel(r'Arbitrary norm', fontsize=12)
        axarr[1].set_title(r'$\gamma$.')

        e = axarr[2].scatter(xshift(df[r"teff_1"], ii), df[xcorr],
                             c=df[r"gamma"].values, alpha=0.9, label=xcorr)
        axarr[2].set_xlabel(r'Companion temperature', fontsize=15)
        axarr[2].set_ylabel(r'Arbitrary norm ', fontsize=15)
        axarr[2].set_title(r'Companion Temperature')
    axarr[0].grid(True)
    axarr[1].grid(True)
    axarr[2].grid(True)

    cbar0 = plt.colorbar(c)
    cbar0.ax.set_ylabel(r" teff_1")
    cbar1 = plt.colorbar(d)
    cbar1.ax.set_ylabel(r" teff_1")
    cbar2 = plt.colorbar(e)
    cbar1.ax.set_ylabel(r"$\gamma$")
    fig.tight_layout()
    fig.suptitle("Arbitrary normalization used, \n slight shift with detetor")
    name = "{0}-{1}_{2}_plot_xcorr_value_{3}.pdf".format(
        params["star"], params["obsnum"], params["chip"], params["suffix"])
    plt.savefig(os.path.join(params["path"], "plots", name))
    plt.savefig(os.path.join(params["path"], "plots", name.replace(".pdf", ".png")))
    plt.close()


@timeit2
def host_parameters(table, params):
    extractor = DBExtractor(table)
    print("Fixed host analysis.")
    nrows, ncols = 3, 2
    columns = ["teff_1", "logg_1", "feh_1", "gamma"]
    assert len(columns) <= (nrows * ncols)

    for jj, (chi2_val, npix_val) in enumerate(zip(chi2_names, npix_names)):
        if jj == 4:
            chi2legend = "coadd chi2"
        else:
            chi2legend = "det {0}".format(jj + 1)
        fig, axes = plt.subplots(nrows, ncols)
        fig.tight_layout()
        indices = np.arange(nrows * ncols).reshape(nrows, ncols)
        for ii, col in enumerate(columns):
            df = extractor.fixed_extraction([col, chi2_val],
                                            fixed={"logg_1": params["logg"],
                                                   "feh_1": params["fe_h"]})

            axis_pos = [int(x) for x in np.where(indices == ii)]
            df.plot(x=col, y=chi2_val, kind="scatter",
                    ax=axes[axis_pos[0], axis_pos[1]], label=chi2legend)  # , c="gamma", colorbar=True)

        plt.suptitle("Co-add Chi**2 Results (fixed_logg_feh): {0}-{1}".format(
            params["star"], params["obsnum"]))
        name = "{0}-{1}_coadd_fixed_logg_feh_params_full_gamma_{2}_{3}.png".format(
            params["star"], params["obsnum"], params["suffix"], chi2_val)
        fig.savefig(os.path.join(params["path"], "plots", name))
        fig.savefig(os.path.join(params["path"], "plots", name.replace(".pdf", ".png")))
        plt.close()

    for jj, (chi2_val, npix_val) in enumerate(zip(chi2_names, npix_names)):
        if jj == 4:
            chi2legend = "coadd chi2"
        else:
            chi2legend = "det {0}".format(jj + 1)

        red_chi2 = "red_{0}".format(chi2_val)

        fig, axes = plt.subplots(nrows, ncols)
        fig.tight_layout()
        indices = np.arange(nrows * ncols).reshape(nrows, ncols)

        for ii, col in enumerate(columns):
            df = extractor.fixed_extraction([col, chi2_val],
                                            fixed={"logg_1": params["logg"],
                                                   "feh_1": params["fe_h"]})
            df[red_chi2] = reduced_chi_squared(df[chi2_val], params["npix"][npix_val], params["npars"])

            axis_pos = [int(x) for x in np.where(indices == ii)]
            df.plot(x=col, y=red_chi2, kind="scatter",
                    ax=axes[axis_pos[0], axis_pos[1]], label=chi2legend)  # , c="gamma", colorbar=True)

        plt.suptitle("Co-add reduced Chi**2 Results (fixed_logg_feh): {0}-{1}".format(
            params["star"], params["obsnum"]))
        name = "{0}-{1}_coadd_fixed_logg_feh_params_full_gamma_{2}_{3}.png".format(
            params["star"], params["obsnum"], params["suffix"], red_chi2)
        fig.savefig(os.path.join(params["path"], "plots", name))
        fig.savefig(os.path.join(params["path"], "plots", name.replace(".pdf", ".png")))
        plt.close()

    host_parameters_individual(table, params)


def host_parameters_individual(table, params):
    extractor = DBExtractor(table)
    nrows, ncols = 1, 1
    columns = ["teff_1", "logg_1", "feh_1", "gamma"]
    for ii, col in enumerate(columns):
        for jj, (chi2_val, npix_val) in enumerate(zip(chi2_names, npix_names)):
            if jj == 4:
                chi2legend = "coadd chi2"
            else:
                chi2legend = "det {0}".format(jj + 1)

            red_chi2 = "red_{0}".format(chi2_val)

            fig, axes = plt.subplots(nrows, ncols)
            fig.tight_layout()
            df = extractor.fixed_extraction([col, chi2_val],
                                            fixed={"logg_1": params["logg"],
                                                   "feh_1": params["fe_h"]})
            df[red_chi2] = reduced_chi_squared(df[chi2_val], params["npix"][npix_val], params["npars"])

            df.plot(x=col, y=red_chi2, kind="scatter",
                    ax=axes, label=chi2legend)
            name = "{0}-{1}_coadd_fixed_logg_feh_params_full_gamma_{2}_{3}_individual_{4}.png".format(
                params["star"], params["obsnum"], params["suffix"], chi2_val, col)
            plt.suptitle("Co-add {2}-Chi**2 Results (fixed_logg_feh): {0}-{1}: {3}".format(
                params["star"], params["obsnum"], chi2_val, col))
            fig.savefig(os.path.join(params["path"], "plots", name))
            fig.savefig(os.path.join(params["path"], "plots", name.replace(".pdf", ".png")))
            plt.close()
        plt.close()


def parabola_plots(table, params):
    extractor = DBExtractor(table)
    parabola_list = ["teff_1", "logg_1", "feh_1", "gamma"]
    for par in parabola_list:
        df = extractor.simple_extraction(columns=[par])
        unique_par = list(set(df[par].values))
        unique_par.sort()
        print("Unique ", par, " values =", unique_par)

        for chi2_val, npix_val in zip(chi2_names, npix_names):
            min_chi2 = []
            for unique_val in unique_par:
                df_chi2 = extractor.fixed_ordered_extraction(
                    [par, chi2_val], limit=3, fixed={par: float(unique_val)}, order_by=chi2_val)
                min_chi2.append(df_chi2[chi2_val].values[0])

            min_chi2 = reduced_chi_squared(min_chi2, params["npix"][npix_val], params["npars"])

            plt.plot(unique_par, min_chi2, ".-", label=chi2_val)

            # popt, _ = curve_fit(parabola, unique_par, min_chi2)
            popt, _ = fit_chi2_parabola(unique_par, min_chi2)

            x = np.linspace(unique_par[0], unique_par[-1], 40)
            plt.plot(x, parabola(x, *popt), "--")
            plt.xlabel(r"${0}$".format(par))
            plt.ylabel(r"$\Delta \chi^2_{red}$")

        plt.legend()
        filename = "red_Parabola_fit_{0}-{1}_{2}_param_{3}_{4}.png".format(
            params["star"], params["obsnum"], params["chip"], par, params["suffix"])

        plt.savefig(os.path.join(params["path"], "plots", filename))
        plt.close()
        print("saved parabolas for ", par)
    plt.close()


def chi2_parabola_plots(table, params):
    extractor = DBExtractor(table)
    parabola_list = ["teff_1", "logg_1", "feh_1", "gamma"]
    for par in parabola_list:
        df = extractor.simple_extraction(columns=[par])
        unique_par = list(set(df[par].values))
        unique_par.sort()

        for chi2_val, npix_val in zip(chi2_names, npix_names):
            min_chi2 = []
            for unique_val in unique_par:
                df_chi2 = extractor.fixed_ordered_extraction(
                    [par, chi2_val], limit=3, fixed={par: float(unique_val)}, order_by=chi2_val)
                min_chi2.append(df_chi2[chi2_val].values[0])

            # min_chi2 = reduced_chi_squared(min_chi2, params["npix"][npix_val], params["npars"])

            min_chi2 = min_chi2 - min(min_chi2)

            plt.plot(unique_par, min_chi2, ".-", label=chi2_val)

            # popt, _ = curve_fit(parabola, unique_par, min_chi2)
            popt, _ = fit_chi2_parabola(unique_par, min_chi2)
            # print("params", popt)
            x = np.linspace(unique_par[0], unique_par[-1], 40)
            plt.plot(x, parabola(x, *popt))  # , label="parabola")
            plt.xlabel(r"${0}$".format(par))
            plt.ylabel(r"$\Delta \chi^2$ from mimimum")
            plt.ylim([-0.05 * np.max(min_chi2), np.max(min_chi2)])
            # Find roots
            if chi2_val == "coadd_chi2":
                try:
                    warnings.warning("Doing npars sigma limits.")
                    residual = lambda x: parabola(x, *popt) - chi2_at_sigma(1, params["npars"])
                    min_chi2_par = unique_par[np.argmin(min_chi2)]
                    min_chi2_par.astype(np.float64)
                    try:
                        lower_bound = newton(residual, (min_chi2_par + unique_par[0]) / 2) - min_chi2_par
                    except RuntimeError as e:
                        print(e)
                        lower_bound = np.nan
                    try:
                        upper_bound = newton(residual, (min_chi2_par + unique_par[-1]) / 2) - min_chi2_par
                    except RuntimeError as e:
                        print(e)
                        upper_bound = np.nan

                    print("{0} solution {1: 5.3} {2:+5.3} {3:+5.3}".format(chi2_val, float(min_chi2_par),
                                                                           float(lower_bound),
                                                                           float(upper_bound)))
                    plt.annotate("{0: 5.3f} {1:+5.3f} {2:+5.3f}".format(float(min_chi2_par), float(lower_bound),
                                                                        float(upper_bound)),
                                 xy=(min_chi2_par, 0), xytext=(0.4, 0.5), textcoords="figure fraction",
                                 arrowprops={"arrowstyle": "->"})
                except Exception as e:
                    print(e)
                    logging.warning("Could not Annotate the contour plot")

        plt.axhline(y=chi2_at_sigma(1, params["npars"]), label="1 sigma {} par".format(params["npars"]))
        plt.axhline(y=chi2_at_sigma(2, params["npars"]), label="2 sigma {} par".format(params["npars"]))
        plt.axhline(y=chi2_at_sigma(3, params["npars"]), label="3 sigma {}par".format(params["npars"]))
        plt.axhline(y=chi2_at_sigma(1, dof=1), label="1 sigma 1 par", color="k", ls="--")
        plt.axhline(y=chi2_at_sigma(2, dof=1), label="2 sigma 1 par", color="k", ls="--")
        plt.axhline(y=chi2_at_sigma(3, dof=1), label="3 sigma 1 par", color="k", ls="--")

        plt.legend()
        filename = "Chi2_Parabola_fit_{0}-{1}_{2}_param_{3}_{4}.png".format(
            params["star"], params["obsnum"], params["chip"], par, params["suffix"])

        plt.savefig(os.path.join(params["path"], "plots", filename))
        plt.close()
        print("saved chi2 parabolas for ", par)
    plt.close()


def chi2_individual_parabola_plots(table, params):
    extractor = DBExtractor(table)
    parabola_list = ["teff_1", "logg_1", "feh_1", "gamma"]
    for par in parabola_list:
        df = extractor.simple_extraction(columns=[par])
        unique_par = list(set(df[par].values))
        unique_par.sort()

        for chi2_val, npix_val in zip(chi2_names, npix_names):
            min_chi2 = []
            for unique_val in unique_par:
                df_chi2 = extractor.fixed_ordered_extraction(columns=[par, chi2_val], order_by=chi2_val, limit=3,
                                                             fixed={par: float(unique_val)}, asc=True)
                min_chi2.append(df_chi2[chi2_val].values[0])

            # min_chi2 = reduced_chi_squared(min_chi2, params["npix"][npix_val], params["npars"])

            min_chi2 = min_chi2 - min(min_chi2)

            plt.plot(unique_par, min_chi2, ".-", label=chi2_val)

            # popt, _ = curve_fit(parabola, unique_par, min_chi2)
            popt, _ = fit_chi2_parabola(unique_par, min_chi2)
            # print("params", popt)
            x = np.linspace(unique_par[0], unique_par[-1], 40)
            plt.plot(x, parabola(x, *popt))  # , label="parabola")
            plt.xlabel(r"${0}$".format(par))
            plt.ylabel(r"$\Delta \chi^2$ from mimimum")
            plt.ylim([-0.05 * np.max(min_chi2), np.max(min_chi2)])
            # Find roots

            try:
                residual = lambda x: parabola(x, *popt) - chi2_at_sigma(1, params["npars"])
                min_chi2_par = unique_par[np.argmin(min_chi2)]
                min_chi2_par.astype(np.float64)
                try:
                    lower_bound = newton(residual, (min_chi2_par + unique_par[0]) / 2) - min_chi2_par
                except RuntimeError as e:
                    print(e)
                    lower_bound = np.nan
                try:
                    upper_bound = newton(residual, (min_chi2_par + unique_par[-1]) / 2) - min_chi2_par
                except RuntimeError as e:
                    print(e)
                    upper_bound = np.nan
                print("min_chi2_par", min_chi2_par, type(min_chi2_par), "\nlower_bound", lower_bound, type(lower_bound),
                      "\nupper_bound", upper_bound, type(upper_bound))
                print(
                    "{0} solution {1: 5.3} {2:+5.3} {3:+5.3}".format(chi2_val, float(min_chi2_par), float(lower_bound),
                                                                     float(upper_bound)))
                plt.annotate("{0: 5.3f} {1:+5.3f} {2:+5.3f}".format(float(min_chi2_par), (lower_bound), (upper_bound)),
                             xy=(float(min_chi2_par), 0), xytext=(0.4, 0.5), textcoords="figure fraction",
                             arrowprops={"arrowstyle": "->"})
            except Exception as e:
                print(e)
                logging.warning("Could not Annotate the contour plot")

            plt.axhline(y=chi2_at_sigma(1, params["npars"]), label="1 sigma")
            plt.axhline(y=chi2_at_sigma(2, params["npars"]), label="2 sigma")
            plt.axhline(y=chi2_at_sigma(3, params["npars"]), label="3 sigma")

            plt.legend()
            filename = "Chi2_Parabola_fit_{0}-{1}_{2}_param_{3}_{4}_individual_{5}.png".format(
                params["star"], params["obsnum"], params["chip"], par, params["suffix"], chi2_val)

            plt.savefig(os.path.join(params["path"], "plots", filename))
            plt.close()
            print("saved individual chi2 parabolas for ", par)
            plt.close()


def smallest_chi2_values(table, params, num=10):
    """Find smallest chi2 in table."""
    chi2_val = "chi2_1"  # "coadd_chi2"
    df = DBExtractor(table).ordered_extraction(order_by=chi2_val, limit=num, asc=True)
    df[chi2_val] = reduced_chi_squared(df[chi2_val], params["npix"]["coadd_npix"], params["npars"])

    df_min = df[:1]
    print("Smallest Co-add reduced Chi2 values in the database.")
    print(df.head(n=num))
    # name = "{0}-{1}_{2}_test_smallest_chi2_{3}.pdf".format(
    # params["star"], params["obsnum"], params["chip"], params["suffix"])
    name = "minimum_coadd_chi2_db_output_{0}_{1}_{2}.csv".format(params["star"], params["obsnum"], params["suffix"])
    from bin.check_result import main as visual_inspection
    df.to_csv(os.path.join(params["path"], name))

    plot_name = os.path.join(params["path"], "plots",
                             "visual_inspection_min_chi2_coadd_{0}_{1}_{2}.png".format(params["star"],
                                                                                       params["obsnum"],
                                                                                       params["suffix"]))
    visual_inspection(params["star"], params["obsnum"], float(df_min.teff_1), float(df_min.logg_1),
                      float(df_min.feh_1), None, None,
                      None, gamma=float(df_min.gamma), rv=0.0, plot_name=plot_name)


@timeit2
def host_parameters_reduced_gamma(table, params):
    print("Fixed host analysis with reduced gamma.")
    extractor = DBExtractor(table)
    d_gamma = 5

    nrows, ncols = 3, 2
    fig, axes = plt.subplots(nrows, ncols)
    fig.tight_layout()
    indices = np.arange(nrows * ncols).reshape(nrows, ncols)

    for jj, (chi2_val, npix_val) in enumerate(zip(chi2_names, npix_names)):
        if jj == 4:
            chi2legend = "coadd"
        else:
            chi2legend = "det {0}".format(jj + 1)

        # Select lowest chi square gamma values.
        df = extractor.ordered_extraction(order_by=chi2_val, columns=["gamma", chi2_val], limit=1, asc=True)

        min_chi2_gamma = df.loc[0, "gamma"]
        upper_lim = min_chi2_gamma + d_gamma
        lower_lim = min_chi2_gamma - d_gamma

        columns = ["teff_1", "logg_1", "feh_1", "gamma", ]
        assert len(columns) <= (nrows * ncols)

        for ii, col in enumerate(columns):
            df = extractor.simple_extraction(columns={col, chi2_val, "gamma", "teff_1"})
            df = df[[(df.gamma > float(lower_lim)) & (df.gamma < float(upper_lim))]]

            axis_pos = [int(x) for x in np.where(indices == ii)]
            df.plot(x=col, y=chi2_val, kind="scatter",
                    ax=axes[axis_pos[0], axis_pos[1]], label=chi2legend)

    plt.suptitle("Coadd Chi**2 Results: {0}-{1}".format(params["star"], params["obsnum"]))
    name = "{0}-{1}_coadd_fixed_host_params_{2}.png".format(
        params["star"], params["obsnum"], params["suffix"])
    fig.savefig(os.path.join(params["path"], "plots", name))
    fig.savefig(os.path.join(params["path"], "plots", name.replace(".pdf", ".png")))
    plt.close()

    for jj, (chi2_val, npix_val) in enumerate(zip(chi2_names, npix_names)):
        red_chi2 = "red_{0}".format(chi2_val)
        if jj == 4:
            chi2legend = "coadd"
        else:
            chi2legend = "det {0}".format(jj + 1)

        # Select lowest chi square gamma values.
        df = extractor.ordered_extraction(order_by=chi2_val, columns=["gamma", chi2_val], limit=1)

        min_chi2_gamma = df.loc[0, "gamma"]
        upper_lim = min_chi2_gamma + d_gamma
        lower_lim = min_chi2_gamma - d_gamma

        columns = ["teff_1", "logg_1", "feh_1", "gamma", ]
        assert len(columns) <= (nrows * ncols)

        for ii, col in enumerate(columns):
            df = extractor.simple_extraction(columns={col, chi2_val, "gamma", "teff_1"})
            df = df[[(df.gamma > float(lower_lim)) & (df.gamma < float(upper_lim))]]

            df[red_chi2] = reduced_chi_squared(df[chi2_val], params["npix"][npix_val], params["npars"])
            axis_pos = [int(x) for x in np.where(indices == ii)]

            df.plot(x=col, y=red_chi2, kind="scatter",
                    ax=axes[axis_pos[0], axis_pos[1]], label=chi2legend)

    plt.suptitle("Coadd reduced Chi**2 Results: {0}-{1}".format(params["star"], params["obsnum"]))
    name = "{0}-{1}_reduced_coadd_fixed_host_params_{2}.png".format(
        params["star"], params["obsnum"], params["suffix"])
    fig.savefig(os.path.join(params["path"], "plots", name))
    fig.savefig(os.path.join(params["path"], "plots", name.replace(".pdf", ".png")))
    plt.close()
    host_parameters_reduced_gamma_individual(table, params)


def host_parameters_reduced_gamma_individual(table, params):
    print("Fixed host analysis with reduced gamma individual plots.")
    extractor = DBExtractor(table)
    d_gamma = 5
    nrows, ncols = 1, 1

    for jj, (chi2_val, npix_val) in enumerate(zip(chi2_names, npix_names)):
        red_chi2 = "red_{0}".format(chi2_val)
        if jj == 4:
            chi2legend = "coadd"
        else:
            chi2legend = "det {0}".format(jj + 1)

        # Select lowest chi square gamma values.
        df = extractor.ordered_extraction(order_by=chi2_val, columns=["gamma", chi2_val], limit=1)

        min_chi2_gamma = df.loc[0, "gamma"]
        upper_lim = min_chi2_gamma + d_gamma
        lower_lim = min_chi2_gamma - d_gamma
        columns = ["teff_1", "logg_1", "feh_1", "gamma"]

        for ii, col in enumerate(columns):
            df = extractor.simple_extraction(columns={col, chi2_val, "gamma", "teff_1"})
            df = df[[(df.gamma > float(lower_lim)) & (df.gamma < float(upper_lim))]]

            df[red_chi2] = reduced_chi_squared(df[chi2_val], params["npix"][npix_val], params["npars"])
            fig, axes = plt.subplots(nrows, ncols)
            fig.tight_layout()
            df.plot(x=col, y=red_chi2, kind="scatter",
                    ax=axes, label=chi2legend)

            plt.suptitle("Coadd {2}-reduced Chi**2 Results: {0}-{1}".format(params["star"], params["obsnum"], col))
            name = "{0}-{1}_coadd_fixed_host_params_{2}_{3}_individual_{4}.png".format(
                params["star"], params["obsnum"], params["suffix"], chi2_val, col)
            fig.savefig(os.path.join(params["path"], "plots", name))
            fig.savefig(os.path.join(params["path"], "plots", name.replace(".pdf", ".png")))
            plt.close()
        plt.close()


def contours(table, params):
    extractor = DBExtractor(table)
    for chi2_val, npix_val in zip(chi2_names, npix_names):
        df_min_chi2 = extractor.minimum_value_of(chi2_val)

        fixed_params = {"logg_1": float(df_min_chi2["logg_1"].values[0]),
                        "feh_1": float(df_min_chi2["feh_1"].values[0])}
        df = extractor.fixed_extraction(columns=["teff_1", "feh_1", "logg_1", "gamma", chi2_val],
                                        fixed=fixed_params)
        params["this_npix"] = params["npix"][npix_val]
        params["chi2_value"] = chi2_val
        pars = ["gamma", "teff_1", chi2_val]
        dataframe_contour(df, xcol=pars[0], ycol=pars[1], zcol=pars[2], params=params)

    # Reduced chi2
    for chi2_val, npix_val in zip(chi2_names, npix_names):
        red_chi2 = "red_{0}".format(chi2_val)
        df_min_chi2 = extractor.minimum_value_of(chi2_val)

        fixed_params = {"logg_1": float(df_min_chi2["logg_1"].values[0]),
                        "feh_1": float(df_min_chi2["feh_1"].values[0])}
        df = extractor.fixed_extraction(columns=["teff_1", "feh_1", "logg_1", "gamma", chi2_val],
                                        fixed=fixed_params)
        df[red_chi2] = reduced_chi_squared(df[chi2_val], params["npix"][npix_val], params["npars"])
        params["this_npix"] = params["npix"][npix_val]
        params["chi2_value"] = chi2_val
        pars = ["gamma", "teff_1", red_chi2]
        dataframe_contour(df, xcol=pars[0], ycol=pars[1], zcol=pars[2], params=params)


def dataframe_contour(df, xcol, ycol, zcol, params):
    x = sorted(np.array(list(set(df[xcol].values))))
    y = sorted(np.array(list(set(df[ycol].values))))

    # Create grid for chi2 values
    z_grid = np.empty((len(x), len(y)))
    for i, x_value in enumerate(x):
        for j, y_value in enumerate(y):
            try:
                z_grid[i, j] = df.loc[(df[xcol].values == x_value) * (df[ycol].values == y_value), zcol].values
            except ValueError as e:
                print("x_S * y_s", sum((df[xcol].values == x_value) * (df[ycol].values == y_value)))
                print("Check metallicity and logg of companion")
                raise e

    x_grid, y_grid = np.meshgrid(x, y, indexing='ij')

    assert x_grid.shape == z_grid.shape
    assert x_grid.shape == y_grid.shape

    try:
        fig, ax = plt.subplots()
        c = ax.contourf(x_grid, y_grid, z_grid, alpha=0.5, cmap=plt.cm.inferno)
        # Mark minimum with a +.
        min_loci, min_locj = divmod(z_grid.argmin(), z_grid.shape[1])
        plt.plot(x[min_loci], y[min_locj], "g*", markersize=7)
        cbar = plt.colorbar(c)
        cbar.ax.set_ylabel(zcol)
        ax.set_xlabel(r"$ {0}$".format(xcol), fontsize=15)
        ax.set_ylabel(r"$ {0}$".format(ycol), fontsize=15)
        ax.set_title(
            '{0}: {1} contour, at min chi2 {2} value, dof={3}-{4}'.format(params["star"], zcol, params["chi2_value"],
                                                                          params["this_npix"], params["npars"]))
        ax.grid(True)
        fig.tight_layout()
        name = "{0}-{1}_{2}_{3}_{4}_{5}_contour_{6}.pdf".format(
            params["star"], params["obsnum"], params["chip"], xcol, ycol, zcol, params["suffix"])
        plt.savefig(os.path.join(params["path"], "plots", name))
        plt.savefig(os.path.join(params["path"], "plots", name.replace(".pdf", ".png")))
        plt.close()
    except Exception as e:
        logging.warning(__("database_contour did not plot due to \n{0}", e))


def compare_spectra(table, params):
    """Plot the min chi2 result against the observations."""
    extractor = DBExtractor(table)
    gamma_df = extractor.simple_extraction(columns=["gamma"])
    extreme_gammas = [min(gamma_df.gamma.values), max(gamma_df.gamma.values)]
    for ii, chi2_val in enumerate(chi2_names[0:-2]):
        df = extractor.ordered_extraction(columns=["teff_1", "logg_1", "feh_1", "gamma", chi2_val],
                                          order_by=chi2_val, limit=1, asc=True)

        params1 = [df["teff_1"].values[0], df["logg_1"].values[0], df["feh_1"].values[0]]

        params1 = [float(param1) for param1 in params1]

        gamma = df["gamma"].values

        obs_name, obs_params, output_prefix = bhm_helper_function(params["star"], params["obsnum"], ii + 1)
        print(obs_name)
        obs_spec = load_spectrum(obs_name)

        # Mask out bad portion of observed spectra
        # obs_spec = spectrum_masking(obs_spec, params["star"], params["obsnum"], ii + 1)

        # Barycentric correct spectrum
        _obs_spec = barycorr_crires_spectrum(obs_spec, extra_offset=None)
        normalization_limits = [obs_spec.xaxis[0] - 5, obs_spec.xaxis[-1] + 5]
        # models
        # print("params for models", params1)
        mod1 = load_starfish_spectrum(params1, limits=normalization_limits,
                                      hdr=True, normalize=False, area_scale=True,
                                      flux_rescale=True)

        bhm_grid_func = one_comp_model(mod1.xaxis, mod1.flux, gammas=gamma)
        bhm_upper_gamma = one_comp_model(mod1.xaxis, mod1.flux, gammas=extreme_gammas[1])
        bhm_lower_gamma = one_comp_model(mod1.xaxis, mod1.flux, gammas=extreme_gammas[0])

        bhm_grid_model = bhm_grid_func(obs_spec.xaxis).squeeze()
        bhm_grid_model_full = bhm_grid_func(mod1.xaxis).squeeze()
        bhm_upper_gamma = bhm_upper_gamma(obs_spec.xaxis).squeeze()
        bhm_lower_gamma = bhm_lower_gamma(obs_spec.xaxis).squeeze()

        model_spec_full = Spectrum(flux=bhm_grid_model_full, xaxis=mod1.xaxis)
        model_spec = Spectrum(flux=bhm_grid_model, xaxis=obs_spec.xaxis)
        bhm_upper_gamma = Spectrum(flux=bhm_upper_gamma, xaxis=obs_spec.xaxis)
        bhm_lower_gamma = Spectrum(flux=bhm_lower_gamma, xaxis=obs_spec.xaxis)

        model_spec = model_spec.remove_nans()
        model_spec = model_spec.normalize(method="exponential")
        model_spec_full = model_spec_full.remove_nans()
        model_spec_full = model_spec_full.normalize(method="exponential")
        bhm_lower_gamma = bhm_lower_gamma.remove_nans()
        bhm_lower_gamma = bhm_lower_gamma.normalize(method="exponential")
        bhm_upper_gamma = bhm_upper_gamma.remove_nans()
        bhm_upper_gamma = bhm_upper_gamma.normalize(method="exponential")

        from mingle.utilities.chisqr import chi_squared
        chisqr = chi_squared(obs_spec.flux, model_spec.flux)

        print("Recomputed chi^2 = {0}".format(chisqr))
        print("Database chi^2 = {0}".format(df[chi2_val]))
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        plt.plot(obs_spec.xaxis, obs_spec.flux + 0.01, label="0.05 + Observation, {}".format(obs_name))
        plt.plot(model_spec.xaxis, model_spec.flux, label="Minimum \chi^2 model")
        plt.plot(model_spec_full.xaxis, model_spec_full.flux, "--", label="Model_full_res")
        plt.plot(bhm_lower_gamma.xaxis, bhm_lower_gamma.flux, "-.", label="gamma={}".format(extreme_gammas[0]))
        plt.plot(bhm_upper_gamma.xaxis, bhm_upper_gamma.flux, ":", label="gamma={}".format(extreme_gammas[1]))
        plt.title("bhm spectrum")
        plt.legend()

        fig.tight_layout()
        name = "{0}-{1}_{2}_{3}_bhm_min_chi2_spectrum_comparison_{4}.png".format(
            params["star"], params["obsnum"], params["chip"], chi2_val, params["suffix"])
        plt.savefig(os.path.join(params["path"], "plots", name))
        plt.close()

        plt.plot(obs_spec.xaxis, obs_spec.flux, label="Observation")
        plt.show()


def contrast_bhm_results(table, params):
    extractor = DBExtractor(table)
    star_name = params["star"]
    obsnum = params["obsnum"]
    ___, host_params, ___ = bhm_helper_function(star_name, obsnum, 1)
    h_temp, h_logg, h_feh = host_params['temp'], host_params['logg'], host_params["fe_h"]

    print("Expected Parameters\n---------------------\nteff={0}\tlogg={1}\tfeh={2}".format(h_temp, h_logg, h_feh))
    print("BHM SOLUTIONS\n---------------------")
    for ii, chi2_val in enumerate(chi2_names):
        df = extractor.minimum_value_of(chi2_val)
        print("{0}: teff={1:5}\tlogg={2:3.02}\t".format(chi2_val, df.teff_1.values[0], df.logg_1.values[0]) +
              "feh={0:4.1}\tgamma={1:3.2},\txcorr={2:3.2},\tchi2={3:8.03}".format(df.feh_1.values[0],
                                                                                  float(df.gamma.values[0]),
                                                                                  df.xcorr_1.values[0],
                                                                                  df[chi2_val].values[0]))
