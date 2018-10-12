#!/usr/bin/env python
import logging
import os

import numpy as np
from logutils import BraceMessage as __
from matplotlib import pyplot as plt
from matplotlib import rc
from scipy.optimize import curve_fit, newton
from spectrum_overload import Spectrum

from mingle.models.broadcasted_models import inherent_alpha_model
from mingle.utilities import chi2_at_sigma
from mingle.utilities.chisqr import reduced_chi_squared
from mingle.utilities.crires_utilities import barycorr_crires_spectrum
from mingle.utilities.db_utils import DBExtractor
from mingle.utilities.debug_utils import timeit2
from mingle.utilities.phoenix_utils import load_starfish_spectrum
from mingle.utilities.spectrum_utils import load_spectrum
from simulators.iam_module import iam_helper_function

rc("image", cmap="inferno")
chi2_names = ["chi2_1", "chi2_2", "chi2_3", "chi2_4", "coadd_chi2"]
npix_names = ["npix_1", "npix_2", "npix_3", "npix_4", "coadd_npix"]


def get_npix_values(table):
    npix_values = {}
    df_npix = DBExtractor(table).simple_extraction(columns=npix_names)

    for col in npix_names:
        assert len(set(df_npix[col].values)) == 1
        npix_values[col] = df_npix[col].values[0]

    return npix_values


def rv_plot(table, params):
    for chi2_val, npix_val in zip(chi2_names, npix_names):
        red_chi2 = "red_{0}".format(chi2_val)
        df = DBExtractor(table).simple_extraction(columns=["rv", chi2_val, "teff_2"])
        df[red_chi2] = reduced_chi_squared(df[chi2_val], params["npix"][npix_val], params["npars"])
        fig, ax = plt.subplots()
        c = ax.scatter(df["rv"], df[chi2_val], c=df["teff_2"], alpha=0.8)
        cbar = plt.colorbar(c)
        cbar.ax.set_ylabel(r"teff_2")
        ax.set_xlabel(r'RV offset', fontsize=12)
        ax.set_ylabel(r"${0}$".format(chi2_val), fontsize=12)
        ax.set_title(r'$teff_2$ (color) and companion temperature.')
        ax.grid(True)
        fig.tight_layout()
        name = "{0}-{1}_{2}_scatter_rv_{3}_{4}.pdf".format(
            params["star"], params["obsnum"], params["chip"], chi2_val, params["suffix"])
        plt.savefig(os.path.join(params["path"], "plots", name))
        plt.savefig(os.path.join(params["path"], "plots", name.replace(".pdf", ".png")))
        plt.close()

        fig, ax = plt.subplots()
        c = ax.scatter(df["rv"], df[red_chi2], c=df["teff_2"], alpha=0.8)
        cbar = plt.colorbar(c)
        cbar.ax.set_ylabel(r"teff_2")
        ax.set_xlabel(r'RV offset', fontsize=12)
        ax.set_ylabel(r"${0}$".format(red_chi2), fontsize=12)
        ax.set_title(r'$teff_2$ (color) and companion temperature.')
        ax.grid(True)
        fig.tight_layout()
        name = "{0}-{1}_{2}_scatter_rv_reduced_{3}_{4}.pdf".format(
            params["star"], params["obsnum"], params["chip"], chi2_val, params["suffix"])
        plt.savefig(os.path.join(params["path"], "plots", name))
        plt.savefig(os.path.join(params["path"], "plots", name.replace(".pdf", ".png")))
        plt.close()


def xshift(x, num):
    """Shift x position slightly."""
    return x + num * (x * 0.1)


@timeit2
def fix_host_parameters(table, params):
    extractor = DBExtractor(table)
    print("Fixed host analysis.")
    nrows, ncols = 3, 2
    columns = ["teff_2", "logg_2", "feh_2", "gamma", "rv"]
    assert len(columns) <= (nrows * ncols)

    for jj, (chi2_val, npix_val) in enumerate(zip(chi2_names, npix_names)):
        if jj == 4:
            chi2legend = "coadd chi2"
        else:
            chi2legend = "det {0}".format(jj + 1)

        fig, axes = plt.subplots(nrows, ncols)
        fig.tight_layout()
        indices = np.arange(nrows * ncols).reshape(nrows, ncols)

        fixed_params = {"teff_1": params["teff"], "logg_1": params["logg"], "feh_1": params["fe_h"]}
        for ii, col in enumerate(columns):
            df = extractor(columns=[col, chi2_val], fixed=fixed_params)
            axis_pos = [int(x) for x in np.where(indices == ii)]
            df.plot(x=col, y=chi2_val, kind="scatter",
                    ax=axes[axis_pos[0], axis_pos[1]], label=chi2legend)

        plt.suptitle("Co-add Chi**2 Results (Fixed host): {0}-{1}".format(
            params["star"], params["obsnum"]))
        name = "{0}-{1}_coadd_fixed_host_params_full_gamma_{2}_{3}.png".format(
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
            df = extractor(columns=[col, chi2_val], fixed=fixed_params)
        df[red_chi2] = reduced_chi_squared(df[chi2_val], params["npix"][npix_val], params["npars"])

        axis_pos = [int(x) for x in np.where(indices == ii)]
        df.plot(x=col, y=red_chi2, kind="scatter",
                ax=axes[axis_pos[0], axis_pos[1]], label=chi2legend)

        plt.suptitle("Co-add reduced Chi**2 Results (Fixed host): {0}-{1}".format(
            params["star"], params["obsnum"]))
        name = "{0}-{1}_coadd_fixed_host_params_full_gamma_{2}_reduced_{3}.png".format(
            params["star"], params["obsnum"], params["suffix"], chi2_val)
        fig.savefig(os.path.join(params["path"], "plots", name))
        fig.savefig(os.path.join(params["path"], "plots", name.replace(".pdf", ".png")))
        plt.close()

    fix_host_parameters_individual(table, params)


def fix_host_parameters_individual(table, params):
    nrows, ncols = 1, 1
    columns = ["teff_2", "logg_2", "feh_2", "gamma", "rv"]
    extractor = DBExtractor(table)
    for ii, col in enumerate(columns):
        for jj, (chi2_val, npix_val) in enumerate(zip(chi2_names, npix_names)):
            if jj == 4:
                chi2legend = "coadd chi2"
            else:
                chi2legend = "det {0}".format(jj + 1)
            red_chi2 = "red_{0}".format(chi2_val)
            fig, axes = plt.subplots(nrows, ncols)
            fig.tight_layout()
            fixed_params = {"teff_1": params["teff"], "logg_1": params["logg"], "feh_1": params["fe_h"]}
            df = extractor(columns=[col, chi2_val], fixed=fixed_params)

            df[red_chi2] = reduced_chi_squared(df[chi2_val], params["npix"][npix_val], params["npars"])
            df.plot(x=col, y=chi2_val, kind="scatter",
                    ax=axes, label=chi2legend)
            name = "{0}-{1}_coadd_fixed_host_params_full_gamma_{2}_{3}_individual_{4}.png".format(
                params["star"], params["obsnum"], params["suffix"], chi2_val, col)
            plt.suptitle("Co-add {2}-Chi**2 Results (Fixed host): {0}-{1}: {3}".format(
                params["star"], params["obsnum"], chi2_val, col))
            fig.savefig(os.path.join(params["path"], "plots", name))
            fig.savefig(os.path.join(params["path"], "plots", name.replace(".pdf", ".png")))
            plt.close()
        plt.close()


def parabola_plots(table, params):
    extractor = DBExtractor(table)
    parabola_list = ["teff_2", "gamma", "rv"]
    for par in parabola_list:
        df = extractor.simple_extraction(columns=[par])
        unique_par = list(set(df[par].values))
        unique_par.sort()
        print("Unique ", par, " values =", unique_par)

        for chi2_val, npix_val in zip(chi2_names, npix_names):
            min_chi2 = []
            for unique_val in unique_par:
                df_chi2 = extractor.fixed_ordered_extraction(columns=[par, chi2_val], order_by=chi2_val,
                                                             fixed={par: float(unique_val)}, limit=3, asc=True)
                min_chi2.append(df_chi2[chi2_val].values[0])

            plt.plot(unique_par, min_chi2, ".-", label=chi2_val)
            # popt, _ = curve_fit(parabola, unique_par, min_chi2)
            popt, _ = fit_chi2_parabola(unique_par, min_chi2)

            x = np.linspace(unique_par[0], unique_par[-1], 40)
            plt.plot(x, parabola(x, *popt), "--")
            plt.xlabel(r"${0}$".format(par))
            plt.ylabel(r"$\chi^2$")

        plt.legend()
        filename = "Parabola_fit_{0}-{1}_{2}_param_{3}_{4}.png".format(
            params["star"], params["obsnum"], params["chip"], par, params["suffix"])

        plt.savefig(os.path.join(params["path"], "plots", filename))
        plt.close()
        print("saved parabolas for ", par)
    plt.close()


def slice_k_closest_around_x(x, x0, k):
    """Get k closest value a around index x."""
    if k % 2 == 0:
        raise ValueError("k must be odd.")
    lcut = x0 - np.floor(k / 2)
    ucut = x0 + np.floor(k / 2)
    slice_ = np.where((lcut <= x) * (x <= ucut))
    return slice_[0]


def fit_chi2_parabola(x, y, pts=5):
    """Fix parabola to chi_2 values.

    Limit to the 5 minimum points
    """
    x, y = np.asarray(x), np.asarray(y)
    indicies = np.arange(len(y))
    loc = np.argmin(y)
    index = slice_k_closest_around_x(indicies, loc, k=pts)
    new_x = x[index]
    new_y = y[index]
    popt, pcov = curve_fit(parabola, new_x, new_y)
    return popt, pcov


def chi2_parabola_plots(table, params):
    extractor = DBExtractor(table)
    parabola_list = ["teff_2", "gamma", "rv"]
    for par in parabola_list:
        df = extractor.simple_extraction(columns=[par])
        unique_par = list(set(df[par].values))
        unique_par.sort()

        for chi2_val, npix_val in zip(chi2_names, npix_names):
            min_chi2 = []
            for unique_val in unique_par:
                df_chi2 = extractor.fixed_ordered_extraction(columns=[par, chi2_val], order_by=chi2_val,
                                                             fixed={par: float(unique_val)}, limit=3, asc=True)
                min_chi2.append(df_chi2[chi2_val].values[0])

            min_chi2 = min_chi2 - min(min_chi2)

            plt.plot(unique_par, min_chi2, ".-", label=chi2_val)

            # popt, _ = curve_fit(parabola, unique_par, min_chi2)
            popt, _ = fit_chi2_parabola(unique_par, min_chi2)
            # print("params", popt)
            x = np.linspace(unique_par[0], unique_par[-1], 40)
            plt.plot(x, parabola(x, *popt), "--")  # , label="parabola")
            plt.xlabel(r"${0}$".format(par))
            plt.ylabel(r"$\Delta \chi^2$ from minimum")
            plt.ylim([-0.05 * np.max(min_chi2), np.max(min_chi2)])

            # Find roots
            if chi2_val == "coadd_chi2":
                try:
                    residual = lambda x: parabola(x, *popt) - chi2_at_sigma(1, params["npars"])
                    min_chi2_par = unique_par[np.argmin(min_chi2)]
                    try:
                        lower_bound = newton(residual, (min_chi2_par + unique_par[0]) / 2) - min_chi2_par
                    except RuntimeError:
                        lower_bound = np.nan
                    try:
                        upper_bound = newton(residual, (min_chi2_par + unique_par[-1]) / 2) - min_chi2_par
                    except RuntimeError:
                        upper_bound = np.nan

                    print("{0} solution {1: 5.3f} {2:+5.3f} {3:+5.3f}".format(chi2_val, min_chi2_par, lower_bound,
                                                                              upper_bound))
                    plt.annotate("{0: 5.3f} {1:+5.3f} {2:+5.3f}".format(min_chi2_par, lower_bound, upper_bound),
                                 xy=(min_chi2_par, 0),
                                 xytext=(0.5, 0.5), textcoords="figure fraction", arrowprops={"arrowstyle": "->"})
                except Exception as e:
                    print(e)
                    logging.warning("Could not Annotate the contour plot")

        plt.axhline(y=chi2_at_sigma(1, params["npars"]), label="1 sigma")
        plt.axhline(y=chi2_at_sigma(2, params["npars"]), label="2 sigma")
        plt.axhline(y=chi2_at_sigma(3, params["npars"]), label="3 sigma")
        plt.axhline(y=chi2_at_sigma(1, dof=1), label="1 sigma 1 par", color="k", ls="--")
        plt.axhline(y=chi2_at_sigma(2, dof=1), label="2 sigma 1 par", color="k", ls="--")
        plt.axhline(y=chi2_at_sigma(3, dof=1), label="3 sigma 1 par", color="k", ls="--")

        plt.legend()
        filename = "red_Chi2_Parabola_fit_{0}-{1}_{2}_param_{3}_{4}.png".format(
            params["star"], params["obsnum"], params["chip"], par, params["suffix"])

        plt.savefig(os.path.join(params["path"], "plots", filename))
        plt.close()
        print("Saved chi2 parabolas for ", par)
    plt.close()


def chi2_individual_parabola_plots(table, params):
    extractor = DBExtractor(table)
    parabola_list = ["teff_2", "gamma", "rv"]
    for par in parabola_list:
        df = extractor.simple_extraction(columns=[par])
        unique_par = list(set(df[par].values))
        unique_par.sort()

        for chi2_val, npix_val in zip(chi2_names, npix_names):
            plt.figure()
            min_chi2 = []
            for unique_val in unique_par:
                df_chi2 = extractor.fixed_ordered_extraction(columns=[par, chi2_val], order_by=chi2_val,
                                                             fixed={par: float(unique_val)}, limit=3, asc=True)
                min_chi2.append(df_chi2[chi2_val].values[0])

            min_chi2 = min_chi2 - min(min_chi2)

            plt.plot(unique_par, min_chi2, ".-", label=chi2_val)

            # popt, _ = curve_fit(parabola, unique_par, min_chi2)
            popt, _ = fit_chi2_parabola(unique_par, min_chi2)
            # print("params", popt)
            x = np.linspace(unique_par[0], unique_par[-1], 40)
            plt.plot(x, parabola(x, *popt))  # , label="parabola")
            plt.xlabel(r"${0}$".format(par))
            plt.ylabel(r"$\Delta \chi^2$ from minimum")
            plt.ylim([-0.05 * np.max(min_chi2), np.max(min_chi2)])
            # Find roots
            try:
                residual = lambda x: parabola(x, *popt) - chi2_at_sigma(1, params["npars"])
                min_chi2_par = unique_par[np.argmin(min_chi2)]
                try:
                    lower_bound = newton(residual, (min_chi2_par + unique_par[0]) / 2) - min_chi2_par
                except RuntimeError:
                    lower_bound = np.nan
                try:
                    upper_bound = newton(residual, (min_chi2_par + unique_par[-1]) / 2) - min_chi2_par
                except RuntimeError:
                    upper_bound = np.nan

                print("{0} solution {1: 5.3f} {2:+5.3f} {3:+5.3f}".format(chi2_val, min_chi2_par, lower_bound,
                                                                          upper_bound))
                plt.annotate("{0: 5.3f} {1:+5.3f} {2:+5.3f}".format(min_chi2_par, lower_bound, upper_bound),
                             xy=(min_chi2_par, 0), xytext=(0.5, 0.5), textcoords="figure fraction",
                             arrowprops={"arrowstyle": "->"})
            except Exception as e:
                print(e)
                logging.warning("Could not Annotate the parabola plot")

            plt.axhline(y=chi2_at_sigma(1, params["npars"]), label="1 sigma")
            plt.axhline(y=chi2_at_sigma(2, params["npars"]), label="2 sigma")
            plt.axhline(y=chi2_at_sigma(3, params["npars"]), label="3 sigma")

            plt.legend()
            filename = "red_Chi2_Parabola_fit_{0}-{1}_{2}_param_{3}_{4}_individual_{5}.png".format(
                params["star"], params["obsnum"], params["chip"], par, params["suffix"], chi2_val)

            plt.savefig(os.path.join(params["path"], "plots", filename))
            plt.close()
            print("Saved individual chi2 parabolas for ", par)
            plt.close()


def smallest_chi2_values(table, params, num=10):
    """Find smallest chi2 in table."""
    chi2_val = "coadd_chi2"
    df = DBExtractor(table).ordered_extraction(order_by=chi2_val, limit=num, asc=True)
    df[chi2_val] = reduced_chi_squared(df[chi2_val], params["npix"]["coadd_npix"], params["npars"])

    df_min = df[:1]
    print("Smallest Co-add reduced Chi2 values in the database.")
    print(df.head(n=num))
    name = "minimum_coadd_chi2_db_output_{0}_{1}_{2}.csv".format(params["star"], params["obsnum"], params["suffix"])
    from bin.check_result import main as visual_inspection
    df.to_csv(os.path.join(params["path"], name))

    plot_name = os.path.join(params["path"], "plots",
                             "visual_inspection_min_chi2_coadd_{0}_{1}_{2}.png".format(params["star"],
                                                                                       params["obsnum"],
                                                                                       params["suffix"]))
    visual_inspection(params["star"], params["obsnum"], float(df_min.teff_1), float(df_min.logg_1),
                      float(df_min.feh_1), float(df_min.teff_2), float(df_min.logg_2),
                      float(df_min.feh_2), float(df_min.gamma), float(df_min.rv), plot_name=plot_name)


def parabola(x, a, b, c):
    return a * x ** 2 + b * x + c


@timeit2
def fix_host_parameters_reduced_gamma(table, params):
    print("Fixed host analysis with reduced gamma.")
    extractor = DBExtractor(table)
    d_gamma = 5
    nrows, ncols = 3, 2
    columns = ["teff_2", "logg_2", "feh_2", "gamma", "rv"]
    assert len(columns) <= (nrows * ncols)
    fig, axes = plt.subplots(nrows, ncols)
    fig.tight_layout()
    indices = np.arange(nrows * ncols).reshape(nrows, ncols)

    for jj, (chi2_val, npix_val) in enumerate(zip(chi2_names, npix_names)):
        if jj == 4:
            chi2legend = "coadd"
        else:
            chi2legend = "det {0}".format(jj + 1)
        # Select lowest chi square gamma values.
        df = extractor.ordered_extraction(order_by=chi2_val, columns=["gamma", chi2_val], limit=1)

        min_chi2_gamma = df.loc[0, "gamma"]
        upper_lim = min_chi2_gamma + d_gamma
        lower_lim = min_chi2_gamma - d_gamma
        for ii, col in enumerate(columns):
            fixed_values = {"teff_1": int(params["teff"]), "logg_1": float(params["logg"]),
                            "feh_1": float(params["fe_h"])}
            df = extractor.fixed_extraction(columns={col, chi2_val, "gamma", "teff_1"},
                                            fixed=fixed_values)
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
        for ii, col in enumerate(columns):
            fixed_values = {"teff_1": int(params["teff"]), "logg_1": float(params["logg"]),
                            "feh_1": float(params["fe_h"])}
            df = extractor.fixed_extraction(columns={col, chi2_val, "gamma", "teff_1"},
                                            fixed=fixed_values)
            df = df[[(df.gamma > float(lower_lim)) & (df.gamma < float(upper_lim))]]

            df[red_chi2] = reduced_chi_squared(df[chi2_val], params["npix"][npix_val], params["npars"])
            axis_pos = [int(x) for x in np.where(indices == ii)]
            df.plot(x=col, y=red_chi2, kind="scatter",
                    ax=axes[axis_pos[0], axis_pos[1]], label=chi2legend)

    plt.suptitle("Coadd reduced Chi**2 Results: {0}-{1}".format(params["star"], params["obsnum"]))
    name = "{0}-{1}_coadd_fixed_host_params_{2}.png".format(
        params["star"], params["obsnum"], params["suffix"])
    fig.savefig(os.path.join(params["path"], "plots", name))
    fig.savefig(os.path.join(params["path"], "plots", name.replace(".pdf", ".png")))
    plt.close()

    fix_host_parameters_reduced_gamma_individual(table, params)


def fix_host_parameters_reduced_gamma_individual(table, params):
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
        print("Reduced gamma_limits", lower_lim, upper_lim)

        columns = ["teff_2", "logg_2", "feh_2", "gamma", "rv"]

        for ii, col in enumerate(columns):
            fixed_values = {"teff_1": int(params["teff"]), "logg_1": float(params["logg"]),
                            "feh_1": float(params["fe_h"])}
            df = extractor.fixed_extraction(columns={col, chi2_val, "gamma", "teff_1"},
                                            fixed=fixed_values)
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


def get_column_limits(table, params):
    extractor = DBExtractor(table)
    print("Database Column Value Ranges")
    for col in extractor.cols.keys():
        min_df = extractor.ordered_extraction(order_by=col, columns=[col], limit=1, asc=True)
        max_df = extractor.ordered_extraction(order_by=col, columns=[col], limit=1, asc=False)
        print("{0:10}\t\t{1:5.3} - {2:5.3}".format(col, float(min_df[col].values[0]), float(max_df[col].values[0])))


def contours(table, params):
    extractor = DBExtractor(table)
    for par_limit, contour_param in zip(["gamma", "rv"], ["rv", "gamma"]):
        fixed_params = {"teff_1": params["teff"], "logg_1": params["logg"], "feh_1": params["fe_h"],
                        "logg_2": params["logg"], "feh_2": params["fe_h"]}
        for chi2_val, npix_val in zip(chi2_names, npix_names):
            df_min_chi2 = extractor.minimum_value_of(chi2_val)

            fixed_params.update({par_limit: float(df_min_chi2[par_limit][0])})
            df = extractor.fixed_extraction(columns=["teff_2", "rv", "gamma", chi2_val],
                                            fixed=fixed_params)

            params["this_npix"] = params["npix"][npix_val]
            params["par_limit"] = par_limit

            pars = [contour_param, "teff_2", chi2_val]
            dataframe_contour(df, xcol=pars[0], ycol=pars[1], zcol=pars[2], params=params)

    # Using Reduced chi2 value
    for par_limit, contour_param in zip(["gamma", "rv"], ["rv", "gamma"]):
        fixed_params = {"teff_1": params["teff"], "logg_1": params["logg"], "feh_1": params["fe_h"],
                        "logg_2": params["logg"], "feh_2": params["fe_h"]}
        for chi2_val, npix_val in zip(chi2_names, npix_names):
            red_chi2 = "red_{0}".format(chi2_val)

            df_min_chi2 = extractor.minimum_value_of(chi2_val)
            fixed_params.update({par_limit: float(df_min_chi2[par_limit][0])})
            df = extractor.fixed_extraction(columns=["teff_2", "rv", "gamma", chi2_val],
                                            fixed=fixed_params)

            df[red_chi2] = reduced_chi_squared(df[chi2_val], params["npix"][npix_val], params["npars"])

            params["this_npix"] = params["npix"][npix_val]
            params["par_limit"] = par_limit

            pars = [contour_param, "teff_2", red_chi2]
            dataframe_contour(df, xcol=pars[0], ycol=pars[1], zcol=pars[2], params=params)

    # Using teff_1 contour
    for par_limit, contour_param in zip(["gamma", "rv"], ["rv", "gamma"]):
        fixed_params = {"logg_1": params["logg"], "feh_1": params["fe_h"],
                        "logg_2": params["logg"], "feh_2": params["fe_h"]}
        for chi2_val, npix_val in zip(chi2_names, npix_names):
            df_min_chi2 = extractor.minimum_value_of(chi2_val)
            fixed_params.update({par_limit: df_min_chi2[par_limit].values[0],
                                 "teff_2": df_min_chi2["teff_2"].values[0]})
            df = extractor.fixed_extraction(columns=["teff_1", "rv", "gamma", chi2_val],
                                            fixed=fixed_params)

            params["this_npix"] = params["npix"][npix_val]
            params["par_limit"] = par_limit

            pars = [contour_param, "teff_1", chi2_val]
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
            '{0}: {1} contour, at min chi2 {2} value, dof={3}-{4}'.format(params["star"], zcol, params["par_limit"],
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


def test_figure(table, params):
    chi2_val = "coadd_chi2"
    df = DBExtractor(table).simple_extraction(columns=["gamma", chi2_val], limit=10000)
    fig, ax = plt.subplots()
    red_chi2 = reduced_chi_squared(df[chi2_val], params["npix"]["coadd_npix"], params["npars"])
    ax.scatter(df["gamma"], red_chi2, s=3, alpha=0.5)

    ax.set_xlabel(r'$\gamma$', fontsize=15)
    ax.set_ylabel(r"$ Reduced {0}$".format(chi2_val), fontsize=15)
    ax.set_title(r'$\gamma$ and $\chi^2$.')

    ax.grid(True)
    fig.tight_layout()
    name = "{0}-{1}_{2}_red_test_figure_{3}_{4}.pdf".format(
        params["star"], params["obsnum"], params["chip"], chi2_val, params["suffix"])
    plt.savefig(os.path.join(params["path"], "plots", name))
    plt.close()


def compare_spectra(table, params):
    """Plot the min chi2 result against the observations."""
    extractor = DBExtractor(table)
    for ii, chi2_val in enumerate(chi2_names[0:-2]):
        df = extractor.minimum_value_of(chi2_val)
        df = df[["teff_1", "logg_1", "feh_1", "gamma",
                 "teff_2", "logg_2", "feh_2", "rv", chi2_val]]

        params1 = [df["teff_1"].values[0], df["logg_1"].values[0], df["feh_1"].values[0]]
        params2 = [df["teff_2"].values[0], df["logg_2"].values[0], df["feh_2"].values[0]]

        params1 = [float(param1) for param1 in params1]
        params2 = [float(param2) for param2 in params2]

        gamma = df["gamma"].values
        rv = df["rv"].values

        from simulators.iam_module import iam_helper_function
        obs_name, obs_params, output_prefix = iam_helper_function(params["star"], params["obsnum"], ii + 1)

        obs_spec = load_spectrum(obs_name)

        # Mask out bad portion of observed spectra
        # obs_spec = spectrum_masking(obs_spec, params["star"], params["obsnum"], ii + 1)

        # Barycentric correct spectrum
        _obs_spec = barycorr_crires_spectrum(obs_spec, extra_offset=None)
        normalization_limits = [obs_spec.xaxis[0] - 5, obs_spec.xaxis[-1] + 5]
        # models
        # print("params for models", params1, params2)
        mod1 = load_starfish_spectrum(params1, limits=normalization_limits,
                                      hdr=True, normalize=False, area_scale=True,
                                      flux_rescale=True)

        mod2 = load_starfish_spectrum(params2, limits=normalization_limits,
                                      hdr=True, normalize=False, area_scale=True,
                                      flux_rescale=True)

        iam_grid_func = inherent_alpha_model(mod1.xaxis, mod1.flux, mod2.flux,
                                             rvs=rv, gammas=gamma)

        iam_grid_model = iam_grid_func(obs_spec.xaxis).squeeze()
        iam_grid_model_full = iam_grid_func(mod1.xaxis).squeeze()

        model_spec_full = Spectrum(flux=iam_grid_model_full, xaxis=mod1.xaxis)
        model_spec = Spectrum(flux=iam_grid_model, xaxis=obs_spec.xaxis)

        model_spec = model_spec.remove_nans()
        model_spec = model_spec.normalize(method="exponential")
        model_spec_full = model_spec_full.remove_nans()
        model_spec_full = model_spec_full.normalize(method="exponential")

        fig, ax = plt.subplots(1, 1)
        plt.plot(obs_spec.xaxis, obs_spec.flux, label="Observation")
        plt.plot(model_spec.xaxis, model_spec.flux, label="Minimum \chi^2 model")
        plt.plot(model_spec_full.xaxis, model_spec_full.flux, "--", label="Model_full_res")

        plt.legend()

        fig.tight_layout()
        name = "{0}-{1}_{2}_{3}_min_chi2_spectrum_comparison_{4}.png".format(
            params["star"], params["obsnum"], params["chip"], chi2_val, params["suffix"])
        plt.savefig(os.path.join(params["path"], "plots", name))
        plt.close()

        plt.plot(obs_spec.xaxis, obs_spec.flux, label="Observation")
        plt.show()


def contrast_iam_results(table, params):
    extractor = DBExtractor(table)
    star_name = params["star"]
    obsnum = params["obsnum"]
    ___, host_params, ___ = iam_helper_function(star_name, obsnum, 1)
    h_temp, h_logg, h_feh = host_params['temp'], host_params['logg'], host_params["fe_h"]
    c_temp = host_params.get("comp_temp")

    print(
        "Observation {4} - {5}\n"
        "Expected Parameters\n---------------------\n"
        "teff={0:5.0f}  logg={1:3.02f}  feh={2:4.01f} \tcompanion_temp={3:5.0f} ".format(h_temp, h_logg, h_feh, c_temp,
                                                                                         star_name, obsnum))

    print("IAM SOLUTIONS\n---------------------")
    for ii, chi2_val in enumerate(chi2_names):
        df = extractor.minimum_value_of(chi2_val)
        print(
            "{0:10} solution:\nCompanion: teff_2={1:5.0f}  logg2={2:4.02f}  ".format(chi2_val, df.teff_2.values[0],
                                                                                     df.logg_2.values[0]) +
            "feh2={0:4.01f}  gamma={1:4.01f}  rv={2:4.01f}  ".format(df.feh_2.values[0], float(df.gamma.values[0]),
                                                                     float(df.rv.values[0])) +
            "Host: teff={0:5.0f}  logg={1:4.02f}  feh={2:4.01f}  chi2={3:8.02f} median alpha={4:5.03f}".format(
                df.teff_1.values[0],
                df.logg_1.values[0],
                df.feh_1.values[0],
                df[chi2_val].values[0],
                np.median([df["alpha_1"].values[0],
                           df["alpha_2"].values[0],
                           df["alpha_3"].values[0],
                           df["alpha_4"].values[0]])))
