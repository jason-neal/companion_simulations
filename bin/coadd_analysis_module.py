#!/usr/bin/env python
import logging
import os

import numpy as np
import pandas as pd
import sqlalchemy as sa
from matplotlib import pyplot as plt
from matplotlib import rc
from scipy.optimize import curve_fit, newton
from scipy.stats import chi2
from spectrum_overload import Spectrum

from mingle.models.broadcasted_models import inherent_alpha_model
from mingle.utilities.chisqr import reduced_chi_squared
from mingle.utilities.crires_utilities import barycorr_crires_spectrum
from mingle.utilities.debug_utils import timeit2
from mingle.utilities.phoenix_utils import load_starfish_spectrum
from mingle.utilities.spectrum_utils import load_spectrum
from simulators.iam_module import iam_helper_function

rc("image", cmap="inferno")
chi2_names = ["chi2_1", "chi2_2", "chi2_3", "chi2_4", "coadd_chi2"]
npix_names = ["npix_1", "npix_2", "npix_3", "npix_4", "coadd_npix"]


def get_npix_values(table):
    npix_values = {}
    df_npix = pd.read_sql(
        sa.select([table.c[col] for col in npix_names]),
        table.metadata.bind)

    for col in npix_names:
        assert len(set(df_npix[col].values)) == 1
        npix_values[col] = df_npix[col].values[0]

    return npix_values


def rv_plot(table, params):
    for chi2_val, npix_val in zip(chi2_names, npix_names):
        red_chi2 = "red_{0}".format(chi2_val)
        df = pd.read_sql(
            sa.select([table.c["rv"], table.c[chi2_val], table.c["teff_2"]]),
            table.metadata.bind)
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


def display_arbitrary_norm_values(table, params):
    fig, axarr = plt.subplots(3)
    for ii, arbnorm in enumerate([r"arbnorm_1", r"arbnorm_2", r"arbnorm_3", r"arbnorm_4"]):
        df = pd.read_sql(
            sa.select([table.c.gamma, table.c[arbnorm],
                       table.c.rv, table.c.teff_2]), table.metadata.bind)

        c = axarr[0].scatter(xshift(df["rv"], ii), df[arbnorm],
                             c=df[r"teff_2"].values, alpha=0.9, label=arbnorm)

        axarr[0].set_xlabel(r'rv offset', fontsize=12)
        axarr[0].set_ylabel(r'Arbitrary norm', fontsize=12)
        axarr[0].set_title(r'Arbitrary normalization.')

        d = axarr[1].scatter(xshift(df[r"gamma"], ii), df[arbnorm],
                             c=df[r"teff_2"].values, alpha=0.9, label=arbnorm)
        axarr[1].set_xlabel(r'$\gamma$ offset', fontsize=12)
        axarr[1].set_ylabel(r'Arbitrary norm', fontsize=12)
        axarr[1].set_title(r'$\gamma$.')

        e = axarr[2].scatter(xshift(df[r"teff_2"], ii), df[arbnorm],
                             c=df[r"gamma"].values, alpha=0.9, label=arbnorm)
        axarr[2].set_xlabel(r'Companion temperature', fontsize=15)
        axarr[2].set_ylabel(r'Arbitrary norm ', fontsize=15)
        axarr[2].set_title(r'Companion Temperature')
    axarr[0].grid(True)
    axarr[1].grid(True)
    axarr[2].grid(True)

    cbar0 = plt.colorbar(c)
    cbar0.ax.set_ylabel(r" teff_2")
    cbar1 = plt.colorbar(d)
    cbar1.ax.set_ylabel(r" teff_2")
    cbar2 = plt.colorbar(e)
    cbar1.ax.set_ylabel(r"$\gamma$")
    fig.tight_layout()
    fig.suptitle("Arbitrary normalization used, \n slight shift with detector")
    name = "{0}-{1}_{2}_plot_arbnormalization_{3}.pdf".format(
        params["star"], params["obsnum"], params["chip"], params["suffix"])
    plt.savefig(os.path.join(params["path"], "plots", name))
    plt.savefig(os.path.join(params["path"], "plots", name.replace(".pdf", ".png")))
    plt.close()


@timeit2
def fix_host_parameters(table, params):
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

        for ii, col in enumerate(columns):
            df = pd.read_sql(
                sa.select([table.c[col], table.c[chi2_val]]).where(
                    sa.and_(table.c["teff_1"] == params["teff"],
                            table.c["logg_1"] == params["logg"],
                            table.c["feh_1"] == params["fe_h"])
                ), table.metadata.bind)
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
            df = pd.read_sql(
                sa.select([table.c[col], table.c[chi2_val]]).where(
                    sa.and_(table.c["teff_1"] == params["teff"],
                            table.c["logg_1"] == params["logg"],
                            table.c["feh_1"] == params["fe_h"])
                ), table.metadata.bind)
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

    for ii, col in enumerate(columns):
        for jj, (chi2_val, npix_val) in enumerate(zip(chi2_names, npix_names)):
            if jj == 4:
                chi2legend = "coadd chi2"
            else:
                chi2legend = "det {0}".format(jj + 1)
            red_chi2 = "red_{0}".format(chi2_val)
            fig, axes = plt.subplots(nrows, ncols)
            fig.tight_layout()
            df = pd.read_sql(
                sa.select([table.c[col], table.c[chi2_val]]).where(
                    sa.and_(table.c["teff_1"] == params["teff"],
                            table.c["logg_1"] == params["logg"],
                            table.c["feh_1"] == params["fe_h"])
                ), table.metadata.bind)
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
    parabola_list = ["teff_2", "gamma", "rv"]
    for par in parabola_list:
        df = pd.read_sql(sa.select([table.c[par]]), table.metadata.bind)
        unique_par = list(set(df[par].values))
        unique_par.sort()
        print("Unique ", par, " values =", unique_par)

        for chi2_val, npix_val in zip(chi2_names, npix_names):
            min_chi2 = []
            for unique_val in unique_par:
                df_chi2 = pd.read_sql(
                    sa.select([table.c[par], table.c[chi2_val]]).where(
                        table.c[par] == float(unique_val)).order_by(
                        table.c[chi2_val].asc()).limit(3), table.metadata.bind)
                min_chi2.append(df_chi2[chi2_val].values[0])

            plt.plot(unique_par, min_chi2, ".-", label=chi2_val)
            popt, pcov = curve_fit(parabola, unique_par, min_chi2)

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


def chi2_at_sigma(df, sigma):
    """Use inverse survival function to calculate the chi2 value for significances."""
    sigma_percent = {1: 0.68, 2: 0.90, 3: 0.99}
    return chi2(df).isf(1 - sigma_percent[sigma])


def slice_k_closest_around_x(x, x0, k):
    """Get k closest value a around index x."""
    if k % 2 == 0:
        raise ValueError("k must be odd.")
    lcut = x0 - np.floor(k / 2)
    ucut = x0 + np.floor(k / 2)
    slice = np.where((lcut <= x) * (x <= ucut))
    return slice[0]


def chi2_parabola_plots(table, params):
    parabola_list = ["teff_2", "gamma", "rv"]
    for par in parabola_list:
        df = pd.read_sql(sa.select([table.c[par]]), table.metadata.bind)
        unique_par = list(set(df[par].values))
        unique_par.sort()
        print(unique_par)

        for chi2_val, npix_val in zip(chi2_names, npix_names):
            min_chi2 = []
            for unique_val in unique_par:
                df_chi2 = pd.read_sql(
                    sa.select([table.c[par], table.c[chi2_val]]).where(
                        table.c[par] == float(unique_val)).order_by(
                        table.c[chi2_val].asc()).limit(3), table.metadata.bind)
                min_chi2.append(df_chi2[chi2_val].values[0])

            min_chi2 = min_chi2 - min(min_chi2)

            plt.plot(unique_par, min_chi2, ".-", label=chi2_val)

            popt, pcov = curve_fit(parabola, unique_par, min_chi2)
            print("params", popt)
            x = np.linspace(unique_par[0], unique_par[-1], 40)
            plt.plot(x, parabola(x, *popt))  # , label="parabola")
            plt.xlabel(r"${0}$".format(par))
            plt.ylabel(r"$\Delta \chi^2$ from minimum")

            # Find roots
            if chi2_val == "coadd_chi2":
                try:
                    residual = lambda x: parabola(x, *popt) - chi2_at_sigma(params["npars"], 1)
                    min_chi2_par = unique_par[np.argmin(min_chi2)]
                    lower_bound = newton(residual, (min_chi2_par + unique_par[0]) / 2)
                    upper_bound = newton(residual, (min_chi2_par + unique_par[-1]) / 2)

                    print("{0} solution {1} - {2} + {3}".format(chi2_val, min_chi2_par, lower_bound, upper_bound))
                    plt.annotate("{0} -{1} +{2}".format(min_chi2_par, lower_bound, upper_bound), xy=(min_chi2_par, 0),
                                 xytext=(0.5, 0.5), textcoords="figure fraction", arrowprops={"arrowstyle": "<-"})
                except:
                   logging.warning("Could not Annotate the contour plot")

        plt.axhline(y=chi2_at_sigma(params["npars"], 1), label="1 sigma")
        plt.axhline(y=chi2_at_sigma(params["npars"], 2), label="2 sigma")
        plt.axhline(y=chi2_at_sigma(params["npars"], 3), label="3 sigma")

        plt.legend()
        filename = "red_Chi2_Parabola_fit_{0}-{1}_{2}_param_{3}_{4}.png".format(
            params["star"], params["obsnum"], params["chip"], par, params["suffix"])

        plt.savefig(os.path.join(params["path"], "plots", filename))
        plt.close()
        print("Saved chi2 parabolas for ", par)
    plt.close()


def smallest_chi2_values(table, params, num=10):
    """Find smallest chi2 in table."""
    chi2_val = "coadd_chi2"
    df = pd.read_sql(
        sa.select(table.c).order_by(table.c[chi2_val].asc()).limit(num),
        table.metadata.bind)
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
        df = pd.read_sql(
            sa.select([table.c.gamma, table.c[chi2_val]]).order_by(
                table.c[chi2_val].asc()).limit(1),
            table.metadata.bind)

        min_chi2_gamma = df.loc[0, "gamma"]
        upper_lim = min_chi2_gamma + d_gamma
        lower_lim = min_chi2_gamma - d_gamma
        for ii, col in enumerate(columns):
            df = pd.read_sql(
                sa.select([table.c[col], table.c[chi2_val], table.c.gamma, table.c.teff_1], table.c.teff_1).where(
                    sa.and_(table.c["teff_1"] == int(params["teff"]),
                            table.c["logg_1"] == float(params["logg"]),
                            table.c["feh_1"] == float(params["fe_h"]),
                            table.c.gamma > float(lower_lim),
                            table.c.gamma < float(upper_lim)
                            )
                ), table.metadata.bind)
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
        df = pd.read_sql(
            sa.select([table.c.gamma, table.c[chi2_val]]).order_by(
                table.c[chi2_val].asc()).limit(1),
            table.metadata.bind)

        min_chi2_gamma = df.loc[0, "gamma"]
        upper_lim = min_chi2_gamma + d_gamma
        lower_lim = min_chi2_gamma - d_gamma
        for ii, col in enumerate(columns):
            df = pd.read_sql(
                sa.select([table.c[col], table.c[chi2_val], table.c.gamma, table.c.teff_1], table.c.teff_1).where(
                    sa.and_(table.c["teff_1"] == int(params["teff"]),
                            table.c["logg_1"] == float(params["logg"]),
                            table.c["feh_1"] == float(params["fe_h"]),
                            table.c.gamma > float(lower_lim),
                            table.c.gamma < float(upper_lim)
                            )
                ), table.metadata.bind)

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
    d_gamma = 5
    nrows, ncols = 1, 1

    for jj, (chi2_val, npix_val) in enumerate(zip(chi2_names, npix_names)):
        red_chi2 = "red_{0}".format(chi2_val)
        if jj == 4:
            chi2legend = "coadd"
        else:
            chi2legend = "det {0}".format(jj + 1)

        # Select lowest chi square gamma values.
        df = pd.read_sql(
            sa.select([table.c.gamma, table.c[chi2_val]]).order_by(
                table.c[chi2_val].asc()).limit(1),
            table.metadata.bind)

        min_chi2_gamma = df.loc[0, "gamma"]
        upper_lim = min_chi2_gamma + d_gamma
        lower_lim = min_chi2_gamma - d_gamma
        print("Reduced gamma_limits", lower_lim, upper_lim)

        columns = ["teff_2", "logg_2", "feh_2", "gamma", "rv"]

        for ii, col in enumerate(columns):
            df = pd.read_sql(
                sa.select([table.c[col], table.c[chi2_val], table.c.gamma, table.c.teff_1], table.c.teff_1).where(
                    sa.and_(table.c["teff_1"] == int(params["teff"]),
                            table.c["logg_1"] == float(params["logg"]),
                            table.c["feh_1"] == float(params["fe_h"]),
                            table.c.gamma > float(lower_lim),
                            table.c.gamma < float(upper_lim)
                            )
                ), table.metadata.bind)

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
    print("Database Column Value Ranges")
    for col in ["teff_1", "teff_2", "logg_1", "logg_2", "feh_1", "feh_2", "gamma", "rv",
                "chi2_1", "chi2_2", "chi2_3", "chi2_4", "coadd_chi2", "alpha_1",
                "alpha_2", "alpha_3", "alpha_4", "arbnorm_1", "arbnorm_2",
                "arbnorm_3", "arbnorm_4"]:
        min_df = pd.read_sql(
            sa.select([table.c[col]]).order_by(table.c[col].asc()).limit(1),
            table.metadata.bind)
        max_df = pd.read_sql(
            sa.select([table.c[col]]).order_by(table.c[col].desc()).limit(1),
            table.metadata.bind)
        print("{0:10}\t\t{1:5.3} - {2:5.3}".format(col, float(min_df[col].values[0]), float(max_df[col].values[0])))


def contours(table, params):
    for par_limit, contour_param in zip(["gamma", "rv"], ["rv", "gamma"]):
        for chi2_val, npix_val in zip(chi2_names, npix_names):
            df_min_chi2 = pd.read_sql(
                sa.select(table.c).order_by(
                    table.c[chi2_val].asc()).limit(1),
                table.metadata.bind)

            df = pd.read_sql(
                sa.select([table.c["teff_2"], table.c["rv"], table.c["gamma"], table.c[chi2_val]]).where(
                    sa.and_(table.c[par_limit] == float(df_min_chi2[par_limit][0]),
                            table.c.teff_1 == int(params["teff"]),
                            table.c.logg_1 == float(params["logg"]),
                            table.c.feh_1 == float(params["fe_h"]),
                            table.c.logg_2 == float(params["logg"]),  # Fix companion logg
                            table.c.feh_2 == float(params["fe_h"]))),  # Fix companion fe_h
                table.metadata.bind)

            params["this_npix"] = params["npix"][npix_val]
            params["par_limit"] = par_limit

            pars = [contour_param, "teff_2", chi2_val]
            dataframe_contour(df, xcol=pars[0], ycol=pars[1], zcol=pars[2], params=params)

    # Using Reduced chi2 value
    for par_limit, contour_param in zip(["gamma", "rv"], ["rv", "gamma"]):
        for chi2_val, npix_val in zip(chi2_names, npix_names):
            red_chi2 = "red_{0}".format(chi2_val)
            df_min_chi2 = pd.read_sql(
                sa.select(table.c).order_by(
                    table.c[chi2_val].asc()).limit(1),
                table.metadata.bind)

            df = pd.read_sql(
                sa.select([table.c["teff_2"], table.c["rv"], table.c["gamma"], table.c[chi2_val]]).where(
                    sa.and_(table.c[par_limit] == float(df_min_chi2[par_limit][0]),
                            table.c.teff_1 == int(params["teff"]),
                            table.c.logg_1 == float(params["logg"]),
                            table.c.feh_1 == float(params["fe_h"]),
                            table.c.logg_2 == float(params["logg"]),  # Fix companion logg
                            table.c.feh_2 == float(params["fe_h"]))),  # Fix companion fe_h
                table.metadata.bind)

            df[red_chi2] = reduced_chi_squared(df[chi2_val], params["npix"][npix_val], params["npars"])

            params["this_npix"] = params["npix"][npix_val]
            params["par_limit"] = par_limit

            pars = [contour_param, "teff_2", red_chi2]
            dataframe_contour(df, xcol=pars[0], ycol=pars[1], zcol=pars[2], params=params)

    # Using teff_1 contour
    for par_limit, contour_param in zip(["gamma", "rv"], ["rv", "gamma"]):
        for chi2_val, npix_val in zip(chi2_names, npix_names):

            df_min_chi2 = pd.read_sql(
                sa.select(table.c).order_by(
                    table.c[chi2_val].asc()).limit(1),
                table.metadata.bind)

            df = pd.read_sql(
                sa.select([table.c["teff_2"], table.c["rv"], table.c["gamma"], table.c[chi2_val]]).where(
                    sa.and_(table.c[par_limit] == float(df_min_chi2[par_limit][0]),
                            table.c.logg_1 == float(params["logg"]),
                            table.c.feh_1 == float(params["fe_h"]),
                            table.c.logg_2 == float(params["logg"]),  # Fix companion logg
                            table.c.feh_2 == float(params["fe_h"]))),  # Fix companion fe_h
                table.metadata.bind)

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
        min_loc = np.argmin(z_grid)
        plt.plot(x_grid[min_loc], y_grid[min_loc], "r+", markersize=5)
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
        logging.warning("database_contour did not plot due to \n{0}".format(e))


def test_figure(table, params):
    chi2_val = "coadd_chi2"
    df = pd.read_sql_query(sa.select([table.c.gamma, table.c[chi2_val]]).limit(10000), table.metadata.bind)
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
    for ii, chi2_val in enumerate(chi2_names[0:-2]):
        df = pd.read_sql_query(sa.select([table.c.teff_1, table.c.logg_1, table.c.feh_1,
                                          table.c.teff_2, table.c.logg_2, table.c.feh_2,
                                          table.c.rv, table.c.gamma,
                                          table.c[chi2_val]]).order_by(table.c[chi2_val].asc()).limit(1),
                               table.metadata.bind)

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
        name = "{0}-{1}_{2}_min_chi2_spectrum_comparison_{4}.png".format(
            params["star"], params["obsnum"], params["chip"], chi2_val, params["suffix"])
        plt.savefig(os.path.join(params["path"], "plots", name))
        plt.close()

        plt.plot(obs_spec.xaxis, obs_spec.flux, label="Observation")
        plt.show()


def contrast_iam_results(table, params):
    star_name = params["star"]
    obsnum = params["obsnum"]
    __, host_params, __ = iam_helper_function(star_name, obsnum, 1)
    h_temp, h_logg, h_feh = host_params['temp'], host_params['logg'], host_params["fe_h"]
    c_temp = host_params.get("comp_temp")

    print(
        "Observation {4} - {5}\n"
        "Expected Parameters\n---------------------\n"
        "teff={0:5.0f}  logg={1:3.02f}  feh={2:4.01f} \tcompanion_temp={3:5.0f} ".format(h_temp, h_logg, h_feh, c_temp,
                                                                                         star_name, obsnum))

    print("IAM SOLUTIONS\n---------------------")
    for ii, chi2_val in enumerate(chi2_names):
        df = pd.read_sql_query(sa.select([table.c.teff_1, table.c.logg_1, table.c.feh_1,
                                          table.c.teff_2, table.c.logg_2, table.c.feh_2,
                                          table.c.gamma, table.c.rv,
                                          table.c.alpha_1, table.c.alpha_2, table.c.alpha_3, table.c.alpha_4,
                                          table.c[chi2_val]]).order_by(table.c[chi2_val].asc()).limit(1),
                               table.metadata.bind)
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
