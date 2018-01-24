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

from mingle.models.broadcasted_models import one_comp_model
from mingle.utilities.chisqr import reduced_chi_squared
from mingle.utilities.crires_utilities import barycorr_crires_spectrum
from mingle.utilities.debug_utils import timeit2
from mingle.utilities.phoenix_utils import load_starfish_spectrum
from mingle.utilities.spectrum_utils import load_spectrum
from simulators.bhm_module import bhm_helper_function


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


def gamma_plot(table, params):
    for chi2_val, npix_val in zip(chi2_names, npix_names):
        df = pd.read_sql(
            sa.select([table.c["gamma"], table.c[chi2_val], table.c["teff_1"]]),
            table.metadata.bind)
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


def display_arbitrary_norm_values(table, params):
    fig, axarr = plt.subplots(3)
    for ii, arbnorm in enumerate([r"arbnorm_1", r"arbnorm_2", r"arbnorm_3", r"arbnorm_4"]):
        df = pd.read_sql(
            sa.select([table.c.gamma, table.c[arbnorm],
                       table.c.teff_1]), table.metadata.bind)

        c = axarr[0].scatter(xshift(df["gamma"], ii), df[arbnorm],
                             c=df[r"teff_1"].values, alpha=0.9, label=arbnorm)

        axarr[0].set_xlabel(r'host rv offset', fontsize=12)
        axarr[0].set_ylabel(r'Arbitrary norm', fontsize=12)
        axarr[0].set_title(r'Arbitrary normalization.')

        d = axarr[1].scatter(xshift(df[r"gamma"], ii), df[arbnorm],
                             c=df[r"teff_1"].values, alpha=0.9, label=arbnorm)
        axarr[1].set_xlabel(r'$\gamma$ offset', fontsize=12)
        axarr[1].set_ylabel(r'Arbitrary norm', fontsize=12)
        axarr[1].set_title(r'$\gamma$.')

        e = axarr[2].scatter(xshift(df[r"teff_1"], ii), df[arbnorm],
                             c=df[r"gamma"].values, alpha=0.9, label=arbnorm)
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
    name = "{0}-{1}_{2}_plot_arbnormalization_{3}.pdf".format(
        params["star"], params["obsnum"], params["chip"], params["suffix"])
    plt.savefig(os.path.join(params["path"], "plots", name))
    plt.savefig(os.path.join(params["path"], "plots", name.replace(".pdf", ".png")))
    plt.close()


def display_bhm_xcorr_values(table, params):
    fig, axarr = plt.subplots(3)
    for ii, xcorr in enumerate([r"xcorr_1", r"xcorr_2", r"xcorr_3", r"xcorr_4"]):
        df = pd.read_sql(
            sa.select([table.c.gamma, table.c[xcorr],
                       table.c.teff_1]), table.metadata.bind)

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
            df = pd.read_sql(
                sa.select([table.c[col], table.c[chi2_val]]).where(
                    sa.and_(table.c["logg_1"] == params["logg"],
                            table.c["feh_1"] == params["fe_h"])
                ), table.metadata.bind)

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
            df = pd.read_sql(
                sa.select([table.c[col], table.c[chi2_val]]).where(
                    sa.and_(table.c["logg_1"] == params["logg"],
                            table.c["feh_1"] == params["fe_h"])
                ), table.metadata.bind)
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
            df = pd.read_sql(
                sa.select([table.c[col], table.c[chi2_val]]).where(
                    sa.and_(table.c["logg_1"] == params["logg"],
                            table.c["feh_1"] == params["fe_h"])
                ), table.metadata.bind)
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
    parabola_list = ["teff_1", "logg_1", "feh_1", "gamma"]
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

            min_chi2 = reduced_chi_squared(min_chi2, params["npix"][npix_val], params["npars"])

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


def chi2_parabola_plots(table, params):
    parabola_list = ["teff_1", "logg_1", "feh_1", "gamma"]
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

            min_chi2 = reduced_chi_squared(min_chi2, params["npix"][npix_val], params["npars"])

            min_chi2 = min_chi2 - min(min_chi2)

            plt.plot(unique_par, min_chi2, ".-", label=chi2_val)

            popt, pcov = curve_fit(parabola, unique_par, min_chi2)
            print("params", popt)
            x = np.linspace(unique_par[0], unique_par[-1], 40)
            plt.plot(x, parabola(x, *popt))  # , label="parabola")
            plt.xlabel(r"${0}$".format(par))
            plt.ylabel(r"$\Delta \chi^2_{red}$ from mimimum")

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
        print("saved chi2 parabolas for ", par)
    plt.close()


def smallest_chi2_values(table, params, num=10):
    """Find smallest chi2 in table."""
    chi2_val = "chi2_1"  # "coadd_chi2"
    df = pd.read_sql(
        sa.select(table.c).order_by(table.c[chi2_val].asc()).limit(num),
        table.metadata.bind)
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


def parabola(x, a, b, c):
    return a * x ** 2 + b * x + c


@timeit2
def host_parameters_reduced_gamma(table, params):
    print("Fixed host analysis with reduced gamma.")
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
        df = pd.read_sql(
            sa.select([table.c.gamma, table.c[chi2_val]]).order_by(
                table.c[chi2_val].asc()).limit(1),
            table.metadata.bind)

        min_chi2_gamma = df.loc[0, "gamma"]
        upper_lim = min_chi2_gamma + d_gamma
        lower_lim = min_chi2_gamma - d_gamma

        columns = ["teff_1", "logg_1", "feh_1", "gamma", ]
        assert len(columns) <= (nrows * ncols)

        for ii, col in enumerate(columns):
            df = pd.read_sql(
                sa.select([table.c[col], table.c[chi2_val], table.c.gamma, table.c.teff_1], table.c.teff_1).where(
                    sa.and_(table.c.gamma > float(lower_lim),
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

        columns = ["teff_1", "logg_1", "feh_1", "gamma", ]
        assert len(columns) <= (nrows * ncols)

        for ii, col in enumerate(columns):
            df = pd.read_sql(
                sa.select([table.c[col], table.c[chi2_val], table.c.gamma, table.c.teff_1], table.c.teff_1).where(
                    sa.and_(table.c.gamma > float(lower_lim),
                            table.c.gamma < float(upper_lim)
                            )
                ), table.metadata.bind)

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
        columns = ["teff_1", "logg_1", "feh_1", "gamma"]

        for ii, col in enumerate(columns):
            df = pd.read_sql(
                sa.select([table.c[col], table.c[chi2_val], table.c.gamma, table.c.teff_1], table.c.teff_1).where(
                    sa.and_(table.c.gamma > float(lower_lim),
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
    for col in ["teff_1", "logg_1", "feh_1", "gamma",
                "chi2_1", "chi2_2", "chi2_3", "chi2_4", "coadd_chi2", "arbnorm_1", "arbnorm_2",
                "arbnorm_3", "arbnorm_4", "xcorr_1", "xcorr_2",
                "xcorr_3", "xcorr_4"]:
        min_df = pd.read_sql(
            sa.select([table.c[col]]).order_by(table.c[col].asc()).limit(1),
            table.metadata.bind)
        max_df = pd.read_sql(
            sa.select([table.c[col]]).order_by(table.c[col].desc()).limit(1),
            table.metadata.bind)
        print("{0:10}\t\t{1:5.3} - {2:5.3}".format(col, float(min_df[col].values[0]), float(max_df[col].values[0])))


def contours(table, params):
    for chi2_val, npix_val in zip(chi2_names, npix_names):
        df_min_chi2 = pd.read_sql(
            sa.select(table.c).order_by(
                table.c[chi2_val].asc()).limit(1),
            table.metadata.bind)
        df = pd.read_sql(
            sa.select(
                [table.c["teff_1"], table.c["feh_1"], table.c["logg_1"], table.c["gamma"], table.c[chi2_val]]).where(
                sa.and_(table.c["logg_1"] == float(df_min_chi2["logg_1"][0]),
                        table.c["feh_1"] == float(df_min_chi2["feh_1"][0]))),
            table.metadata.bind)
        params["this_npix"] = params["npix"][npix_val]
        params["chi2_value"] = chi2_val
        pars = ["gamma", "teff_1", chi2_val]
        dataframe_contour(df, xcol=pars[0], ycol=pars[1], zcol=pars[2], params=params)

    # Reduced chi2
    for chi2_val, npix_val in zip(chi2_names, npix_names):
        red_chi2 = "red_{0}".format(chi2_val)
        df_min_chi2 = pd.read_sql(
            sa.select(table.c).order_by(
                table.c[chi2_val].asc()).limit(1),
            table.metadata.bind)
        df = pd.read_sql(
            sa.select(
                [table.c["teff_1"], table.c["feh_1"], table.c["logg_1"], table.c["gamma"], table.c[chi2_val]]).where(
                sa.and_(table.c["logg_1"] == float(df_min_chi2["logg_1"][0]),
                        table.c["feh_1"] == float(df_min_chi2["feh_1"][0]))),
            table.metadata.bind)
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
        min_loc = np.argmin(z_grid)
        plt.plot(x_grid[min_loc], y_grid[min_loc], "r+", markersize=5)
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
    gamma_df = pd.read_sql_query(sa.select([table.c.gamma]), table.metadata.bind)
    extreme_gammas = [min(gamma_df.gamma.values), max(gamma_df.gamma.values)]
    for ii, chi2_val in enumerate(chi2_names[0:-2]):
        df = pd.read_sql_query(sa.select([table.c.teff_1, table.c.logg_1, table.c.feh_1,
                                          table.c.gamma,
                                          table.c[chi2_val]]).order_by(table.c[chi2_val].asc()).limit(1),
                               table.metadata.bind)

        params1 = [df["teff_1"].values[0], df["logg_1"].values[0], df["feh_1"].values[0]]

        params1 = [float(param1) for param1 in params1]

        gamma = df["gamma"].values

        from simulators.bhm_module import bhm_helper_function
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
    star_name = params["star"]
    obsnum = params["obsnum"]
    __, host_params, __ = bhm_helper_function(star_name, obsnum, 1)
    h_temp, h_logg, h_feh = host_params['temp'], host_params['logg'], host_params["fe_h"]

    print("Expected Parameters\n---------------------\nteff={0}\tlogg={1}\tfeh={2}".format(h_temp, h_logg, h_feh))
    print("BHM SOLUTIONS\n---------------------")
    for ii, chi2_val in enumerate(chi2_names):
        df = pd.read_sql_query(sa.select([table.c.teff_1, table.c.logg_1, table.c.feh_1,
                                          table.c.gamma, table.c.xcorr_1,
                                          table.c[chi2_val]]).order_by(table.c[chi2_val].asc()).limit(1),
                               table.metadata.bind)
        print("{0}: teff={1:5}\tlogg={2:3.02}\t".format(chi2_val, df.teff_1.values[0], df.logg_1.values[0]) +
              "feh={0:4.1}\tgamma={1:3.2},\txcorr={2:3.2},\tchi2={3:8.03}".format(df.feh_1.values[0],
                                                                                  float(df.gamma.values[0]),
                                                                                  df.xcorr_1.values[0],
                                                                                  df[chi2_val].values[0]))
