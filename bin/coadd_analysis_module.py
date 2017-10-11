import os

import numpy as np
import pandas as pd
import scipy
import sqlalchemy as sa
from matplotlib import pyplot as plt

from utilities.debug_utils import timeit2


def alpha_rv_plot(engine, params, tb_name):
    for chi2 in ["chi2_1", "chi2_2", "chi2_3", "chi2_4", "coadd_chi2"]:
        df = pd.read_sql(sa.text('SELECT alpha, rv, {1}, teff_2 FROM {0}'.format(tb_name, chi2)), engine)

        fig, ax = plt.subplots()
        ax.scatter(df["rv"], df["chi2"], c=df["alpha"], s=df["teff_2"] / 50, alpha=0.5)

        ax.set_xlabel('rv offset', fontsize=15)
        ax.set_ylabel('chi2', fontsize=15)
        ax.set_title('alpha (color) and companion temperature (size=Temp/50).')

        ax.grid(True)
        fig.tight_layout()
        name = "{0}-{1}_{2}_test_alpha_rv.pdf".format(
            params["star"], params["obs_num"], params["chip"])
        plt.savefig(os.path.join(params["path"], "plots", name))
        plt.close()


def display_arbitary_norm_values(table, params):
    for arbnorm in ["arbnorm_1", "arbnorm_2", "arbnorm_3", "arbnorm_4", "arbnorm_4"]:
        df = pd.read_sql(sa.text('SELECT alpha, rv, {1}, teff_2 FROM {0}'.format(tb_name, arbnorm)), engine)

        fig, ax = plt.subplots()
        ax.scatter(df["rv"], df[arbnorm], c=df["teff_2"] / 50, alpha=0.5, legend=arbnorm)

        ax.set_xlabel('rv offset', fontsize=15)
        ax.set_ylabel('chi2', fontsize=15)
        ax.set_title('alpha (color) and companion temperature (size=Temp/50).')

        ax.grid(True)
        fig.tight_layout()
        name = "{0}-{1}_{2}_plot_{3}.pdf".format(
            params["star"], params["obs_num"], params["chip"], arbnorm)
        plt.savefig(os.path.join(params["path"], "plots", name))
        plt.close()


@timeit2
def fix_host_parameters(engine, params, tb_name):
    print("Fixed host analysis.")
    nrows, ncols = 3, 2
    fig, axes = plt.subplots(nrows, ncols)
    # fig.subplots_adjust(hspace=.5, vspace=0.5)
    fig.tight_layout()
    # print("axes", axes)
    indices = np.arange(nrows * ncols).reshape(nrows, ncols)
    # print("indicies", indices)

    columns = ["teff_2", "logg_2", "feh_2", "alpha", "gamma", "rv"]
    assert len(columns) == (nrows * ncols)

    for ii, col in enumerate(columns):
        query = """SELECT {}, {} FROM {} WHERE (teff_1 = {}  AND logg_1 = {} AND feh_1 = {})""" .format(
            col, "chi2", tb_name, params["teff"], params["logg"], params["fe_h"])
        df = pd.read_sql(sa.text(query), engine)
        # print(df.columns)

        axis_pos = [int(x) for x in np.where(indices == ii)]
        df.plot(x=col, y="chi2", kind="scatter", ax=axes[axis_pos[0], axis_pos[1]])  # , c="gamma", colorbar=True)

    name = "{0}-{1}_{2}_fixed_host_params_full_gamma.png".format(
        params["star"], params["obs_num"], params["chip"])
    plt.suptitle("Chi**2 Results (Fixed host): {0}-{1}_{2}".format(params["star"], params["obs_num"], params["chip"]))
    fig.savefig(os.path.join(params["path"], "plots", name))
    plt.close()


def parabola_plots(table, params):
    parabola_list = ["teff_2", "gamma", "rv"]
    for par in parabola_list:
        df = pd.read_sql(sa.select([table.c[par]]), table.metadata.bind)
        unique_par = list(set(df[par].values))
        unique_par.sort()
        print(unique_par)

        min_chi2 = []
        for unique_val in unique_par:
            df_chi2 = pd.read_sql(
                sa.select([table.c[par], table.c.chi2]).where(table.c[par] == float(unique_val)).order_by(table.c.chi2.asc()).limit(3), table.metadata.bind)
            min_chi2.append(df_chi2.chi2.values[0])
        print(min_chi2)
        plt.plot(unique_par, min_chi2)

        popt, pcov = scipy.optimize.curve_fit(parabola, unique_par, min_chi2)
        print("params", popt)
        x = np.linspace(unique_par[0], unique_par[-1], 40)
        plt.plot(x, parabola(x, *popt), label="parabola")
        plt.xlabel("{}".format(par))
        plt.ylabel("Chi2")
        filename = "Parabola_fit_{0}-{1}_{2}_param_{3}.png".format(
            params["star"], params["obs_num"], params["chip"], par)

        plt.savefig(os.path.join(params["path"], "plots", filename))
        plt.close()
        print("saved parabolas for ", par)


def smallest_chi2_values(engine, params, tb_name):
    """Find smallest chi2 in table."""
    df = pd.read_sql(sa.text('SELECT * FROM {0} ORDER BY chi2 ASC LIMIT 10'.format(tb_name)), engine)
    # df = pd.read_sql_query('SELECT alpha  FROM table', engine)

    print("Samllest Chi2 values in the database.")
    print(df.head(n=15))
    name = "{0}-{1}_{2}_test_smallest_chi2.pdf".format(
        params["star"], params["obs_num"], params["chip"])
    plt.savefig(os.path.join(params["path"], "plots", name))
    plt.close()
    # plt.show()


def parabola(x, a, b, c):
    return a * x**2 + b * x + c


@timeit2
def fix_host_parameters_reduced_gamma(engine, params, tb_name):
    print("Fixed host analysis.")
    d_gamma = 5
    # Select lowest chisqr gamma values.
    query = """SELECT {0}, {1} FROM {2} ORDER BY {1} ASC LIMIT 1""" .format(
        "gamma", "chi2", tb_name)
    df = pd.read_sql(sa.text(query), engine)
    min_chi2_gamma = df.loc[0, "gamma"]
    print("min_chi2_gamma = ", min_chi2_gamma)
    upper_lim = min_chi2_gamma + d_gamma
    lower_lim = min_chi2_gamma - d_gamma
    print("gamma_limits", lower_lim, upper_lim)

    nrows, ncols = 3, 2
    fig, axes = plt.subplots(nrows, ncols)
    # fig.subplots_adjust(hspace=.5, vspace=0.5)
    fig.tight_layout()
    # print("axes", axes)
    indices = np.arange(nrows * ncols).reshape(nrows, ncols)
    # print("indicies", indices)

    columns = ["teff_2", "logg_2", "feh_2", "alpha", "gamma", "rv"]
    assert len(columns) == (nrows * ncols)

    for ii, col in enumerate(columns):
        query = """SELECT {0}, {1}, gamma FROM {2} WHERE (teff_1 = {3}  AND logg_1 = {4} AND
            feh_1 = {5} AND gamma > {6} AND gamma < {7})""".format(
            col, "chi2", tb_name, params["teff"], params["logg"], params["fe_h"],
            lower_lim, upper_lim)

        df = pd.read_sql(sa.text(query), engine)
        # print(df.columns)
        # print(df.dtypes)

        # plt.subplot(3, 2, ii + 1)
        # if col == "gamma":   # Duplicate columns
        #    df["gamma2"] = df.gamma.iloc[:, 0]
        #    df.plot(ax=axes.ravel()[ii]).scatter("gamma2", "chi2")  #, c="gamma2", colorbar=True)
        # else:
        #    df.plot(ax=axes.ravel()[ii]).scatter(col, "chi2")  #, c="gamma", colorbar=True)
        axis_pos = [int(x) for x in np.where(indices == ii)]
        if col == "gamma":   # Duplicate columns
            df["gamma2"] = df.gamma.iloc[:, 0]
            df.plot(x="gamma2", y="chi2", kind="scatter",
                    ax=axes[axis_pos[0], axis_pos[1]])  # , c="gamma2", colorbar=True)
        else:
            df.plot(x=col, y="chi2", kind="scatter",
                    ax=axes[axis_pos[0], axis_pos[1]])  # , c="gamma", colorbar=True)

    name = "{0}-{1}_{2}_fixed_host_params.png".format(
        params["star"], params["obs_num"], params["chip"], col)
    plt.suptitle("Chi**2 Results: {0}-{1}_{2}".format(params["star"], params["obs_num"], params["chip"]))
    fig.savefig(os.path.join(params["path"], "plots", name))
    plt.close()


def get_column_limits(engine, params, tb_name):
    print("Column Value Ranges")
    for col in ["teff_1", "teff_2", "logg_1", "logg_2", "alpha", "gamma", "rv", "chi2"]:
        query = """
               SELECT * FROM (SELECT {1} FROM {0} ORDER BY {1} ASC LIMIT 1)
               UNION ALL
               SELECT * FROM (SELECT {1} FROM {0} ORDER BY {1} DESC LIMIT 1)
               """.format(tb_name, col)
        df = pd.read_sql(sa.text(query), engine)
        print(col, min(df[col]), max(df[col]))


def alpha_rv_contour(engine, params, tb_name):
    df_min_chi2 = pd.read_sql(sa.text('SELECT * FROM {0} ORDER BY chi2 ASC LIMIT 1'.format(tb_name)), engine)
    print("Need to check host parameters")
    db_names = {}
    print("columns", df_min_chi2.columns)
    pars = ["teff_2", "rv", "chi2"]
    cols = ['teff_2', 'alpha', 'rv', 'gamma', 'chi2']
    par_limit = "gamma"    # gamma value at minimum chi2
    print("df_min_chi2[par_limit]", df_min_chi2[par_limit].values)
    df = pd.read_sql(sa.text((
        "SELECT teff_2, alpha, rv, gamma, chi2 "
        "FROM {0} "
        "WHERE ({1} = {2} AND teff_1 = {3} AND logg_1 = {4} AND feh_1 = {5})").format(tb_name, par_limit, df_min_chi2[par_limit][0], params["teff"], params["logg"], params["fe_h"])), engine)

    print("len df", len(df))
    print("columns", df.columns)

    for col in cols:
        if col is not "chi2":
            print("unique {}".format(col), set(df[col].values), "length=", len(list(set(df[col].values))))
    # make_arrays
    x = np.array(list(set(df[pars[0]].values)))
    x.sort()
    print("x", x)
    y = np.array(list(set(df[pars[1]].values)))
    y.sort()
    print("y", y)

    # Create array for chi2 values
    Z = np.empty((len(x), len(y)))
    for i, xval in enumerate(x):
        for j, yval in enumerate(y):
            Z[i, j] = df.loc[(df[pars[0]].values == xval) * (df[pars[1]].values == yval), "chi2"].values


    X, Y = np.meshgrid(x, y, indexing='ij')

    print(X, Y, Z)
    print("shapes", X.shape, Y.shape, Z.shape)
    print(len(x), len(y), "lenx*leny", len(x) * len(y))
    fig, ax = plt.subplots()
    c = ax.contourf(X, Y, Z, alpha=0.5)
    # c.colorbar()
    # plt.clabel(pars[2])
    ax.set_xlabel(pars[0], fontsize=15)
    ax.set_ylabel(pars[1], fontsize=15)
    ax.set_title('Chi2 map for {}'.format(params["star"]))

    ax.grid(True)
    fig.tight_layout()
    name = "{0}-{1}_{2}_test_alpha_rv_contour1.pdf".format(
        params["star"], params["obs_num"], params["chip"])
    plt.savefig(os.path.join(params["path"], "plots", name))
    # plt.close()
    plt.show()


def alpha_rv_contour_old(engine, params, tb_name):
    df = pd.read_sql(
        sa.text('SELECT alpha, rv, chi2, teff_2 FROM {0}'.format(
            tb_name)), engine)

    fig, ax = plt.subplots()
    ax.contourf(df["rv"], df["chi2"], c=df["alpha"], alpha=0.5)

    ax.set_xlabel('rv offset', fontsize=15)
    ax.set_ylabel('chi2', fontsize=15)
    ax.set_title('alpha (color)')

    ax.grid(True)
    fig.tight_layout()
    name = "{0}-{1}_{2}_test_alpha_rv_contour2.pdf".format(
        params["star"], params["obs_num"], params["chip"])
    plt.savefig(os.path.join(params["path"], "plots", name))
    plt.close()
    # plt.show()


def test_figure(engine, params, tb_name):
    df = pd.read_sql_query('SELECT alpha, chi2 FROM {0} LIMIT 10000'.format(tb_name), engine)

    fig, ax = plt.subplots()
    ax.scatter(df["alpha"], df["chi2"], s=3, alpha=0.5)

    ax.set_xlabel('alpha', fontsize=15)
    ax.set_ylabel('chi2', fontsize=15)
    ax.set_title('alpha and Chi2.')

    ax.grid(True)
    fig.tight_layout()
    name = "{0}-{1}_{2}_test_test_figure1.pdf".format(
        params["star"], params["obs_num"], params["chip"])
    plt.savefig(os.path.join(params["path"], "plots", name))
    # plt.show()

    alpha_rv_plot(engine, params, tb_name)
    # alpha_rv_contour(engine, tb_name)
