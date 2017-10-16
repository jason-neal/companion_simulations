import os

import numpy as np

import pandas as pd
import scipy
import sqlalchemy as sa
from matplotlib import pyplot as plt
from utilities.debug_utils import timeit2

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('text', usetex=True)
rc("image", cmap="inferno")
chi2_names = ["chi2_1", "chi2_2", "chi2_3", "chi2_4", "coadd_chi2"]

def rv_plot(table, params):
    for chi2_val in ["chi2_1", "chi2_2", "chi2_3", "chi2_4", "coadd_chi2"]:
        # df = pd.read_sql(sa.text('SELECT alpha, rv, {1}, teff_2 FROM {0}'.format(tb_name, chi2)), engine)
        df = pd.read_sql(
            sa.select([table.c["rv"], table.c[chi2_val], table.c["teff_2"]]),
            table.metadata.bind)
        fig, ax = plt.subplots()
        c = ax.scatter(df["rv"], df[chi2_val], c=df["teff_2"], alpha=0.8)

        cbar = plt.colorbar(c)
        cbar.ax.set_ylabel(r"teff_2")

        ax.set_xlabel(r'RV offset', fontsize=12)
        ax.set_ylabel(r"${0}$".format(chi2_val), fontsize=12)
        ax.set_title(r'$teff_2$ (color) and companion temperature.')

        ax.grid(True)
        fig.tight_layout()
        name = "{0}-{1}_{2}_scatter_rv_{3}.pdf".format(
            params["star"], params["obs_num"], params["chip"], chi2_val)
        plt.savefig(os.path.join(params["path"], "plots", name))
        plt.savefig(os.path.join(params["path"], "plots", name.replace(".pdf", ".png")))
        plt.close()


def display_arbitary_norm_values(table, params):
    fig, axarr = plt.subplots(3)
    for ii, arbnorm in enumerate([r"arbnorm_1", r"arbnorm_2", r"arbnorm_3", r"arbnorm_4"]):
        xshift = lambda x, num: x + num * (x * 0.1)   # to separate points

        df = pd.read_sql(
            sa.select([table.c.gamma, table.c[arbnorm],
                       table.c.rv, table.c.teff_2]), table.metadata.bind)

        c = axarr[0].scatter(xshift(df["rv"], ii), df[arbnorm],
            c=df[r"teff_2"].values, alpha=0.9, label=arbnorm)

        axarr[0].set_xlabel(r'rv offset', fontsize=12)
        axarr[0].set_ylabel(r'Arbitary norm', fontsize=12)
        axarr[0].set_title(r'Arbitary normalization.')

        d = axarr[1].scatter(xshift(df[r"gamma"], ii), df[arbnorm],
            c=df[r"teff_2"].values, alpha=0.9, label=arbnorm)
        axarr[1].set_xlabel(r'$\gamma$ offset', fontsize=12)
        axarr[1].set_ylabel(r'Arbitary norm', fontsize=12)
        axarr[1].set_title(r'$\gamma$.')

        e = axarr[2].scatter(xshift(df[r"teff_2"], ii), df[arbnorm],
            c=df[r"gamma"].values, alpha=0.9, label=arbnorm)
        axarr[2].set_xlabel(r'Companion temperature', fontsize=15)
        axarr[2].set_ylabel(r'Arbitary norm ', fontsize=15)
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
    fig.suptitle("Arbitrary normalization used, \n slight shift with detetor")
    name = "{0}-{1}_{2}_plot_arbnormalization.pdf".format(
        params["star"], params["obs_num"], params["chip"])
    plt.savefig(os.path.join(params["path"], "plots", name))
    plt.savefig(os.path.join(params["path"], "plots", name.replace(".pdf", ".png")))
    plt.close()



@timeit2
def fix_host_parameters(table, params):
    print("Fixed host analysis.")
    nrows, ncols = 3, 2
    fig, axes = plt.subplots(nrows, ncols)
    fig.tight_layout()
    indices = np.arange(nrows * ncols).reshape(nrows, ncols)

    columns = ["teff_2", "logg_2", "feh_2", "gamma", "rv"]
    assert len(columns) <= (nrows * ncols)

    for ii, col in enumerate(columns):
        for jj, chi2_val in enumerate(chi2_names):
            if jj == 4:
                chi2legend = "coadd chi2"
            else:
                chi2legend = "det {}".format(jj + 1)

            df = pd.read_sql(
                    sa.select([table.c[col], table.c[chi2_val]]).where(
                        sa.and_(table.c["teff_1"] == params["teff"],
                                table.c["logg_1"] == params["logg"],
                                table.c["feh_1"] == params["fe_h"])
                        ), table.metadata.bind)

            axis_pos = [int(x) for x in np.where(indices == ii)]
            df.plot(x=col, y=chi2_val, kind="scatter", ax=axes[axis_pos[0], axis_pos[1]], label=chi2legend)  # , c="gamma", colorbar=True)

    name = "{0}-{1}_coadd_fixed_host_params_full_gamma.png".format(
        params["star"], params["obs_num"])
    plt.suptitle("Co-add Chi**2 Results (Fixed host): {0}-{1}".format(params["star"], params["obs_num"]))
    fig.savefig(os.path.join(params["path"], "plots", name))
    plt.close()


def parabola_plots(table, params, limit=None, norm=False):
    parabola_list = ["teff_2", "gamma", "rv"]
    for par in parabola_list:
        df = pd.read_sql(sa.select([table.c[par]]), table.metadata.bind)
        unique_par = list(set(df[par].values))
        unique_par.sort()
        print(unique_par)
        min_chi2 = 1
        for chi2_val in chi2_names:
            min_chi2 = []
            for unique_val in unique_par:
                df_chi2 = pd.read_sql(
                    sa.select([table.c[par], table.c[chi2_val]]).where(
                    table.c[par] == float(unique_val)).order_by(
                    table.c[chi2_val].asc()).limit(3), table.metadata.bind)
                min_chi2.append(df_chi2[chi2_val].values[0])
            #print(min_chi2)
            if norm:
                mc2 = min(min_chi2)
                min_chi2 = [c2/mc2 for c2 in min_chi2]
            plt.plot(unique_par, min_chi2, label=chi2_val)

            popt, pcov = scipy.optimize.curve_fit(parabola, unique_par, min_chi2)
            print("params", popt)
            x = np.linspace(unique_par[0], unique_par[-1], 40)
            plt.plot(x, parabola(x, *popt)) # , label="parabola")
            plt.xlabel(r"${}$".format(par))
            plt.ylabel(r"$\chi^2$")
            # if limit:
                #plt.xlim()
            #    plt.ylim([])
        plt.legend()
        filename = "Parabola_fit_{0}-{1}_{2}_param_{3}.png".format(
            params["star"], params["obs_num"], params["chip"], par)

        plt.savefig(os.path.join(params["path"], "plots", filename))
        plt.close()
        print("saved parabolas for ", par)


def smallest_chi2_values(table, params, num=10):
    """Find smallest chi2 in table."""
    chi2_val = "coadd_chi2"
    df = pd.read_sql(
        sa.select(table.c).order_by(table.c[chi2_val].asc()).limit(num),
        table.metadata.bind)

    print("Samllest Co-add Chi2 values in the database.")
    print(df.head(n=num))
    name = "{0}-{1}_{2}_test_smallest_chi2.pdf".format(
        params["star"], params["obs_num"], params["chip"])
    # plt.savefig(os.path.join(params["path"], "plots", name))
    # plt.close()
    # plt.show()


def parabola(x, a, b, c):
    return a * x**2 + b * x + c


@timeit2
def fix_host_parameters_reduced_gamma(table, params):
    print("Fixed host analysis with reduced gamma.")
    d_gamma = 5
    for jj, chi2_val in enumerate(chi2_names):
        if jj == 4:
            chi2legend = "coadd chi2"
        else:
            chi2legend = "det {}".format(jj + 1)

        # Select lowest chisqr gamma values.
        df = pd.read_sql(
            sa.select([table.c.gamma, table.c[chi2_val]]
            ).order_by(table.c[chi2_val].asc()
            ).limit(1),
            table.metadata.bind)

        min_chi2_gamma = df.loc[0, "gamma"]
        upper_lim = min_chi2_gamma + d_gamma
        lower_lim = min_chi2_gamma - d_gamma
        print("Reduced gamma_limits", lower_lim, upper_lim)

        nrows, ncols = 3, 2
        fig, axes = plt.subplots(nrows, ncols)
        fig.tight_layout()
        indices = np.arange(nrows * ncols).reshape(nrows, ncols)

        columns = ["teff_2", "logg_2", "feh_2", "gamma", "rv"]
        assert len(columns) <= (nrows * ncols)

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
            # print("head", df.head())

            axis_pos = [int(x) for x in np.where(indices == ii)]

            df.plot(x=col, y=chi2_val, kind="scatter",
                ax=axes[axis_pos[0], axis_pos[1]], label=chi2legend)  # , c="gamma", colorbar=True)

    name = "{0}-{1}_coadd_fixed_host_params.png".format(
        params["star"], params["obs_num"])
    plt.suptitle("Coadd Chi**2 Results: {0}-{1}".format(params["star"], params["obs_num"]))
    fig.savefig(os.path.join(params["path"], "plots", name))
    plt.close()


def get_column_limits(table, params):
    print("Database Column Value Ranges")
    for col in ["teff_1", "teff_2", "logg_1", "logg_2", "gamma", "rv",
            "chi2_1", "chi2_2", "chi2_3", "chi2_4", "coadd_chi2", "alpha_1",
            "alpha_2", "alpha_3", "alpha_4", "arbnorm_1", "arbnorm_2",
            "arbnorm_3", "arbnorm_4"]:
        #query = """
        #       SELECT * FROM (SELECT {1} FROM {0} ORDER BY {1} ASC LIMIT 1)
        #       UNION ALL
        #       SELECT * FROM (SELECT {1} FROM {0} ORDER BY {1} DESC LIMIT 1)
        #       """.format(tb_name, col)
        # df = pd.read_sql(sa.text(query), engine)
        min_df = pd.read_sql(
            sa.select([table.c[col]]).order_by(table.c[col].asc()).limit(1),
            table.metadata.bind)
        max_df = pd.read_sql(
            sa.select([table.c[col]]).order_by(table.c[col].desc()).limit(1),
            table.metadata.bind)
        print(col, min_df[col].values[0], max_df[col].values[0])


def contours(table, params):

    for chi2_val in  ["chi2_1", "chi2_2", "chi2_3", "chi2_4", "coadd_chi2"]:
        # df_min_chi2 = pd.read_sql(sa.text('SELECT * FROM {0} ORDER BY chi2 ASC LIMIT 1'.format(tb_name)), engine)
        df_min_chi2 = pd.read_sql(
            sa.select(table.c
            ).order_by(table.c[chi2_val].asc()
            ).limit(1),
            table.metadata.bind)

        print("columns", df_min_chi2.columns)
        pars = ["teff_2", "rv", chi2_val]
        cols = ['teff_2', 'rv', 'gamma', chi2_val]
        par_limit = "gamma"    # gamma value at minimum chi2
        print("df_min_chi2[par_limit]", df_min_chi2[par_limit].values[0])

        df = pd.read_sql(
            sa.select([table.c["teff_2"], table.c["rv"], table.c["gamma"], table.c[chi2_val]]
            ).where(sa.and_(
                table.c[par_limit] == float(df_min_chi2[par_limit][0]),
                table.c.teff_1 == int(params["teff"]),
                table.c.logg_1 == float(params["logg"]),
                table.c.feh_1 == float(params["fe_h"]),
                table.c.logg_2 == float(params["logg"]),  # Fix companion logg
                table.c.feh_2 == float(params["fe_h"]))   # Fix companion fe_h
            ), table.metadata.bind)

        print(df.head())

        for col in cols:
            if col is not chi2_val:
                print("unique {}".format(col), set(df[col].values), "length=", len(list(set(df[col].values))))

        dataframe_contour(df, xcol=pars[0], ycol=pars[1], zcol=pars[2], params=params)


def dataframe_contour(df, xcol, ycol, zcol, params):
    x = sorted(np.array(list(set(df[xcol].values))))
    y = sorted(np.array(list(set(df[ycol].values))))

    # Create grid for chi2 values
    Z = np.empty((len(x), len(y)))
    for i, xval in enumerate(x):
        for j, yval in enumerate(y):
            try:
                Z[i, j] = df.loc[(df[xcol].values == xval) * (df[ycol].values == yval), zcol].values
            except ValueError as e:
                print("x_S * y_s", sum((df[xcol].values == xval)*(df[ycol].values == yval)))
                print("Check metalicity and logg of companion")
                raise e

    X, Y = np.meshgrid(x, y, indexing='ij')

    print(X, Y, Z)
    # print("shapes", X.shape, Y.shape, Z.shape)
    assert X.shape == Z.shape
    assert X.shape == Y.shape

    fig, ax = plt.subplots()
    c = ax.contourf(X, Y, Z, alpha=0.5, cmap=plt.cm.inferno)
    cbar = plt.colorbar(c)
    cbar.ax.set_ylabel(zcol)
    # plt.clabel(pars[2])
    ax.set_xlabel(r"$ {0}$".format(xcol), fontsize=15)
    ax.set_ylabel(r"$ {0}$".format(ycol), fontsize=15)
    ax.set_title('{0}: {1} contour'.format(params["star"], zcol))

    ax.grid(True)
    fig.tight_layout()
    name = "{0}-{1}_{2}_{3}_{4}_{5}_contour.pdf".format(
        params["star"], params["obs_num"], params["chip"], xcol, ycol, zcol)
    plt.savefig(os.path.join(params["path"], "plots", name))
    plt.savefig(os.path.join(params["path"], "plots", name.replace(".pdf", ".png")))
    # plt.show()
    plt.close()

def test_figure(table, params):
    chi2_val = "coadd_chi2"
    #df = pd.read_sql_query('SELECT alpha, chi2 FROM {0} LIMIT 10000'.format(tb_name), engine)
    df = pd.read_sql_query(sa.select([table.c.gamma, table.c[chi2_val]]).limit(10000), table.metadata.bind)
    fig, ax = plt.subplots()
    ax.scatter(df["gamma"], df[chi2_val], s=3, alpha=0.5)

    ax.set_xlabel(r'$\gamma$', fontsize=15)
    ax.set_ylabel(r"$ {0}$".format(chi2_val), fontsize=15)
    ax.set_title(r'$\gamma$ and $\chi^2$.')

    ax.grid(True)
    fig.tight_layout()
    name = "{0}-{1}_{2}_test_test_figure1_{3}.pdf".format(
        params["star"], params["obs_num"], params["chip"], chi2_val)
    plt.savefig(os.path.join(params["path"], "plots", name))
    # plt.show()
    plt.close()


































############################################
# Oolder engine code


@timeit2
def fix_host_parameters_reduced_gamma_engine(engine, params, tb_name):
    print("Fixed host analysis with reduced gamma.")
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
    assert len(columns) <= (nrows * ncols)

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





@timeit2
def fix_host_parameters_engine(engine, params, tb_name):
    # "older version"
    print("Fixed host analysis.")
    nrows, ncols = 3, 2
    fig, axes = plt.subplots(nrows, ncols)
    # fig.subplots_adjust(hspace=.5, vspace=0.5)
    fig.tight_layout()
    # print("axes", axes)
    indices = np.arange(nrows * ncols).reshape(nrows, ncols)
    # print("indicies", indices)

    columns = ["teff_2", "logg_2", "feh_2", "alpha", "gamma", "rv"]
    assert len(columns) <= (nrows * ncols)

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


def get_column_limits_engine(engine, params, tb_name):
    print("Column Value Ranges")
    for col in ["teff_1", "teff_2", "logg_1", "logg_2", "gamma", "rv", "chi2"]:
        query = """
               SELECT * FROM (SELECT {1} FROM {0} ORDER BY {1} ASC LIMIT 1)
               UNION ALL
               SELECT * FROM (SELECT {1} FROM {0} ORDER BY {1} DESC LIMIT 1)
               """.format(tb_name, col)
        df = pd.read_sql(sa.text(query), engine)
        print(col, min(df[col]), max(df[col]))



def alpha_rv_plot_engine(engine, params, tb_name):
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



def get_column_limits_engine(engine, params, tb_name):
    print("Column Value Ranges")
    for col in ["teff_1", "teff_2", "logg_1", "logg_2", "gamma", "rv", "chi2"]:
        query = """
               SELECT * FROM (SELECT {1} FROM {0} ORDER BY {1} ASC LIMIT 1)
               UNION ALL
               SELECT * FROM (SELECT {1} FROM {0} ORDER BY {1} DESC LIMIT 1)
               """.format(tb_name, col)
        df = pd.read_sql(sa.text(query), engine)
        print(col, min(df[col]), max(df[col]))


def alpha_rv_contour_engine(engine, params, tb_name):
    df_min_chi2 = pd.read_sql(sa.text('SELECT * FROM {0} ORDER BY chi2 ASC LIMIT 1'.format(tb_name)), engine)
    print("Need to check host parameters")

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
    plt.close()

def test_figure_engine(engine, params, tb_name):
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
