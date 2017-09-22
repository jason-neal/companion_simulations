"""Script to load part files into a sql database.

For the inherent alpha model

We assume that the host temperature is well known so we will resitrict the
results to the host temperature. (within error bars)
For HD30501 and HD211847 this means  +- 50K so a fixed temperature.

"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sqlalchemy as sa
from utilities.debug_utils import timeit2
from utilities.param_file import parse_paramfile, get_host_params
from utilities.phoenix_utils import closest_model_params


def _parser():
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Chisquare analysis.')
    parser.add_argument('database', help='Database name.')
    parser.add_argument("-e", "--echo", help="Echo the SQL queries", action="store_true")
    # parser.add_argument('-s', '--suffix', help='Suffix to add to database name.')
    # parser.add_argument('-v', '--verbose', help='Turn on Verbose.', action="store_true")
    return parser.parse_args()


def decompose_database_name(database):
    """Database names of form */Star_obsnum_chip...db ."""
    os.path.split(database)
    path, name = os.path.split(database)
    name_split = name.split("_")
    star, obsnum = name_split[0].split("-")
    chip = name_split[1]
    return path, star, obsnum, chip


def main(database, echo=False):
    path, star, obs_num, chip = decompose_database_name(database)
    os.makedirs(os.path.join(path, "plots"), exist_ok=True)  # make dir for plots

    teff, logg, fe_h = closest_model_params(*get_host_params(star))
    params = {"path": path, "star": star, "obs_num": obs_num, "chip": chip, "teff": teff, "logg": logg, "fe_h": fe_h}

    sqlite_db = 'sqlite:///{}'.format(database)
    try:
        engine = sa.create_engine(sqlite_db, echo=echo)
        table_names = engine.table_names()
    except:
        print("\nAccessing sqlite_db = {}\n".format(sqlite_db))
        print("cwd =", os.getcwd())
        raise

    print("Table names in database =", engine.table_names())
    if len(table_names) == 1:
        tb_name = table_names[0]
    else:
        raise ValueError("Database has two many tables {}".format(table_names))

    # fix_host_parameters_reduced_gamma(engine, params, tb_name)
    # fix_host_parameters(engine, params, tb_name)

    # get_column_limits(engine, params, tb_name)
    # smallest_chi2_values(engine, params, tb_name)
    alpha_rv_contour(engine, params, tb_name)
    # test_figure(engine, params, tb_name)

    return 0


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


def alpha_rv_plot(engine, params, tb_name):
    # df = pd.read_sql(sa.text('SELECT alpha, rvs, chi2, teff_2 FROM table WHERE teff_1=:teff1, logg_1=:logg1, feh_1=:feh1'), engine, params={'teff1': 5200, 'logg1': 4.5, 'feh1': 0.0})
    # df = pd.read_sql(sa.text('SELECT alpha, rvs, chi2, teff_2 FROM table WHERE teff_1=:teff_1'), engine, params={'teff_1': 5200})
    # df = pd.read_sql(sa.text('SELECT alpha, rvs, chi2, teff_2 FROM table WHERE teff_1=5200 and logg_1=4.5 and feh_1=0.0'), engine)
    df = pd.read_sql(sa.text('SELECT alpha, rv, chi2, teff_2 FROM {0}'.format(tb_name)), engine)
    # df = pd.read_sql_query('SELECT alpha  FROM table', engine)

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
    # plt.show()


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
    name = "{0}-{1}_{2}_test_alpha_rv_contour.pdf".format(
        params["star"], params["obs_num"], params["chip"])
    plt.savefig(os.path.join(params["path"], "plots", name))
    # plt.close()
    plt.show()


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
        params["star"], params["obs_num"], params["chip"], col)
    plt.suptitle("Chi**2 Results (Fixed host): {0}-{1}_{2}".format(params["star"], params["obs_num"], params["chip"]))
    fig.savefig(os.path.join(params["path"], "plots", name))
    plt.close()


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
            df.plot(x="gamma2", y="chi2", kind="scatter", ax=axes[axis_pos[0], axis_pos[1]])  # , c="gamma2", colorbar=True)
        else:
            df.plot(x=col, y="chi2", kind="scatter", ax=axes[axis_pos[0], axis_pos[1]])  #, c="gamma", colorbar=True)

    name = "{0}-{1}_{2}_fixed_host_params.png".format(
        params["star"], params["obs_num"], params["chip"], col)
    plt.suptitle("Chi**2 Results: {0}-{1}_{2}".format(params["star"], params["obs_num"], params["chip"]))
    fig.savefig(os.path.join(params["path"], "plots", name))
    plt.close()


if __name__ == '__main__':
    args = vars(_parser())
    opts = {k: args[k] for k in args}

    sys.exit(main(**opts))
