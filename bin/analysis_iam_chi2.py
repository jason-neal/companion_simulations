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
import pandas as pd
import sqlalchemy as sa

from utilities.param_file import parse_paramfile
from utilities.phoenix_utils import closest_model_params
from utilities.debug_utils import timeit2


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


def get_host_params(star):
    """Find host star parameters from param file."""
    param_file = "/home/jneal/Phd/data/parameter_files/{}_params.dat".format(star)
    params = parse_paramfile(param_file)
    return params["temp"], params["logg"], params["fe_h"]


def main(database, echo=False):
    path, star, obs_num, chip = decompose_database_name(database)

    teff, logg, fe_h = closest_model_params(*get_host_params(star))
    params = {"path": path, "star": star, "obs_num": obs_num, "chip": chip, "teff": teff, "logg": logg, "fe_h": fe_h}

    engine = sa.create_engine('sqlite:///{}'.format(database), echo=echo)
    table_names = engine.table_names()
    print("Table names in database =", engine.table_names())
    if len(table_names) == 1:
        tb_name = table_names[0]
    else:
        raise ValueError("Database has two many tables {}".format(table_names))

    fix_host_parameters_reduced_gamma(engine, params, tb_name)
    # fix_host_parameters(engine, params, tb_name)

    # get_column_limits(engine, params, tb_name)
    # smallest_chi2_values(engine, params, tb_name)

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
    #$plt.show()

    alpha_rv_plot(engine, params, tb_name)
    # alpha_rv_contour(engine, tb_name)


def alpha_rv_plot(engine, params, tb_name):
    # df = pd.read_sql(sa.text('SELECT alpha, rvs, chi2, teff_2 FROM table WHERE teff_1=:teff1, logg_1=:logg1, feh_1=:feh1'), engine, params={'teff1': 5200, 'logg1': 4.5, 'feh1': 0.0})
    #df = pd.read_sql(sa.text('SELECT alpha, rvs, chi2, teff_2 FROM table WHERE teff_1=:teff_1'), engine, params={'teff_1': 5200})
    #df = pd.read_sql(sa.text('SELECT alpha, rvs, chi2, teff_2 FROM table WHERE teff_1=5200 and logg_1=4.5 and feh_1=0.0'), engine)
    df = pd.read_sql(sa.text('SELECT alpha, rv, chi2, teff_2 FROM {0}'.format(tb_name)), engine)
    #df = pd.read_sql_query('SELECT alpha  FROM table', engine)

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

    # plt.show()


def alpha_rv_contour(engine, params, tb_name):
    # df = pd.read_sql(sa.text('SELECT alpha, rvs, chi2, teff_2 FROM table WHERE teff_1=:teff1, logg_1=:logg1, feh_1=:feh1'), engine, params={'teff1': 5200, 'logg1': 4.5, 'feh1': 0.0})
    #df = pd.read_sql(sa.text('SELECT alpha, rvs, chi2, teff_2 FROM table WHERE teff_1=:teff_1'), engine, params={'teff_1': 5200})
    #df = pd.read_sql(sa.text('SELECT alpha, rvs, chi2, teff_2 FROM table WHERE teff_1=5200 and logg_1=4.5 and feh_1=0.0'), engine)
    df = pd.read_sql(sa.text('SELECT alpha, rv, chi2, teff_2 FROM {0}'.format(tb_name)), engine)
    #df = pd.read_sql_query('SELECT alpha  FROM table', engine)

    fig, ax = plt.subplots()
    ax.contourf(df["rv"], df["chi2"], c=df["alpha"], alpha=0.5)

    ax.set_xlabel('rv offset', fontsize=15)
    ax.set_ylabel('chi2', fontsize=15)
    ax.set_title('alpha (color)')

    ax.grid(True)
    fig.tight_layout()
    name = "{0}-{1}_{2}_test_alpha_rv_contour.pdf".format(
        params["star"], params["obs_num"], params["chip"])
    plt.savefig(os.path.join(params["path"], "plots", name))
    # plt.show()


def smallest_chi2_values(engine, params, tb_name):
    """Find smallest chi2 in table."""
    df = pd.read_sql(sa.text('SELECT * FROM {0} ORDER BY chi2 ASC LIMIT 10'.format(tb_name)), engine)
    # df = pd.read_sql_query('SELECT alpha  FROM table', engine)

    print("Samllest Chi2 values in the database.")
    print(df.head(n=15))
    name = "{0}-{1}_{2}_test_smallest_chi2.pdf".format(
        params["star"], params["obs_num"], params["chip"])
    plt.savefig(os.path.join(params["path"], "plots", name))
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
    for col in ["teff_2", "logg_2", "feh_2", "alpha", "gamma", "rv"]:
        query = """SELECT {}, {} FROM {} WHERE (teff_1 = {}  AND logg_1 = {} AND feh_1 = {})""" .format(
            col, "chi2", tb_name, params["teff"], params["logg"], params["fe_h"])
        df = pd.read_sql(sa.text(query), engine)
        print(df.columns)
        df.plot.scatter(col, "chi2")
        name = "{0}-{1}_{2}_fixed_host_params_{3}.pdf".format(
            params["star"], params["obs_num"], params["chip"], col)
        plt.savefig(os.path.join(params["path"], "plots", name))
        plt.close()


@timeit2
def fix_host_parameters_reduced_gamma(engine, params, tb_name):
    print("Fixed host analysis.")
    d_gamma = 4
    # Select lowest chisqr gamma values.
    query = """SELECT {0}, {1} FROM {2} ORDER BY {1} ASC LIMIT 1""" .format(
        "gamma", "chi2", tb_name)
    df = pd.read_sql(sa.text(query), engine)
    min_chi2_gamma = df.loc[0, "gamma"]
    print("min_chi2_gamma = ", min_chi2_gamma)
    upper_lim = min_chi2_gamma + d_gamma
    lower_lim = min_chi2_gamma - d_gamma
    print("gamma_limits", lower_lim, upper_lim)

    plt.figure()

    for ii, col in enumerate(["teff_2", "logg_2", "feh_2", "alpha", "gamma", "rv"]):
        query = """SELECT {0}, {1}, gamma FROM {2} WHERE (teff_1 = {3}  AND logg_1 = {4} AND
            feh_1 = {5} AND gamma > {6} AND gamma < {7})""".format(
            col, "chi2", tb_name, params["teff"], params["logg"], params["fe_h"],
            lower_lim, upper_lim)

        df = pd.read_sql(sa.text(query), engine, mangle_dupe_cols=False)
        print(df.columns)
        print(df.dtypes)
        print("lengths", len(df[col]), len(df["chi2"]))
        plt.subplot(3, 2, ii + 1)
        if col == "gamma":
            print("gamma cols", df.gamma)
            df.plot.scatter(col, "chi2", c="gamma", colorbar=True)
        else:
            df.plot.scatter(col, "chi2", c="gamma", colorbar=True)
    name = "{0}-{1}_{2}_fixed_host_params.pdf".format(
        params["star"], params["obs_num"], params["chip"], col)
    plt.suptitle("Chi**2 Results: {0}-{1}_{2}".format(params["star"], params["obs_num"], params["chip"]))
    plt.savefig(os.path.join(params["path"], "plots", name))
    plt.close()


if __name__ == '__main__':
    args = vars(_parser())
    opts = {k: args[k] for k in args}

    sys.exit(main(**opts))
