"""Script to load part files into a sql database.

For the inherent alpha model

We assume that the host temperature is well known so we will resitrict the results to the host temperature. (within error bars)
For HD30501 and HD211847 this means  +- 50K so a fixed temperature.



"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import sqlalchemy as sa

from utilities.param_file import parse_paramfile


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
    return path, name.split("_")[:3]


def get_host_params(star):
    """Find host star parameters from param file."""
    param_file = "/home/jneal/Phd/data/parameter_files/{}_params.txt".format(star)
    params = parse_paramfile(param_file)
    return params["temp"], params["logg"], params["fe_h"]

def savefig(name):
    plt.savefig(os.path.join(path, "plots", name), bbox="tight")


def main(database, echo=False):
    path, star, obs_num, chip = decompose_database_name(database)

    teff, logg, fe_h = closest_model_params(*get_host_params(star), path)

    engine = sa.create_engine('sqlite:///{}'.format(database), echo=echo)
    table_names = engine.table_names()
    print("Table names in database =", engine.table_names())
    if len(table_names) == 1:
        tb_name = table_names[0]
    else:
        raise ValueError("Database has two many tables {}".format(table_names))


    fix_host_parameters(engine, teff, logg, fe_h)

    get_column_limits(engine, tb_name)
    smallest_chi2_values(engine, tb_name)

    df = pd.read_sql_query('SELECT alpha, chi2 FROM {0} LIMIT 10000'.format(tb_name), engine)

    fig, ax = plt.subplots()
    ax.scatter(df["alpha"], df["chi2"], s=3, alpha=0.5)

    ax.set_xlabel('alpha', fontsize=15)
    ax.set_ylabel('chi2', fontsize=15)
    ax.set_title('alpha and Chi2.')

    ax.grid(True)
    fig.tight_layout()

    plt.show()

    alpha_rv_plot(engine, tb_name)
    # alpha_rv_contour(engine, tb_name)


def alpha_rv_plot(engine, tb_name=None):
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

    plt.show()


def alpha_rv_contour(engine, tb_name=None):
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

    plt.show()


def smallest_chi2_values(engine, tb_name):
    """Find smallest chi2 in table."""
    df = pd.read_sql(sa.text('SELECT * FROM {0} ORDER BY chi2 ASC LIMIT 10'.format(tb_name)), engine)
    # df = pd.read_sql_query('SELECT alpha  FROM table', engine)

    print("Samllest Chi2 values in the database.")
    print(df.head(n=15))

    plt.show()


def get_column_limits(engine, tb_name):
    print("Column Value Ranges")
    for col in ["teff_1", "teff_2", "logg_1", "logg_2", "alpha", "gamma", "rv", "chi2"]:
        query = """
               SELECT * FROM (SELECT {1} FROM {0} ORDER BY {1} ASC LIMIT 1)
               UNION ALL
               SELECT * FROM (SELECT {1} FROM {0} ORDER BY {1} DESC LIMIT 1)
               """.format(tb_name, col)
        df = pd.read_sql(sa.text(query), engine)
        print(col, min(df[col]), max(df[col]))


def fix_host_parameters(engine, teff, logg, fe_h):
    print("Fixed host analysis")
    for col in ["teff_2", "logg_2", "alpha", "gamma", "rv"]
    query = """
           SELECT {} FROM {} WHERE teff_1 == {} logg_1 == {} feh_1 == {})""" .format(col, "chi2", teff, logg, fe_h)
    df = pd.read_sql(sa.text(query), engine)
    df.plot.scatter()
    savefig("fixed_host_params_")
    plt.show()


if __name__ == '__main__':
    args = vars(_parser())
    opts = {k: args[k] for k in args}

    sys.exit(main(**opts))
