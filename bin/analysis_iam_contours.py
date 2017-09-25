"""Script to load part files into a sql database.

For the inherent alpha model

We assume that the host temperature is well known so we will restrict the
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


from bin.analysis_iam_chi2 import (decompose_database_name)
from bin.analysis_module import alpha_rv_plot, fix_host_parameters, smallest_chi2_values, \
    fix_host_parameters_reduced_gamma, get_column_limits


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



if __name__ == '__main__':
    args = vars(_parser())
    opts = {k: args[k] for k in args}

    sys.exit(main(**opts))
