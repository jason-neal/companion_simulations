#!/usr/bin/env python
"""Script to load part files into a sql database.

For the inherent alpha model

We assume that the host temperature is well known so we will restrict the
results to the host temperature. (within error bars)
For HD30501 and HD211847 this means  +- 50K so a fixed temperature.

"""
import argparse
import os
import sys

import sqlalchemy as sa

import simulators
from bin.analysis_module import (alpha_rv_contour, alpha_rv_contour_old,
                                 fix_host_parameters,
                                 fix_host_parameters_reduced_gamma,
                                 get_column_limits, parabola_plots,
                                 smallest_chi2_values, test_figure)
from utilities.param_file import get_host_params
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
    parser.add_argument("-m", "--mode", help="Analysis mode to choose", default="parabola",
                        choices=["parabola", "fixed_host_params", "param_limits", "smallest_chi2", "test", "contour",
                                 "contour_old"])
    return parser.parse_args()


def decompose_database_name(database):
    """Database names of form */Star_obsnum_chip...db ."""
    os.path.split(database)
    path, name = os.path.split(database)
    name_split = name.split("_")
    star, obsnum = name_split[0].split("-")
    chip = name_split[1]
    return path, star, obsnum, chip


def load_sql_table(database, name="chi2_table", echo=False):
    sqlite_db = 'sqlite:///{}'.format(database)
    try:
        engine = sa.create_engine(sqlite_db, echo=echo)
        table_names = engine.table_names()
    except Exception as e:
        print("\nAccessing sqlite_db = {}\n".format(sqlite_db))
        print("cwd =", os.getcwd())
        raise e
    print("Table names in database =", engine.table_names())
    if len(table_names) == 1:
        tb_name = table_names[0]
    else:
        raise ValueError("Database has two many tables {}".format(table_names))
    if tb_name != name:
        raise NameError("Name given does not match table in database.")

    meta = sa.MetaData(bind=engine)
    db_table = sa.Table(name, meta, autoload=True)
    return db_table


def main(database, echo=False, mode="parabola"):
    path, star, obs_num, chip = decompose_database_name(database)
    os.makedirs(os.path.join(path, "plots"), exist_ok=True)  # make dir for plots

    teff, logg, fe_h = closest_model_params(*get_host_params(star))
    params = {"path": path, "star": star, "obs_num": obs_num, "chip": chip, "teff": teff, "logg": logg, "fe_h": fe_h}

    sqlite_db = 'sqlite:///{}'.format(database)

    try:
        engine = sa.create_engine(sqlite_db, echo=echo)
        table_names = engine.table_names()
    except Exception as e:
        print("\nAccessing sqlite_db = {}\n".format(sqlite_db))
        print("cwd =", os.getcwd())
        raise e

    print("Table names in database =", engine.table_names())
    if len(table_names) == 1:
        tb_name = table_names[0]
    else:
        raise ValueError("Database has two many tables {}".format(table_names))

    db_table = load_sql_table(database)

    print("Mode =", mode)

    if mode == "fixed_host_params":
        fix_host_parameters_reduced_gamma(engine, params, tb_name)
        fix_host_parameters(engine, params, tb_name)
    elif mode == "param_limits":
        get_column_limits(engine, params, tb_name)
    elif mode == "parabola":
        parabola_plots(db_table, params)
    elif mode == "smallest_chi2":
        smallest_chi2_values(engine, params, tb_name)
    elif mode == "contour":
        alpha_rv_contour(engine, params, tb_name)
    elif mode == "contour":
        alpha_rv_contour_old(engine, params, tb_name)
    elif mode == "test":
        test_figure(engine, params, tb_name)

    print("Done")
    return 0


if __name__ == '__main__':
    args = vars(_parser())
    opts = {k: args[k] for k in args}

    sys.exit(main(**opts))
