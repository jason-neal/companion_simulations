"""Script to load part files into a sql database.

For the inherent alpha model

We assume that the host temperature is well known so we will restrict the
results to the host temperature. (within error bars)
For HD30501 and HD211847 this means  +- 50K so a fixed temperature.

"""
import argparse
import os
import sys

import simulators
import sqlalchemy as sa
from bin.coadd_analysis_module import (contours,
                                       display_arbitary_norm_values,
                                       fix_host_parameters, rv_plot,
                                       fix_host_parameters_reduced_gamma,
                                       get_column_limits,chi2_parabola_plots,
                                        parabola_plots,
                                       smallest_chi2_values, test_figure)
from utilities.param_file import get_host_params
from utilities.phoenix_utils import closest_model_params


def _parser():
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Chisquare analysis.')
    parser.add_argument('star', help='Star Name')
    parser.add_argument('obsnum', help="Observation label")
    parser.add_argument('--suffix', default=None,
                        help='Suffix to add to database name.')
    parser.add_argument("-e", "--echo", action="store_true",
                        help="Echo the SQL queries")
    parser.add_argument('-v', '--verbose', action="store_true",
                        help='Turn on Verbose.')
    parser.add_argument('-p', '--npars', type=int, default=3,
                        help='Number of interesting parameters. (default=3)')
    parser.add_argument("-m", "--mode", default="parabola",
                        help="Analysis mode to choose",
        choices=["parabola", "fixed_host_params", "param_limits", "smallest_chi2",
                 "test", "contour", "arbnorm", "all", "rvplot", "chi2_parabola"])
    parser.add_argument('-n', '--norm', action="store_true",
                        help='Normalized chi2 (min(chi**2) == 1).')
    return parser.parse_args()


def decompose_database_name(database):
    """Database names of form */Star_obsnum_chip...db."""
    os.path.split(database)
    path, name = os.path.split(database)
    name_split = name.split("_")
    star, obsnum = name_split[0].split("-")
    chip = name_split[1]
    return path, star, obsnum, chip


def load_sql_table(database, name="chi2_table", echo=False, verbose=False):
    sqlite_db = 'sqlite:///{}'.format(database)
    try:
        engine = sa.create_engine(sqlite_db, echo=echo)
        table_names = engine.table_names()
    except Exception as e:
        print("\nAccessing sqlite_db = {}\n".format(sqlite_db))
        print("cwd =", os.getcwd())
        raise e
    if verbose:
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


def main(star, obsnum, suffix=None, echo=False, mode="parabola",
         verbose=False, norm=False, npars=3):
    suffix = "" if suffix is None else suffix
    database = coadd_database = os.path.join(simulators.paths["output_dir"], star,
        "{0}-{1}_coadd_iam_chisqr_results{2}.db".format(star, obsnum, suffix))

    if verbose:
        print("Database name ", database)
        print("Database exists", os.path.isfile(database))
    if not os.path.isfile(database):
        raise IOError("Database '{}' does not exist.".format(database))

    path, dbstar, db_obsnum, chip = decompose_database_name(database)
    assert dbstar == star
    assert db_obsnum == obsnum
    assert chip == "coadd"

    os.makedirs(os.path.join(path, "plots"), exist_ok=True)  # make dir for plots

    teff, logg, fe_h = closest_model_params(*get_host_params(star))
    params = {"path": path, "star": star, "obs_num": obsnum, "chip": chip, "suffix": suffix,
              "teff": int(teff), "logg": float(logg), "fe_h": float(fe_h)}

    #sqlite_db = 'sqlite:///{}'.format(database)

    #try:
        # Careful this creates an empty db if it doesn't exist
    #    engine = sa.create_engine(sqlite_db, echo=echo)
    #    table_names = engine.table_names()
    #except Exception as e:
    #    print("\nAccessing sqlite_db = {}\n".format(sqlite_db))
    #    print("cwd =", os.getcwd())
    #    raise e

    db_table = load_sql_table(database, verbose=verbose)

    print("Mode =", mode)

    if mode == "fixed_host_params":
        fix_host_parameters_reduced_gamma(db_table, params)
        fix_host_parameters(db_table, params)
    elif mode == "param_limits":
        get_column_limits(db_table, params)
    elif mode == "parabola":
        parabola_plots(db_table, params, norm=norm)
    elif mode == "smallest_chi2":
        smallest_chi2_values(db_table, params)
    elif mode == "contour":
        contours(db_table, params)
    elif mode == "test":
        test_figure(db_table, params)
    elif mode == "rvplot":
        rv_plot(db_table, params)
    elif mode == "arbnorm":
        display_arbitary_norm_values(db_table, params)
    elif mode == "chi2_parabola":
        chi2_parabola_plots(db_table, params, npars=npars)
    elif mode == "all":
        fix_host_parameters_reduced_gamma(db_table, params,)
        get_column_limits(db_table, params)
        fix_host_parameters(db_table, params)
        display_arbitary_norm_values(db_table, params)
        smallest_chi2_values(db_table, params)
        parabola_plots(db_table, params, norm=norm)
        contours(db_table, params)
        test_figure(db_table, params)
        chi2_parabola_plots(db_table, params, npars=npars)
    print("Done")
    return 0


if __name__ == '__main__':
    args = vars(_parser())
    opts = {k: args[k] for k in args}

    sys.exit(main(**opts))
