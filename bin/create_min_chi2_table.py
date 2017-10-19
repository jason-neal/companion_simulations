"""create_min_chi2_table.py.

Create Table of minimum Chi_2 values and save to a table.
"""
import argparse
import os

import corner
import matplotlib.pyplot as plt
import pandas as pd
import simulators
import sqlalchemy as sa
from bin.analysis_iam_chi2 import decompose_database_name
from joblib import Parallel, delayed
from pandas.plotting import scatter_matrix
from utilities.param_file import get_host_params
from utilities.phoenix_utils import closest_model_params
from utilities.scatter_corner import scatter_corner


def _parser():
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Minimum chi-squared table.')
    parser.add_argument('-s', '--stars', help='Star names', nargs="+", default=None)
    parser.add_argument('--suffix', help='Suffix to add to the file names.', default="")
    return parser.parse_args()


def main(star, obs_num, chip, suffix="", echo=False):
    database = os.path.join(simulators.paths["output_dir"], star, "{0}-{1}_{2}_iam_chisqr_results{3}.db".format(star, obs_num, chip, suffix))
    path, star, obs_num, chip = decompose_database_name(database)
    os.makedirs(os.path.join(path, "plots"), exist_ok=True)
    save_name = os.path.join(path, "{0}_iam_all_observation_min_chi2{1}.tsv".format(star, suffix))

    teff, logg, fe_h = closest_model_params(*get_host_params(star))
    params = {"path": path, "star": star, "obs_num": obs_num, "chip": chip,
              "teff": teff, "logg": logg, "fe_h": fe_h}

    # Hack to run from editor
    if os.getcwd().endswith("companion_simulations/bin"):
        database = "../" + database
        save_name = "../" + save_name

    if os.path.exists(database):
        engine = sa.create_engine('sqlite:///{0}'.format(database), echo=echo)
    else:
        raise IOError("Database does not exist.")
    table_names = engine.table_names()
    if len(table_names) == 1:
        tb_name = table_names[0]
    else:
        raise ValueError("Database has two many tables {0}".format(table_names))

    query = """SELECT * FROM {0}
               WHERE (teff_1 = {1} AND logg_1 = {2} AND feh_1 = {3})
               ORDER BY chi2 LIMIT 1
               """ .format(tb_name, params["teff"], params["logg"], params["fe_h"])
    df = pd.read_sql(sa.text(query), engine)

    df["obs_num"] = obs_num
    df["chip"] = chip
    columns = ["obs_num", "chip", "teff_1", "logg_1", "feh_1", "teff_2",
               "logg_2", "feh_2", "alpha", "rv", "gamma", "chi2"]

    if os.path.exists(save_name):
        df.to_csv(save_name, columns=columns, sep='\t', mode="a", index=False, header=False)
    else:
        df.to_csv(save_name, columns=columns, sep='\t', mode="a", index=False, header=True)

    return save_name


def scatter_plots(star, filename):
    """Load minimum chi2 table and make scatter plots across chips."""
    df = pd.read_table(filename, sep="\t")

    df.loc[:, "chip"] = df.loc[:, "chip"].astype(int)

    fig, axes = plt.subplots(5, 2)
    subdf = df.loc[:, ["chip", "teff_2", "alpha", "rv", "gamma", "chi2"]]  # "logg_2", "feh_2"

    scatter_matrix(subdf, alpha=1, figsize=(12, 12), diagonal='hist')
    plt.suptitle("{0} Observation/chip variations".format(star))

    path, fname = os.path.split(filename)
    figname = os.path.join(path, "plots", "{0}_scatter.pdf".format(fname.split(".")[0]))
    plt.savefig(figname)

    figname = os.path.join(path, "plots", "{0}_scatter.png".format(fname.split(".")[0]))
    plt.savefig(figname)
    plt.close()


def scatter_corner_plots(star, filename):
    """Load minimum chi2 table and make scatter plots across chips."""
    df = pd.read_table(filename, sep="\t")

    df.loc[:, "chip"] = df.loc[:, "chip"].astype(int)

    fig, axes = plt.subplots(5, 2)
    subdf = df.loc[:, ["chip", "teff_2", "alpha", "rv", "gamma", "chi2"]]  # "logg_2", "feh_2"

    scatter_corner(subdf, alpha=1, figsize=(12, 12), diagonal='hist', corner="lower")
    plt.suptitle("{0} Observation/chip variations".format(star))

    path, fname = os.path.split(filename)
    figname = os.path.join(path, "plots", "{0}_scatter_corner.pdf".format(fname.split(".")[0]))
    plt.savefig(figname)

    figname = os.path.join(path, "plots", "{0}_scatter_corner.png".format(fname.split(".")[0]))
    plt.savefig(figname)
    plt.close()


# Corner.corner
def min_chi2_corner_plot(star, filename):
    df = pd.read_table(filename, sep="\t")

    df.loc[:, "chip"] = df.loc[:, "chip"].astype(int)

    subdf = df.loc[:, ["chip", "teff_2", "alpha", "rv", "gamma", "chi2"]]  # "logg_2", "feh_2"

    corner.corner(subdf.values, labels=subdf.columns.values)
    plt.suptitle("{0} Observation/chip variations".format(star))

    path, fname = os.path.split(filename)
    figname = os.path.join(path, "plots", "{0}_corner_corner.png".format(fname.split(".")[0]))
    plt.savefig(figname)

    corner.corner(subdf.values, labels=subdf.columns.values, plot_contours=False)
    plt.suptitle("{0} Observation/chip variations".format(star))
    figname = os.path.join(path, "plots", "{0}_corner_contoured.png".format(fname.split(".")[0]))
    plt.savefig(figname)
    plt.close()


# TODO common function to determine observations and chips for different stars  (like here)
if __name__ == "__main__":
    args = _parser()
    stars = args.stars
    if stars is None:
        stars = ["HD30501", "HD211847", "HD4747"]
    print("Stars", stars)
    obs_nums = {"HD30501": ["1", "2a", "2b", "3"], "HD211847": ["1", "2"], "HD4747": ["1"]}
    chips = range(1, 5)

    def paralleled_main(star):
        star_obs_nums = obs_nums[star]
        for obs_num in star_obs_nums:
            for chip in chips:
                try:
                    save_name = main(star, obs_num, chip, suffix=args.suffix)
                except Exception as e:
                    print(e)
                    print("Table creation failed for {0}-{1}_{2}".format(star, obs_num, chip))
                    continue
        try:
            scatter_plots(star, save_name)

            scatter_corner_plots(star, save_name)

            min_chi2_corner_plot(star, save_name)
        except Exception as e:
            print(" Corner plots did not work.")
            raise e

    # Run in parallel
    Parallel(n_jobs=-1)(delayed(paralleled_main)(star) for star in stars)