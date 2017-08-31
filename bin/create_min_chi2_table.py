"""create_min_chi2_table.py

Create Table of minimum Chi_2 values and save to a table.
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import sqlalchemy as sa
from pandas.plotting import scatter_matrix

from bin.analysis_iam_chi2 import decompose_database_name, get_host_params
from utilities.phoenix_utils import closest_model_params


def main(star, obs_num, chip, echo=False):

    database = os.path.join("Analysis/{0}/{0}-{1}_{2}_iam_chisqr_results.db".format(star, obs_num, chip))
    path, star, obs_num, chip = decompose_database_name(database)
    save_name = os.path.join(path, "{0}_iam_all_observation_min_chi2.tsv".format(star))

    teff, logg, fe_h = closest_model_params(*get_host_params(star))
    params = {"path": path, "star": star, "obs_num": obs_num, "chip": chip, "teff": teff, "logg": logg, "fe_h": fe_h}

    # Hack to run from editor
    if os.getcwd().endswith("companion_simulations/bin"):
        database = "../" + database
        save_name = "../" + save_name

    if os.path.exists(database):
        engine = sa.create_engine('sqlite:///{}'.format(database), echo=echo)
    else:
        raise IOError("Database does not exist.")
    table_names = engine.table_names()
    if len(table_names) == 1:
        tb_name = table_names[0]
    else:
        raise ValueError("Database has two many tables {}".format(table_names))

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


def scatter_plots(star, name):
    """Load minimum chi2 table and make scatter plots across chips."""
    df = pd.read_table(name, sep="\t")
    print(df.columns)
    print(df.dtypes)

    df.loc[:, "chip"] = df.loc[:, "chip"].astype(int)
    print(df.dtypes)
    print(df)
    fig, axes = plt.subplots(5, 2)
    print(axes)
    print(axes[0, 0])
    # df.plot(ax=axes[0, 0]).scatter("chip", "chi2")
    # df.plot(ax=axes[0, 1]).scatter("chip", "teff_2", c="chi2", colorbar=True)
    # df.plot(ax=axes[1, 0]).scatter("chip", "alpha", c="chi2", colorbar=True)
    # df.plot(ax=axes[1, 1]).scatter("chip", "rv", c="chi2", colorbar=True)
    # df.plot(ax=axes[2, 0]).scatter("chip", "gamma", c="chi2", colorbar=True)
    # df.plot(ax=axes[2, 1]).scatter("gamma", "alpha", c="chi2", colorbar=True)
    # df.plot(ax=axes[3, 0]).scatter("teff_2", "rv", c="chi2", colorbar=True)
    # df.plot(ax=axes[3, 1]).scatter("teff_2", "gamma", c="chi2", colorbar=True)
    # df.plot(ax=axes[4, 0]).scatter("teff_2", "alpha", c="chi2", colorbar=True)
    # df.plot(ax=axes[4, 1]).scatter("rv", "alpha", c="chi2", colorbar=True)
    subdf = df.loc[:, ["chip", "teff_2", "alpha", "rv", "gamma", "chi2"]]  # "logg_2", "feh_2"

    scatter_matrix(subdf, alpha=1, figsize=(12, 12), diagonal='hist')
    plt.suptitle("{} Observation/chip variations".format(star))

    path, fname = os.path.split(name)
    figname = os.path.join(path, "plots", "{}_scatter.pdf".format(fname.split(".")[0]))
    plt.savefig(figname)

    figname = os.path.join(path, "plots", "{}_scatter.png".format(fname.split(".")[0]))
    plt.savefig(figname)


if __name__ == "__main__":
    stars = ["HD30501"]

    obs_nums = {"HD30501": ["1", "2a", "2b", "3"], "HD211847": ["1", "2"]}
    chips = range(1, 5)
    for star in stars:
        star_obs_nums = obs_nums[star]
        for obs_num in star_obs_nums:
            for chip in chips:
                try:
                    save_name = main(star, obs_num, chip)
                except Exception as e:
                    print(e)
                    print("Table creation failed for {0}-{1}_{2}".format(star, obs_num, chip))
                    continue
        scatter_plots(star, save_name)
