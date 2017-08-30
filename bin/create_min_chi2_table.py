"""creat_min_ch2_table.py

Create Table of minimum Chi_2 values and save to a table.
"""
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import sqlalchemy as sa

from bin.analysis_iam_chi2 import decompose_database_name
from utilities.phoenix_utils import closest_model_params


def main(star, obs_num, chip, echo=False):
    database = os.path.join("Analysis/{0}/{0}-{1}_{2}_iam_chisqr_results.db".format(star, obs_num, chip))
    path, star, obs_num, chip = decompose_database_name(database)

    teff, logg, fe_h = closest_model_params(*get_host_params(star))
    params = {"path": path, "star": star, "obs_num": obs_num, "chip": chip, "teff": teff, "logg": logg, "fe_h": fe_h}

    engine = sa.create_engine('sqlite:///{}'.format(database), echo=echo)
    table_names = engine.table_names()

    query = """SELECT * FROM {} SORTBY chi2 LIMIT 1""" .format(
        col, "chi2", tb_name, params["teff"], params["logg"], params["fe_h"])
    df = pd.read_sql(sa.text(query), engine)

    name = os.path.join(path, "{0}_iam_all_observation_min_chi2.tsv".format(star))

    df["obs_num"] = obs_ṇum
    df["chip"] = chip
    columns = ["obs_num", "chip", "teff_1", "logg_1", "feh_1", "teff_2", "logg_2", "feh_2", "alpha", "rv", "gamma"]

    if os.exists(name):
        df[columns].to_csv(name, sep='\t', mode="a", index=False, header=True)
    else:
        df[columns].to_csv(name, sep='\t', mode="a", index=False, header=False)


if __name__ == "__main__":
    stas = ["HD30501"]

    obs_nums = {"HD30501": ["1", "2a", "2b", "3"], "HD211847": ["1", "2"]}
    chips = range(1, 5)
    for star in stars:
        for obs_num in obs_nums:
            for chip in chips:
                try:
                    main(star, obs_nums, chips)
                except:
                    print("Table creation failed for {0}-{1}_{2}".format(star, obs_num, chip))
                    continue
