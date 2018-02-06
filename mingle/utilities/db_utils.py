"""Location for database handling codes."""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import and_

import simulators
from mingle.utilities import chi2_at_sigma
from mingle.utilities.param_file import parse_paramfile
from mingle.utilities.phoenix_utils import closest_model_params
from simulators.iam_module import target_params


class DBExtractor(object):
    """Methods for extracting the relevant code out of database table."""

    def __init__(self, table):
        self.table = table
        self.cols = table.c
        self.bind = self.table.metadata.bind

    def simple_extraction(self, columns, limit=-1):
        """Simple table extraction, cols provided as list

        col: list of string
        limit: int (optional) default=10000

        Returns as pandas dataframe.
        """
        table_columns = [self.cols[c] for c in columns]
        df = pd.read_sql(
            sa.select(table_columns).limit(limit), self.bind)
        return df

    def fixed_extraction(self, columns, fixed, limit=-1):
        """Table extraction with fixed value contitions.

        col: list of string
            Columns to return
        fixed: dict(key, value)
        limit: int (optional) default=10000

        Returns as pandas dataframe.
        """
        assert isinstance(fixed, dict)

        table_columns = [self.cols[c] for c in columns]

        conditions = and_(self.cols[key] == value for key, value in fixed.items())

        df = pd.read_sql(
            sa.select(table_columns).where(conditions).limit(limit), self.bind)
        return df

    def fixed_ordered_extraction(self, columns, fixed, order, limit=-1, asc=True):
        """Table extraction with fixed value contitions.

        col: list of string
            Columns to return
        fixed: dict(key, value)
        limit: int (optional) default=10000

        Returns as pandas dataframe.
        """
        assert isinstance(fixed, dict)

        table_columns = [self.cols[c] for c in columns]

        conditions = and_(self.cols[key] == value for key, value in fixed.items())

        if asc:
            df = pd.read_sql(
                sa.select(table_columns).where(conditions).order_by(
                    self.cols[order].asc()).limit(limit), self.bind)
        else:
            df = pd.read_sql(
                sa.select(table_columns).where(conditions).order_by(
                    self.cols[order].desc()).limit(limit), self.bind)
        return df

    def minimum_value_of(self, column):
        """Return only the entry for the minimum column value, limited to one value.
        """
        selection = self.cols[column]
        df = pd.read_sql(
            sa.select(self.cols).order_by(selection.asc()).limit(1), self.bind)
        return df

    def full_extraction(self):
        return pd.read_sql(
            sa.select(self.table.c), self.bind)

        """Return Full database:"""
        import warnings
        warnings.warn("Loading in a database may cause memory cap issues.")


class SingleSimReader(object):
    def __init__(self, base=".",
                 name="BSBHMNOISE",
                 prefix="", mode="bhm", chi2_val="coadd_chi2"):
        self.base = base
        self.name = name.upper()
        self.prefix = prefix.upper()
        if mode in ["iam", "tcm", "bhm"]:
            self.mode = mode
        else:
            raise ValueError("Invalid SimReader mode")
        if chi2_val in ["chi2_1", "chi2_2", "chi2_3", "chi2_4", "coadd_chi2"]:
            self.chi2_val = chi2_val
        else:
            raise ValueError("Invalid chi2_val.")

    def list_sims(self):
        return glob.glob(os.path.join(self.base, "*"))

    def load_df(self, params=["teff_1", "teff_2", "logg_1", "feh_1"]):
        params.append(self.chi2_val)

        table = self.get_table()
        # print(table.c)
        params = [table.c[p] for p in params]
        dbdf = pd.read_sql(sa.select(params).order_by(table.c[self.chi2_val].asc()), table.metadata.bind)

        # Coerce to be numeric columns
        c = dbdf.columns[dbdf.dtypes.eq(object)]
        dbdf[c] = dbdf[c].apply(pd.to_numeric, errors='coerce', axis=0)
        return dbdf

    def get_table(self):
        starname = self.name
        directory = os.path.join(self.base, starname, self.mode)
        # print(directory)
        dbs = glob.glob(os.path.join(directory, "*_coadd_{}_chisqr_results.db".format(self.mode)))
        # print(dbs)
        assert len(dbs) == 1, print(len(dbs))
        dbname = dbs[0]
        table = load_sql_table(dbname, verbose=False, echo=False)
        return table

    def params(self):
        """Get params from param file."""
        if simulators.paths["parameters"].startswith("."):
            param_file = os.path.join(self.base, "../", simulators.paths["parameters"],
                                      "{}_params.dat".format(self.name))
        else:
            param_file = os.path.join(simulators.paths["parameters"], "{}_params.dat".format(self.name))
        params = parse_paramfile(param_file, path=None)
        print(params)
        print("self mode", self.mode)
        if self.mode == "bhm":
            host_params = target_params(params, mode=self.mode)
            closest_host_model = closest_model_params(*host_params)  # unpack temp, logg, fe_h with *
        else:
            host_params, comp_params = target_params(params, mode=self.mode)
            closest_host_model = closest_model_params(*host_params)  # unpack temp, logg, fe_h with *
            closest_comp_model = closest_model_params(*comp_params)
            params.update(
                {"teff_2": closest_comp_model[0], "logg_2": closest_comp_model[1], "feh_2": closest_comp_model[2]})

        params.update(
            {"teff_1": closest_host_model[0], "logg_1": closest_host_model[1], "feh_1": closest_host_model[2]})
        return params


def df_contour(df, xcol, ycol, zcol, df_min, lim_params, correct=None):
    df_lim = df.copy()
    for param in lim_params:
        df_lim = df_lim[df_lim[param] == df_min[param].values[0]]

    dfpivot = df_lim.pivot(xcol, ycol, zcol)

    Y = dfpivot.columns.values
    X = dfpivot.index.values
    Z = dfpivot.values
    print(X, Y, Z.shape)

    x, y = np.meshgrid(X, Y, indexing="ij")

    fig, ax = plt.subplots()
    c = ax.contourf(x, y, np.log(Z))

    # Chi levels values
    sigmas = [np.log(Z.ravel()[Z.argmin()] + chi2_at_sigma(sigma, df=1)) for sigma in range(1, 6)]
    print(sigmas, len(sigmas))
    sigma_labels = {sigmas[sig-1]: "${}-\sigma$".format(sig) for sig in range(1, 6)}
    print(" sigma_labels= ", sigma_labels)
    c2 = plt.contour(c, levels=sigmas)
    plt.clabel(c2, fmt=sigma_labels, colors='w', fontsize=14)
    cbar = plt.colorbar(c)
    cbar.ax.set_ylabel(zcol)
    plt.xlabel(xcol)
    plt.ylabel(ycol)

    if correct:
        # Correct location of simulation
        plt.plot(correct[xcol], correct[ycol], "ro", markersize=7)

    # Mark minimum with a +.
    min_i, min_j = divmod(Z.argmin(), Z.shape[1])
    plt.plot(X[min_i], Y[min_j], "g*", markersize=7, label="$Min \chi^2$")

    plt.show()

