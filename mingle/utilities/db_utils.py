"""Location for database handling codes."""
import glob
import os
from itertools import product
from typing import Tuple, Union, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlalchemy as sa
from matplotlib import ticker, cm
from pandas.core.frame import DataFrame
from py._path.local import LocalPath
from sqlalchemy import and_, asc
from sqlalchemy.sql.schema import Table

import simulators
from mingle.utilities import chi2_at_sigma
from mingle.utilities.param_file import parse_paramfile
from mingle.utilities.param_utils import target_params, closest_model_params

odd_chi2s = ["chi2_123"]


class DBExtractor(object):
    """Methods for extracting the relevant code out of database table."""

    def __init__(self, table: Table) -> None:
        self.table = table
        self.cols = table.c
        self.bind = self.table.metadata.bind

    @staticmethod
    def chi2_columns(columns: List[str]):
        """Fixing columns to extract if a different chi2 is requested."""
        if any([True for col in columns if col in odd_chi2s]):
            columns = [x for x in columns if x not in odd_chi2s]
            columns = columns + ["chi2_1", "chi2_2", "chi2_3"]
        return columns

    @staticmethod
    def combine_chi2(df: DataFrame, columns: List[str]) -> DataFrame:
        """Fixing columns to extract if a different chi2 is requested."""
        if "chi2_123" in columns:
            df["chi2_123"] = df[["chi2_1", "chi2_2", "chi2_3"]].sum(axis=1)
        return df

    def simple_extraction(self, columns: List[str], limit: int = -1) -> DataFrame:
        """Simple table extraction, cols provided as list

        col: list of string
        limit: int (optional) default=10000

        Returns as pandas DataFrame.
        """
        new_columns = self.chi2_columns(columns)
        table_columns = [self.cols[c] for c in new_columns]
        df = pd.read_sql(
            sa.select(table_columns).limit(limit), self.bind)

        return self.combine_chi2(df, columns)

    def fixed_extraction(self, columns: List[str], fixed: Dict[str, Union[int, float]], limit: int = -1) -> DataFrame:
        """Table extraction with fixed value contitions.

        col: list of string
            Columns to return
        fixed: dict(key, value)
        limit: int (optional) default=10000

        Returns as pandas DataFrame.
        """
        assert isinstance(fixed, dict)
        new_columns = self.chi2_columns(columns)
        table_columns = [self.cols[c] for c in new_columns]

        conditions = and_(self.cols[key] == value for key, value in fixed.items())

        df = pd.read_sql(
            sa.select(table_columns).where(conditions).limit(limit), self.bind)
        return self.combine_chi2(df, columns)

    def ordered_extraction(self, order_by: str, columns: Optional[List[str]] = None, limit: int = -1,
                           asc: bool = True) -> DataFrame:
        """Table extraction with fixed value contitions.

        order_by: string
            Column name to order by.
        columns: list of strings
            Columns to return, default=None returns all.
        limit: int (optional) default=10000

        Returns as pandas dataframe.
        """
        if columns is not None:
            new_columns = self.chi2_columns(columns)
            table_columns = [self.cols[c] for c in new_columns]
        else:
            table_columns = self.cols

        if asc:
            df = pd.read_sql(
                sa.select(table_columns).order_by(
                    self.cols[order_by].asc()).limit(limit), self.bind)
        else:
            df = pd.read_sql(
                sa.select(table_columns).order_by(
                    self.cols[order_by].desc()).limit(limit), self.bind)
        return self.combine_chi2(df, columns)

    def fixed_ordered_extraction(self, columns: List[str], fixed: Dict[str, Union[int, float]], order_by: str,
                                 limit: int = -1,
                                 asc: bool = True) -> DataFrame:
        """Table extraction with fixed value contitions.

        col: list of string
            Columns to return
        fixed: dict(key, value)
        order_by: string
            Column name to order by.
        limit: int (optional) default=10000

        Returns as pandas dataframe.
        """
        assert isinstance(fixed, dict)
        new_columns = self.chi2_columns(columns)
        table_columns = [self.cols[c] for c in new_columns]

        conditions = and_(self.cols[key] == value for key, value in fixed.items())

        if asc:
            df = pd.read_sql(
                sa.select(table_columns).where(conditions).order_by(
                    self.cols[order_by].asc()).limit(limit), self.bind)
        else:
            df = pd.read_sql(
                sa.select(table_columns).where(conditions).order_by(
                    self.cols[order_by].desc()).limit(limit), self.bind)
        return self.combine_chi2(df, columns)

    def minimum_value_of(self, column: str) -> DataFrame:
        """Return only the entry for the minimum column value, limited to one value.
        """
        if column in odd_chi2s:
            if column == "chi2_123":
                selection = sa.sql.expression.label(column, self.cols["coadd_chi2"] - self.cols["chi2_4"])
            else:
                raise NotImplementedError("column {} is not catered for".format(column))
            columns = self.cols + [selection]
            df = pd.read_sql(
                sa.select(columns).order_by(asc(column)).limit(1), self.bind)
        else:
            selection = self.cols[column]
            df = pd.read_sql(
                sa.select(self.cols).order_by(selection.asc()).limit(1), self.bind)
        return df

    def full_extraction(self) -> DataFrame:
        """Return Full database:"""
        import warnings
        warnings.warn("Loading in a database may cause memory cap issues.")
        return pd.read_sql(sa.select(self.table.c), self.bind)


class SingleSimReader(object):
    def __init__(self, base=".",
                 name="BSBHMNOISE",
                 prefix="", mode="bhm", chi2_val="coadd_chi2", suffix="", obsnum="") -> None:
        self.base = base
        self.name = name.upper()
        self.suffix = suffix
        self.prefix = prefix.upper()
        self.obsnum = obsnum

        if mode in ["iam", "tcm", "bhm"]:
            self.mode = mode
        else:
            raise ValueError("Invalid SimReader mode")
        if chi2_val in ["chi2_1", "chi2_2", "chi2_3", "chi2_4", "coadd_chi2", "chi2_123"]:
            self.chi2_val = chi2_val
        else:
            raise ValueError("Invalid chi2_val.")

    def list_sims(self) -> List[str]:
        return glob.glob(os.path.join(self.base, "*"))

    def load_df(self, params: List[str] = ["teff_1", "teff_2", "logg_1", "feh_1"]) -> DataFrame:
        params.append(self.chi2_val)

        table = self.get_table()
        params = [table.c[p] for p in params]
        dbdf = pd.read_sql(sa.select(params).order_by(table.c[self.chi2_val].asc()), table.metadata.bind)

        # Coerce to be numeric columns
        c = dbdf.columns[dbdf.dtypes.eq(object)]
        dbdf[c] = dbdf[c].apply(pd.to_numeric, errors='coerce', axis=0)
        return dbdf

    def get_table(self) -> Table:
        directory = os.path.join(self.base, self.name, self.mode)
        looking_for = "{0}*{1}*_coadd_{2}_chisqr_results{3}.db".format(self.name, self.obsnum, self.mode, self.suffix)
        print("looking in ", directory, "for\n", looking_for)
        dbs = glob.glob(os.path.join(
            directory, looking_for))
        try:
            assert len(dbs) == 1, "len(dbs)={} not 1".format(len(dbs))
        except AssertionError as e:
            print("check number of databases found (should be 1)")
            print(dbs, len(dbs))
            raise e

        dbname = dbs[0]
        table = load_sql_table(dbname, verbose=False, echo=False)
        return table

    def params(self) -> Dict[str, Union[str, float, List[Union[str, float]]]]:
        """Get params from param file."""
        if simulators.paths["parameters"].startswith("."):
            param_file = os.path.join(self.base, "../", simulators.paths["parameters"],
                                      "{}_params.dat".format(self.name))
        else:
            param_file = os.path.join(simulators.paths["parameters"], "{}_params.dat".format(self.name))
        params = parse_paramfile(param_file, path=None)

        if self.mode == "bhm":
            host_params, _ = target_params(params, mode=self.mode)
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


def df_contour(df: DataFrame, xcol: str, ycol: str, zcol: str, df_min: DataFrame, lim_params: List[str],
               correct: Optional[Dict[str, float]] = None, logscale: bool = False, dof: int = 1,
               xlim: Optional[List[Union[float, int]]] = None, ylim: Optional[List[Union[float, int]]] = None) -> None:
    df_lim = df.copy()
    for param in lim_params:
        df_lim = df_lim[df_lim[param] == df_min[param].values[0]]

    if xlim is not None:
        assert len(xlim) == 2
        df_lim = df_lim[df_lim[xcol] >= xlim[0]]
        df_lim = df_lim[df_lim[xcol] <= xlim[1]]
    if ylim is not None:
        assert len(ylim) == 2
        df_lim = df_lim[df_lim[ycol] >= ylim[0]]
        df_lim = df_lim[df_lim[ycol] <= ylim[1]]

    dfpivot = df_lim.pivot(xcol, ycol, zcol)

    Y = dfpivot.columns.values
    X = dfpivot.index.values
    Z = dfpivot.values

    x, y = np.meshgrid(X, Y, indexing="ij")

    fig, ax = plt.subplots()
    if logscale:
        c = ax.contourf(x, y, Z, locator=ticker.LogLocator(), cmap=cm.viridis)
    else:
        c = ax.contourf(x, y, Z, cmap=cm.viridis)

    # Chi levels values
    sigmas = [Z.ravel()[Z.argmin()] + chi2_at_sigma(sigma, dof=dof) for sigma in range(1, 6)]
    sigma_labels = {sigmas[sig - 1]: "${}-\sigma$".format(sig) for sig in range(1, 6)}

    c2 = plt.contour(c, levels=sigmas)
    plt.clabel(c2, fmt=sigma_labels, colors='w', fontsize=14)
    cbar = plt.colorbar(c)
    cbar.ax.set_ylabel(zcol)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if correct:
        # Mark the "correct" location for the minimum chi squared
        try:
            plt.plot(correct[xcol], correct[ycol], "ro", markersize=10)
        except:
            pass

    # Mark minimum with a +.
    min_i, min_j = divmod(Z.argmin(), Z.shape[1])
    plt.plot(X[min_i], Y[min_j], "y*", markersize=10, label="$Min \chi^2$")

    plt.show()


def df_contour2(df, xcol, ycol, zcol, df_min, lim_params, correct=None, logscale=False, dof=1):
    # Need to be the minimum chi2 value for the current value of x and y
    df_lim = df.copy()
    # for param in lim_params:
    #     df_lim = df_lim[df_lim[param] == df_min[param].values[0]]
    new_df = pd.DataFrame(columns=[xcol, ycol, zcol])
    import datetime
    startx = datetime.datetime.now()
    print("Starting loop at", startx)

    x_values = df_lim[xcol].unique()
    y_values = df_lim[ycol].unique()
    print(len(df_lim))
    for x, y in product(x_values, y_values):
        # print("xval", x, "yval", y)
        df_xy = df.loc[(df_lim[xcol] == x) | (df_lim[ycol] == y)]
        new_df = new_df.append({xcol: x, ycol: y, zcol: df_xy[zcol].min()}, ignore_index=True)

    print("time to finishd", datetime.datetime.now() - startx)
    print(new_df.head())

    # df groupby
    groupstartx = datetime.datetime.now()
    grouped_df = df_lim.groupby([xcol, ycol])[[xcol, ycol, zcol]]
    print("grouped df", grouped_df.head())


    print("time for groupby", datetime.datetime.now()-groupstartx)
    print("len(new-df)", len(new_df))
    print("len(grouped-df", len(grouped_df))

    print("min loop", new_df.loc[new_df[zcol] == min(new_df[zcol])])
    # print("min grouped", grouped_df.loc[grouped_df[zcol] == min(grouped_df[zcol])])

    #

    # dfpivot = df_lim.pivot(xcol, ycol, zcol)
    dfpivot = new_df.pivot(xcol, ycol, zcol)

    Y = dfpivot.columns.values
    X = dfpivot.index.values
    Z = dfpivot.values

    x, y = np.meshgrid(X, Y, indexing="ij")

    fig, ax = plt.subplots()
    if logscale:
        c = ax.contourf(x, y, Z, locator=ticker.LogLocator(), cmap=cm.viridis)
    else:
        c = ax.contourf(x, y, Z, cmap=cm.viridis)

    # Chi levels values
    print("Using chi squared dof=", dof)
    sigmas = [Z.ravel()[Z.argmin()] + chi2_at_sigma(sigma, dof=dof) for sigma in range(1, 6)]
    sigma_labels = {sigmas[sig - 1]: "${}-\sigma$".format(sig) for sig in range(1, 6)}

    c2 = plt.contour(c, levels=sigmas)
    plt.clabel(c2, fmt=sigma_labels, colors='w', fontsize=14)
    cbar = plt.colorbar(c)
    cbar.ax.set_ylabel(zcol)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title("Correct minimum $\chi^2$ contour")
    if correct:
        # Correct location of simulation
        plt.plot(correct[xcol], correct[ycol], "ro", markersize=7)

    # Mark minimum with a +.
    min_i, min_j = divmod(Z.argmin(), Z.shape[1])
    plt.plot(X[min_i], Y[min_j], "g*", markersize=7, label="$Min \chi^2$")

    plt.show()


def decompose_database_name(database: str) -> Tuple[str, str, str, str]:
    """Database names of form */Star_obsnum_chip...db."""
    os.path.split(database)
    path, name = os.path.split(database)
    name_split = name.split("_")
    star, obsnum = name_split[0].split("-")
    chip = name_split[1]
    return path, star, obsnum, chip


def load_sql_table(database: Union[LocalPath, str], name: str = "chi2_table", echo: bool = False,
                   verbose: bool = False) -> Table:
    sqlite_db = 'sqlite:///{0}'.format(database)
    try:
        engine = sa.create_engine(sqlite_db, echo=echo)
        table_names = engine.table_names()
    except Exception as e:
        print("\nAccessing sqlite_db = {0}\n".format(sqlite_db))
        print("cwd =", os.getcwd())
        raise e
    if verbose:
        print("Table names in database =", engine.table_names())
    if len(table_names) == 1:
        tb_name = table_names[0]
    else:
        raise ValueError("Database does not just have 1 table. {0}, len={1}".format(table_names, len(table_names)))
    if tb_name != name:
        raise NameError("Name {0} given does not match table in database, {1}.".format(tb_name, table_names))

    meta = sa.MetaData(bind=engine)
    db_table = sa.Table(name, meta, autoload=True)
    return db_table
