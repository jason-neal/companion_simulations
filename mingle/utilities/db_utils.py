"""Location for database handling codes."""
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import and_


class DBExtractor(object):
    """Methods for extracting the relevant code out of database table.

    """

    def __init__(self, table, limit=10000):
        self.table = table
        self.limit = limit
        self.tab_cols = table.c

    # def complex_extraction(self, cols, fixed_values, order_value, limit=None):
    #    if limit is None:
    #        limit = self.limit
    #    df = pd.read_sql(
    #        sa.select([self.tab_cols[par], self.tab_cols[chi2_val]]).where(
    #            self.tab_cols[par] == float(unique_val)).order_by(
    #            self.tab_cols[chi2_val].asc()).limit(3), self.table.metadata.bind)
    #    return df

    def simple_extraction(self, cols, limit=None):
        """Simple table extraction, cols provided as list

        col: list of string
        limit: int (optional) default=10000

        Returns as pandas DataFrame.
        """
        if limit is None:
            limit = self.limit
        columns = [self.tab_cols[c] for c in cols]
        df = pd.read_sql(
            sa.select(columns).limit(limit), self.table.metadata.bind)
        return df

    def fixed_extraction(self, cols, fixed, limit=None):
        """Table extraction with fixed value conditions.

        col: list of string
            Columns to return
        fixed: dict(key, value)
        limit: int (optional) default=10000

        Returns as pandas DataFrame.
        """
        assert isinstance(fixed, dict)
        if limit is None:
            limit = self.limit
        columns = [self.tab_cols[c] for c in cols]

        conditions = and_(self.tab_cols[key] == value for key, value in fixed.items())

        df = pd.read_sql(
            sa.select(columns).where(conditions).limit(limit), self.table.metadata.bind)
        return df
