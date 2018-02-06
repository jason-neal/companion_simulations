"""Location for database handling codes."""
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import and_



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

