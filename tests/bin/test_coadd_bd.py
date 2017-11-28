import os

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa

import simulators
from bin.coadd_analysis_module import smallest_chi2_values
from bin.coadd_analysis_script import load_sql_table, decompose_database_name
from bin.coadd_chi2_db import main as iam_db_main
from bin.coadd_chi2_db import parse_args


# from bin.coadd_analysis_module import contours, smallest_chi2_values, compare_spectra


@pytest.fixture()
def db_name():
    name = os.path.join("tests", "testdata", "HD30501-1_coadd_iam_chisqr_results.db")
    return name


@pytest.fixture()
def db_table(db_name):
    table = load_sql_table(db_name, name="chi2_table", echo=False, verbose=False)

    return table


@pytest.fixture()
def db_params():
    """Database params for fixture db."""
    return {}


def test_load_sql_table(db_name):
    table = load_sql_table(db_name, name="chi2_table", echo=False, verbose=False)
    assert isinstance(table, sa.Table)


@pytest.mark.parametrize("invalid_name", ["", "no_table"])
def test_load_sql_table_with_invalid_table(db_name, invalid_name):
    with pytest.raises(NameError):
        load_sql_table(db_name, name=invalid_name, echo=False, verbose=False)


@pytest.mark.xfail()


def test_decompose_database_name(db_name):
    path, star, obsnum, chip = decompose_database_name(db_name)

    assert path == os.path.join("tests", "testdata")
    assert star == "HD30501"
    assert chip == "coadd"
    assert obsnum == "1"




def test_simple_database_returns_correctly_from_sql_db(tmpdir):
    fname = os.path.join(tmpdir, "test_db.db")
    x = np.linspace(1, 5, 20)
    y = x ** 2
    z = x + y
    df = pd.DataFrame({"x": x, "y": y, "z": z})

    assert np.all(df.x.values == x)
    assert np.all(df.y.values == y)
    assert np.all(df.z.values == z)

    database_name = 'sqlite:///{0}'.format(fname)
    engine = sa.create_engine(database_name)
    df.to_sql('test_table', engine, if_exists='append')

    db_table = load_sql_table(fname, name="test_table")

    df = pd.read_sql(
        sa.select([table.c["x"], table.c["y"], table.c["z"]]),
        table.metadata.bind)

    # NEED to query out x from database inot dataframe.
    # select x, y, z from db_tbale using sqlalchemy
    assert np.all(df.x.values == x)
    assert np.all(df.y.values == y)
    assert np.all(df.z.values == z)


from simulators.iam_module import iam_helper_function

def test_coadd_chi2_bd_parser_defaults():
    args = ["HDdefault", "0", ]
    parsed = parse_args(args)
    assert parsed.star == "HDdefault"
    assert parsed.obsnum == "0"
    assert parsed.suffix is ""
    assert parsed.chunksize == 1000
    assert isinstance(parsed.chunksize, int)
    assert parsed.replace is False
    assert parsed.verbose is False
    assert parsed.move is False


def test_coadd_chi2_bd_parser():
    args = ["HDswitches", "1a", "--suffix", "_test", "-v", "-r", "-c", "50000", "-m"]
    parsed = parse_args(args)
    assert parsed.star == "HDswitches"
    assert parsed.obsnum == "1a"
    assert parsed.suffix is "_test"
    assert parsed.chunksize == 50000
    assert isinstance(parsed.chunksize, int)
    assert parsed.replace is True
    assert parsed.verbose is True
    assert parsed.move is True
