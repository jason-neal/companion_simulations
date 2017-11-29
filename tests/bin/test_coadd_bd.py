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
def test_sql_table_with_more_than_one_table():
    db_name = "db with many tables."
    with pytest.raises(ValueError):
        load_sql_table(db_name, echo=False, verbose=False)


def test_decompose_database_name(db_name):
    path, star, obsnum, chip = decompose_database_name(db_name)

    assert path == os.path.join("tests", "testdata")
    assert star == "HD30501"
    assert chip == "coadd"
    assert obsnum == "1"


# Assert they return None
# An image appears somewhere
# And some text is captured?


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
        sa.select([db_table.c["x"], db_table.c["y"], db_table.c["z"]]),
        db_table.metadata.bind)

    # NEED to query out x from database inot dataframe.
    # select x, y, z from db_tbale using sqlalchemy
    assert np.all(df.x.values == x)
    assert np.all(df.y.values == y)
    assert np.all(df.z.values == z)


from simulators.iam_module import iam_helper_function, setup_iam_dirs


def test_iam_db_main(tmpdir):
    simulators.paths["output_dir"] = tmpdir
    # Setup
    star = "test_star"
    star = star.upper()
    obsnum = "11"
    suffix = "_test"
    # Gen fake param file
    
    setup_iam_dirs(star)
    num = 20
    # Standard values
    teff = np.linspace(3000, 5000, num)
    logg = np.linspace(0.5, 6, num)
    feh = np.linspace(-3, 1, num)
    gamma = np.linspace(-20, 20, num)

    for chip in range(1, 4):
        #        "TEST_STAR - 11_2_iam_chisqr_results_test *.csv"
        fname = os.path.join(tmpdir, star, "iam", f"{star}-{obsnum}_{chip}_iam_chisqr_results{suffix}.csv")
        chi2 = chip + gamma + teff / logg
        npix = (985 - chip) * np.ones_like(teff)

        df = pd.DataFrame({'teff_1': teff, 'logg_1': logg, 'feh_1': feh,
                           'gamma': gamma, 'chi2': chi2, "npix": npix})
        df.to_csv(fname)
        # database_name = 'sqlite:///{0}'.format(fname)
        # engine = sa.create_engine(database_name)
        # df.to_sql('test_table', engine, if_exists='append')

    expected_db_name = os.path.join(tmpdir, star,
                                    "{0}-{1}_coadd_iam_chisqr_results{2}.db".format(star, obsnum, suffix))
    assert not os.path.exists(expected_db_name)
    # make 4 databases to add together()
    res = iam_db_main(star, obsnum, suffix, replace=False, verbose=True, chunksize=5, move=False)
    assert res is None
    assert os.path.exists(os.path.exists(expected_db_name))

    db_table = load_sql_table(expected_db_name)
    assert isinstance(db_table, pd.DataFrame)
    assert np.all(db_table.teff_1.values == teff)
    assert np.all(db_table.logg_1.values == logg)
    assert np.all(db_table.feh_1.values == feh)
    assert len(db_table) == num
    assert False


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


@pytest.mark.xfail()
@pytest.mark.parametrize("func", [smallest_chi2_values, ])
def test_analysis_functions_run(func, db_table, db_params):
    res = func(db_table, db_params)
    out, err = capsys.readouterr()
    assert res is None
    #    assert
    assert False
