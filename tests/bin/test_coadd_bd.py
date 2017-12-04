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
from mingle.utilities import list_files


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


def test_sql_table_with_no_table(tmpdir):
    db_name = tmpdir.join("db_no_tables.db")
    with pytest.raises(ValueError):
        load_sql_table(db_name, echo=False, verbose=False)


@pytest.mark.xfail()
def test_sql_table_with_more_than_one_table(tmpdir):
    db_name = tmpdir.join("db with many tables.db")
    assert False  # need to make the db

    with pytest.raises(ValueError):
        load_sql_table(db_name, echo=False, verbose=False)


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
        sa.select([db_table.c["x"], db_table.c["y"], db_table.c["z"]]),
        db_table.metadata.bind)

    # NEED to query out x from database inot dataframe.
    # select x, y, z from db_tbale using sqlalchemy
    assert np.all(df.x.values == x)
    assert np.all(df.y.values == y)
    assert np.all(df.z.values == z)


from simulators.iam_module import setup_iam_dirs


def test_iam_db_main_single_host_model(tmpdir):
    simulators.paths["output_dir"] = str(tmpdir)
    # Setup
    star = "test_star"
    star = star.upper()
    obsnum = "11"
    suffix = "_test"
    # Gen fake param file

    setup_iam_dirs(star)
    list_files(str(tmpdir))
    num = 20

    # Setting values
    teff = 3000
    logg = 4.5
    feh = 0.0
    teff2 = np.linspace(2300, 4300, num)
    logg2 = np.linspace(1.5, 5, num)
    feh2 = np.linspace(-2, 2, num)
    rv = np.linspace(-15, 15, num)
    gamma = np.linspace(-20, 20, num)

    for chip in range(1, 5):
        fname = os.path.join(tmpdir, star, "iam",
                             "{0}-{1}_{2}_iam_chisqr_results{3}[{4}_{5}_{6}].csv".format(
                                 star, obsnum, chip, suffix, teff, logg, feh))
        chi2 = chip + (feh + gamma + teff / logg) * (feh2 + rv + teff2 / logg2)
        npix = (985 - chip) * np.ones_like(teff)

        df = pd.DataFrame({'gamma': gamma, 'teff_2': teff2, 'logg_2': logg2, 'feh_2': feh2, "rv": rv,
                           'chi2': chi2, "npix": npix})
        df.to_csv(fname)

    list_files(str(tmpdir))
    expected_db_name = os.path.join(tmpdir, star, "iam",
                                    "{0}-{1}_coadd_iam_chisqr_results{2}.db".format(star, obsnum, suffix))
    assert not os.path.exists(expected_db_name)
    # make 4 databases to add together()
    res = iam_db_main(star, obsnum, suffix, replace=False, verbose=True, chunksize=5, move=False)
    assert res is None
    assert os.path.exists(os.path.exists(expected_db_name))

    db_table = load_sql_table(expected_db_name)
    assert isinstance(db_table, sa.Table)
    df = pd.read_sql(
        sa.select(db_table.c), db_table.metadata.bind)

    assert isinstance(df, pd.DataFrame)
    assert np.all(df.teff_1.values == teff)
    assert np.all(df.logg_1.values == logg)
    assert np.all(df.feh_1.values == feh)
    assert np.allclose(df.teff_2.values, teff2)
    assert np.allclose(df.logg_2.values, logg2)
    assert np.allclose(df.feh_2.values, feh2)
    assert np.allclose(df.gamma.values, gamma)
    assert np.allclose(df.rv.values, rv)
    assert len(df) == num

    x = (feh + gamma + teff / logg) * (feh2 + rv + teff2 / logg2)
    assert np.allclose(df.chi2_1, 1 + x)
    assert np.allclose(df.chi2_2, 2 + x)
    assert np.allclose(df.chi2_3, 3 + x)
    assert np.allclose(df.chi2_4, 4 + x)
    assert np.allclose(df.coadd_chi2, 10 + 4 * x)
    assert np.all(df.npix_1 == (985 - 1))
    assert np.all(df.npix_2 == (985 - 2))
    assert np.all(df.npix_3 == (985 - 3))
    assert np.all(df.npix_4 == (985 - 4))


def test_iam_db_main_multiple_host_model(tmpdir):
    simulators.paths["output_dir"] = str(tmpdir)
    # Setup
    star = "test_star"
    star = star.upper()
    obsnum = "11"
    suffix = "_test"
    # Gen fake param file
    print("before dirs")
    list_files(str(tmpdir))
    setup_iam_dirs(star)

    print("after dirs")
    list_files(str(tmpdir))
    num = 20
    # Standard values
    teff = np.linspace(3000, 5000, 4)
    logg = np.linspace(3.5, 4.5, 3)
    feh = np.linspace(-0.5, 0.5, 2)
    teff2 = np.linspace(2300, 4300, num)
    logg2 = np.linspace(1.5, 5, num)
    feh2 = np.linspace(-2, 2, num)
    rv = np.linspace(-15, 15, num)
    gamma = np.linspace(-20, 20, num)
    import itertools

    for chip in range(1, 5):
        for t, l, f in itertools.product(teff, logg, feh):
            fname = os.path.join(tmpdir, star, "iam",
                                 "{0}-{1}_{2}_iam_chisqr_results{3}[{4}_{5}_{6}].csv".format(star, obsnum, chip, suffix,
                                                                                             t, l, f))
            chi2 = chip + (f + gamma + t / l) * (feh2 + rv + teff2 / logg2)
            npix = (985 - chip) * np.ones_like(chi2)

            # print("chi2 shape", chi2.shape)
            # print("tshape", t.shape)
            # print("tgamma shape", gamma.shape)

            df = pd.DataFrame({'gamma': gamma,
                               'teff_2': teff2, 'logg_2': logg2, 'feh_2': feh2, "rv": rv,
                               'chi2': chi2, "npix": npix})
            df["teff_1"] = t
            df["logg_1"] = l
            df["feh_1"] = f
            df.to_csv(fname)

    print("after df.to_csv")
    list_files(str(tmpdir))
    expected_db_name = os.path.join(tmpdir, star, "iam",
                                    "{0}-{1}_coadd_iam_chisqr_results{2}.db".format(star, obsnum, suffix))
    assert not os.path.exists(expected_db_name)
    # make 4 databases to add together()
    res = iam_db_main(star, obsnum, suffix, replace=False, verbose=False, chunksize=5,
                      move=False)  # move=True does not test well.
    print("After iam db main")
    assert res is None
    assert os.path.exists(os.path.exists(expected_db_name))

    db_table = load_sql_table(expected_db_name)
    assert isinstance(db_table, sa.Table)
    df = pd.read_sql(
        sa.select(db_table.c), db_table.metadata.bind)

    print("df head", df.head())
    print("types", df.dtypes)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == num * (len(teff) * len(feh) * len(logg))

    x = (df.feh_1 + df.gamma + df.teff_1 / df.logg_1) * (df.feh_2 + df.rv + df.teff_2 / df.logg_2)
    assert np.allclose(df.chi2_1, 1 + x)
    assert np.allclose(df.chi2_2, 2 + x)
    assert np.allclose(df.chi2_3, 3 + x)
    assert np.allclose(df.chi2_4, 4 + x)
    assert np.allclose(df.coadd_chi2, 10 + 4 * x)
    assert np.all(df.npix_1 == (985 - 1))
    assert np.all(df.npix_2 == (985 - 2))
    assert np.all(df.npix_3 == (985 - 3))
    assert np.all(df.npix_4 == (985 - 4))

    assert np.allclose(np.unique(df.teff_1.values), teff)
    assert np.allclose(np.unique(df.logg_1.values), logg)
    assert np.allclose(np.unique(df.feh_1.values), feh)
    assert np.allclose(np.unique(df.teff_2.values), teff2)
    assert np.allclose(np.unique(df.logg_2.values), logg2)
    assert np.allclose(np.unique(df.feh_2.values), feh2)
    assert np.allclose(np.unique(df.gamma.values), gamma)
    assert np.allclose(np.unique(df.rv.values), rv)


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
