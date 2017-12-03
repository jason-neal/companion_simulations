import os

import numpy as np
import pandas as pd
import sqlalchemy as sa

import simulators
from bin.coadd_analysis_script import load_sql_table
from bin.coadd_bhm_db import main as bhm_db_main
from bin.coadd_bhm_db import parse_args
from simulators.bhm_module import setup_bhm_dirs


def test_bhm_db_main(tmpdir):
    simulators.paths["output_dir"] = tmpdir

    # make directory !
    # Setup
    star = "test_star"
    star = star.upper()
    obsnum = "11"
    suffix = "_test"
    # Gen fake param file
    setup_bhm_dirs(star)
    num = 20
    # Standard values
    teff = np.linspace(3000, 5000, num)
    logg = np.linspace(0.5, 6, num)
    feh = np.linspace(-3, 1, num)
    gamma = np.linspace(-20, 20, num)
    print(tmpdir.join(star, "bhm"))
    # assert os.path.exists(tmpdir.join(f"{star}", "bhm"))

    for chip in range(1, 5):
        # "TEST_STAR - 11_2_bhm_chisqr_results_test *.csv"
        fname = os.path.join(tmpdir, star, "bhm", "{0}-{1}_{2}_bhm_chisqr_results{3}.csv".format(star, obsnum, chip, suffix))
        print("fname", fname)
        chi2 = chip + gamma + teff / logg
        npix = (985 - chip) * np.ones_like(teff)

        df = pd.DataFrame({'teff_1': teff, 'logg_1': logg, 'feh_1': feh,
                           'gamma': gamma, 'chi2': chi2, "npix": npix})
        df.to_csv(fname)
        # database_name = 'sqlite:///{0}'.format(fname)
        # engine = sa.create_engine(database_name)
        # df.to_sql('test_table', engine, if_exists='append')

    expected_db_name = os.path.join(tmpdir, star, "bhm",
                                    "{0}-{1}_coadd_bhm_chisqr_results{2}.db".format(star, obsnum, suffix))
    assert not os.path.exists(expected_db_name)
    # make 4 databases to add together()
    res = bhm_db_main(star, obsnum, suffix, replace=False, verbose=True, chunksize=5, move=False)
    assert res is None
    assert os.path.exists(os.path.exists(expected_db_name))

    db_table = load_sql_table(expected_db_name)
    assert isinstance(db_table, sa.Table)

    df = pd.read_sql(
        sa.select(db_table.c),
        db_table.metadata.bind)

    assert isinstance(df, pd.DataFrame)
    assert np.allclose(df.teff_1.values, teff)
    assert np.allclose(df.logg_1.values, logg)
    assert np.allclose(df.feh_1.values, feh)
    assert np.allclose(df.gamma.values, gamma)
    assert len(df) == num
    x = gamma + teff / logg
    assert np.allclose(df.chi2_1, 1 + x)
    assert np.allclose(df.chi2_2, 2 + x)
    assert np.allclose(df.chi2_3, 3 + x)
    assert np.allclose(df.chi2_4, 4 + x)
    assert np.allclose(df.coadd_chi2, 10 + 4 * x)
    assert np.all(df.npix_1 == (985 - 1))
    assert np.all(df.npix_2 == (985 - 2))
    assert np.all(df.npix_3 == (985 - 3))
    assert np.all(df.npix_4 == (985 - 4))


def test_coadd_bhm_db_parser_defaults():
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


def test_coadd_bhm_db_parser():
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
