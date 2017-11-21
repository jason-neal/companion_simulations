import os

import numpy as np
import pandas as pd
import pytest

from simulators.bhm_module import save_full_bhm_chisqr
from simulators.iam_module import save_full_iam_chisqr
from simulators.tcm_module import save_full_tcm_chisqr


def test_save_full_ima_chisqr(tmpdir):
    params_1 = [5000, 4.5, 0.0]
    params_2 = [3000, 3.0, 0.0]
    rvs = range(7, 10)
    gammas = range(-2, 4)
    R, G = np.meshgrid(rvs, gammas, indexing="ij")
    results = R * G
    norms = results / 100
    npix = 1000
    alpha = 5  # ratio between par1 and par2

    savename = os.path.join(tmpdir, "saving_test_iam_filename.csv")
    save_full_iam_chisqr(savename, params_1, params_2,
                         alpha, rvs, gammas, results, norms, npix)

    ### Now reload and probe result
    df = pd.read_csv(savename)
    print(df.head())

    assert np.all(df.npix == npix)
    assert np.all(df.alpha == alpha)
    assert np.all(df.teff_1 == params_1[0])
    assert np.all(df.logg_1 == params_1[1])
    assert np.all(df.feh_1 == params_1[2])
    assert np.all(df.teff_2 == params_2[0])
    assert np.all(df.logg_2 == params_2[1])
    assert np.all(df.feh_2 == params_2[2])
    assert np.all(df.chi2 == df.gamma * df.rv)


def test_save_full_bhm_chisqr(tmpdir):
    params_1 = [5000, 4.5, 0.0]
    params_2 = [3000, 3.0, 0.0]
    rvs = range(7, 10)
    gammas = range(-2, 4)
    R, G = np.meshgrid(rvs, gammas, indexing="ij")
    results = R * G
    norms = results / 100
    npix = 1000
    alpha = 5  # ratio between par1 and par2

    savename = os.path.join(tmpdir, "saving_test_iam_filename.csv")
    save_full_bhm_chisqr(savename, params_1, params_2,
                         alpha, rvs, gammas, results, norms, npix)

    ### Now reload and probe
    df = pd.read_csv(savename)
    print(df.head())

    assert np.all(df.npix == npix)
    assert np.all(df.alpha == alpha)
    assert np.all(df.teff_1 == params_1[0])
    assert np.all(df.logg_1 == params_1[1])
    assert np.all(df.feh_1 == params_1[2])
    assert np.all(df.teff_2 == params_2[0])
    assert np.all(df.logg_2 == params_2[1])
    assert np.all(df.feh_2 == params_2[2])
    assert np.all(df.chi2 == df.gamma * df.rv)


def test_save_full_tcm_chisqr(tmpdir):
    params_1 = [5000, 4.5, 0.0]
    params_2 = [3000, 3.0, 0.0]
    rvs = range(7, 10)
    gammas = range(-2, 4)
    R, G = np.meshgrid(rvs, gammas, indexing="ij")
    results = R * G
    norms = results / 100
    npix = 1000
    alpha = 5  # ratio between par1 and par2

    savename = os.path.join(tmpdir, "saving_test_iam_filename.csv")
    res = save_full_tcm_chisqr(savename, params_1, params_2,
                         alpha, rvs, gammas, results, norms, npix)
    assert res is None
    ### Now reload and probe
    df = pd.read_csv(savename)
    print(df.head())

    assert np.all(df.npix == npix)
    assert np.all(df.alpha == alpha)
    assert np.all(df.teff_1 == params_1[0])
    assert np.all(df.logg_1 == params_1[1])
    assert np.all(df.feh_1 == params_1[2])
    assert np.all(df.teff_2 == params_2[0])
    assert np.all(df.logg_2 == params_2[1])
    assert np.all(df.feh_2 == params_2[2])
    assert np.all(df.chi2 == df.gamma * df.rv)

