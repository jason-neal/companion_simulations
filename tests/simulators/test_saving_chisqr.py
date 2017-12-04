import os

import numpy as np
import pandas as pd

from simulators.bhm_module import save_full_bhm_chisqr
from simulators.iam_module import save_full_iam_chisqr
from simulators.tcm_module import save_full_tcm_chisqr


def test_save_full_ima_chisqr(tmpdir):
    savename = str(tmpdir.join("saving_test_iam_filename.csv"))
    params_1 = [5000, 4.5, 0.0]
    params_2 = [3000, 3.0, 0.0]
    rvs = np.arange(7, 10)
    gammas = np.arange(-2, 4)
    R, G = np.meshgrid(rvs, gammas, indexing="ij")
    results = R * G
    norms = results / 100
    npix = 1000
    alpha = 5  # ratio between par1 and par2

    res = save_full_iam_chisqr(savename, params_1, params_2,
                               alpha, rvs, gammas, results, norms, npix)
    assert res is None

    ### Now reload and probe result
    df = pd.read_csv(savename)

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
    savename = str(tmpdir.join("saving_test_bhm_filename.csv"))
    params_1 = [5000, 4.5, 0.0]
    params_2 = [3000, 3.0, 0.0]
    gammas = np.arange(-2, 4)
    G, = np.meshgrid(gammas, indexing="ij")
    results = G ** 2
    npix = 780
    norms = np.arange(len(G))
    xcorr = 7
    res = save_full_bhm_chisqr(savename, params_1, gammas, results,
                               arbitrary_norms=norms, npix=npix, xcorr_value=xcorr)
    assert res is None

    ### Now reload and probe
    df = pd.read_csv(savename)

    assert np.all(df.npix.values == npix)
    assert np.all(df.teff_1 == params_1[0])
    assert np.all(df.logg_1 == params_1[1])
    assert np.all(df.feh_1 == params_1[2])
    assert np.all(df.chi2 == df.gamma ** 2)
    assert np.all(gammas == df.gamma.values)
    assert np.all(norms == df.arbnorm.values)
    assert np.all(df.xcorr.values == xcorr)


def test_save_full_bhm_chisqr_with_xcorr_None(tmpdir):
    savename = str(tmpdir.join("saving_test_bhm_filename.csv"))
    params_1 = [5000, 4.5, 0.0]
    params_2 = [3000, 3.0, 0.0]
    gammas = np.arange(-2, 4)
    G, = np.meshgrid(gammas, indexing="ij")
    results = G ** 2
    npix = 780
    norms = np.arange(len(G))
    res = save_full_bhm_chisqr(savename, params_1, gammas, results,
                               arbitrary_norms=norms, npix=npix, xcorr_value=None)
    assert res is None

    ### Now reload and probe
    df = pd.read_csv(savename)

    assert np.all(df.npix.values == npix)
    assert np.all(df.teff_1 == params_1[0])
    assert np.all(df.logg_1 == params_1[1])
    assert np.all(df.feh_1 == params_1[2])
    assert np.all(df.chi2 == df.gamma ** 2)
    assert np.all(gammas == df.gamma.values)
    assert np.all(norms == df.arbnorm.values)
    assert np.all(df.xcorr.values == -9999999)


def test_save_full_tcm_chisqr(tmpdir):
    savename = str(tmpdir.join("saving_test_tcm_filename.csv"))
    params_1 = [5000, 4.5, 0.0]
    params_2 = [3000, 3.0, 0.0]
    alphas = np.arange(0.5, 1.1, 0.1)
    rvs = np.arange(7, 10)
    gammas = np.arange(-2, 4)
    A, R, G = np.meshgrid(alphas, rvs, gammas, indexing="ij")
    result_grid = R * G
    print(type(result_grid))
    npix = 1000

    res = save_full_tcm_chisqr(savename, params_1, params_2,
                               alphas, rvs, gammas, result_grid, npix)
    assert res is None

    ### Now reload and probe
    df = pd.read_csv(savename)
    assert np.all(df.npix.values == npix)
    assert np.allclose(sorted(list(set(df.alpha.values))), alphas)
    assert np.all(df.teff_1.values == params_1[0])
    assert np.all(df.logg_1.values == params_1[1])
    assert np.all(df.feh_1.values == params_1[2])
    assert np.all(df.teff_2.values == params_2[0])
    assert np.all(df.logg_2.values == params_2[1])
    assert np.all(df.feh_2.values == params_2[2])
    assert np.all(df.chi2.values == df.gamma * df.rv)
