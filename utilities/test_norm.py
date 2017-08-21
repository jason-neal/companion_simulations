import copy
import time

import numpy as np
import pytest

from models.broadcasted_models import one_comp_model, two_comp_model
from utilities.norm import (chi2_model_norms, chi2_model_norms2, continuum,
                            continuum2, get_continuum_points,
                            local_normalization, local_normalization2,
                            spec_local_norm, spec_local_norm2)
from utilities.phoenix_utils import (load_normalized_phoenix_spectrum,
                                     load_normalized_starfish_spectrum)


@pytest.fixture
def host():
    """Host spectrum fixture."""
    mod_spec = load_normalized_starfish_spectrum([5200, 4.50, 0.0], limits=[2100, 2105])
    return mod_spec


@pytest.fixture
def comp():
    """Companion spectrum fixture."""
    mod_spec = load_normalized_starfish_spectrum([2600, 4.50, 0.0], limits=[2100, 2105])
    return mod_spec


@pytest.fixture(params=["scalar", "linear", "quadratic", "exponential"])
def norm_method(request):
    return request.param


@pytest.fixture()
def tcm_model(host, comp):
    return two_comp_model(host.xaxis, host.flux, comp.xaxis,
                          alphas=[0.1, 0.2, 0.3], rvs=[-0.25, 0.25],
                          gammas=[1, 2, 3, 4])


def test_continuums_are_equvalient(host, norm_method):
    x, y = host.xaxis, host.flux

    start = time.time()
    cont = continuum(x, y, method=norm_method)
    mid = time.time()
    cont2 = continuum2(x, y, method=norm_method)
    end = time.time()
    assert np.allclose(cont, cont2)

    t1, t2 = mid - start, end - mid
    assert t2 < t1
    assert t2 < t1 * 0.15  # Reduction by 85%


def test_local_normalization_equivelent(host, norm_method):
    x, y = host.xaxis, host.flux

    assert np.allclose(local_normalization(x, y, splits=50, method=norm_method, plot=False, top=5),
                       local_normalization2(x, y, splits=50, method=norm_method, plot=False, top=5))


@pytest.mark.parametrize("method1,method2", [
    ("scalar", "linear"),
    ("quadratic", "scalar"),
    ("quadratic", "exponential")])
def test_local_normalization_methods_work(host, method1, method2):
    x, y = host.xaxis, host.flux

    assert not np.allclose(local_normalization(x, y, splits=50, method=method1, plot=False, top=5),
                           local_normalization(x, y, splits=50, method=method2, plot=False, top=5))


def test_spec_local_norm_equivalent(host, norm_method):
    norm = spec_local_norm(host, method=norm_method, top=5)
    norm2 = spec_local_norm2(host, method=norm_method, top=5)

    assert np.allclose(norm.xaxis, norm2.xaxis)
    assert np.allclose(norm.flux, norm2.flux)



def test_spec_local_norm_applies_local_normalization(host, norm_method):
    norm = spec_local_norm(host, method=norm_method, top=5)
    local_norm = local_normalization(host.xaxis, host.flux, method=norm_method, top=5)

    assert np.allclose(norm.flux, local_norm)


def test_manual_normalization():
    # 2 is the new version
    x = np.arange(1, 100)
    y = np.arange(1, 100)

    local_norm = local_normalization(x, y, splits=5, plot=False, top=5, method="linear")
    local_norm2 = local_normalization2(x, y, splits=5, plot=False, top=5, method="linear")
    print(local_norm)

    cont = continuum(x, y, splits=3, plot=False, top=2, method="linear")
    cont2 = continuum2(x, y, splits=3, plot=False, top=2, method="linear")
    print("cont", cont)
    print("cont2", cont2)
    assert np.allclose(cont, y)
    assert np.allclose(local_norm, np.ones_like(y))

    assert np.allclose(cont2, cont)
    assert np.allclose(local_norm, local_norm2)


def test_chi2_model_norms(host, tcm_model, norm_method):
    host.wav_select(2100.5, 2104.5)   # cut to avoid Nans from doppler shifts
    wave = host.xaxis
    obs = host.xaxis
    models = tcm_model(wave)
    # print(len(wave))
    # print("isnan?", np.any(np.isnan(wave)), np.any(np.isnan(host.flux)), np.any(np.isnan(models)))
    start = time.time()
    chi2norm = chi2_model_norms(wave, obs, models, method=norm_method, splits=15, top=10)
    mid = time.time()
    chi2norm2 = chi2_model_norms2(wave, obs, models, method=norm_method, splits=15, top=10)
    end = time.time()

    assert chi2norm.shape == chi2norm2.shape
    assert np.allclose(chi2norm, chi2norm2)

    t1, t2 = mid - start, end - mid
    assert t2 < t1
    assert t2 < t1 * 0.25   # Reduces time by 75 %


@pytest.mark.parametrize("splits", [13, 27, 50])
def test_shortening_array(splits):
    x = np.arange(2033)
    rem = len(x) % splits
    z = x[:-rem]

    while len(x) % splits != 0:
        x = x[:-1]

    assert len(x) == len(z)
    assert np.allclose(x, z)


@pytest.mark.parametrize("splits,top,size", [
    (10, 5, (10,)),   # zero remainder
    (11, 4, (11,)),
    (51, 5, (51,)),
    (100, 7, (100,))  # zero remainder
])
def test_get_continuum_points(splits, top, size):

    x = np.arange(200)
    x1, x2 = get_continuum_points(x, x, splits=splits, top=top)

    assert np.all(x1 == x2)
    assert x1.shape == size
