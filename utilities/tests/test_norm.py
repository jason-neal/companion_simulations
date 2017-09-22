import time

import numpy as np
import pytest

from models.broadcasted_models import two_comp_model
from utilities.norm import (chi2_model_norms, chi2_model_norms_slow,
                            continuum, continuum_slow, get_continuum_points,
                            local_normalization, local_normalization_slow,
                            spec_local_norm, spec_local_norm_slow)
from utilities.phoenix_utils import load_starfish_spectrum


@pytest.fixture
def host():
    """Host spectrum fixture."""
    mod_spec = load_starfish_spectrum([5200, 4.50, 0.0], limits=[2100, 2105], normalize=True)
    return mod_spec


@pytest.fixture
def comp():
    """Companion spectrum fixture."""
    mod_spec = load_starfish_spectrum([2600, 4.50, 0.0], limits=[2100, 2105], normalize=True)
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
    cont_s = continuum_slow(x, y, method=norm_method)
    mid = time.time()
    cont = continuum(x, y, method=norm_method)
    end = time.time()
    t1, t2 = mid - start, end - mid

    assert np.allclose(cont_s, cont)
    assert t2 < t1
    assert t2 < t1 * 0.20  # Reduction by 80%


def test_local_normalization_equivelent(host, norm_method):
    x, y = host.xaxis, host.flux

    assert np.allclose(local_normalization_slow(x, y, splits=50, method=norm_method, plot=False, top=5),
                       local_normalization(x, y, splits=50, method=norm_method, plot=False, top=5))


@pytest.mark.parametrize("method1,method2", [
    ("scalar", "linear"),
    ("quadratic", "scalar"),
    ("quadratic", "exponential")])
def test_local_normalization_methods_work(host, method1, method2):
    x, y = host.xaxis, host.flux
    norm_1 = local_normalization(x, y, splits=50, method=method1, top=5)
    norm_2 = local_normalization(x, y, splits=50, method=method2, top=5)

    assert not np.allclose(norm_1, norm_2)


def test_spec_local_norm_equivalent(host, norm_method):
    norm = spec_local_norm(host, method=norm_method, top=5)
    norm_s = spec_local_norm_slow(host, method=norm_method, top=5)

    assert np.allclose(norm.xaxis, norm_s.xaxis)
    assert np.allclose(norm.flux, norm_s.flux)


def test_spec_local_norm_applies_local_normalization(host, norm_method):
    norm = spec_local_norm_slow(host, method=norm_method, top=5)
    local_norm = local_normalization_slow(host.xaxis, host.flux, method=norm_method, top=5)

    assert np.allclose(norm.flux, local_norm)


def test_manual_normalization():
    x = np.arange(1, 100)
    y = np.arange(1, 100)

    local_norm = local_normalization(x, y, splits=5, plot=False, top=5, method="linear")
    local_norm_s = local_normalization_slow(x, y, splits=5, plot=False, top=5, method="linear")

    cont = continuum(x, y, splits=3, plot=False, top=2, method="linear")
    cont_s = continuum_slow(x, y, splits=3, plot=False, top=2, method="linear")

    assert np.allclose(cont_s, y)
    assert np.allclose(local_norm, np.ones_like(y))

    assert np.allclose(cont_s, cont)
    assert np.allclose(local_norm, local_norm_s)


def test_chi2_model_norms(host, tcm_model, norm_method):
    host.wav_select(2100.5, 2104.5)   # cut to avoid Nans from doppler shifts
    wave = host.xaxis
    obs = host.xaxis
    models = tcm_model(wave)

    start = time.time()
    chi2norm_s = chi2_model_norms_slow(wave, obs, models, method=norm_method, splits=15, top=10)
    mid = time.time()
    chi2norm = chi2_model_norms(wave, obs, models, method=norm_method, splits=15, top=10)
    end = time.time()

    assert chi2norm_s.shape == chi2norm.shape
    assert np.allclose(chi2norm_s, chi2norm)

    t1, t2 = mid - start, end - mid
    assert t2 < t1
    assert t2 < t1 * 0.50   # Reduces time by 50 %


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
