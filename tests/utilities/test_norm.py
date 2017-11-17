import numpy as np
import pytest

from mingle.utilities.norm import (chi2_model_norms, continuum,
                              get_continuum_points,
                              local_normalization, spec_local_norm)


@pytest.mark.parametrize("method1, method2", [
    ("scalar", "linear"),
    ("quadratic", "scalar"),
    ("quadratic", "exponential")])
def test_local_normalization_methods_work(host, method1, method2):
    x, y = host.xaxis, host.flux
    norm_1 = local_normalization(x, y, splits=50, method=method1, top=5)
    norm_2 = local_normalization(x, y, splits=50, method=method2, top=5)

    assert not np.allclose(norm_1, norm_2)


def test_spec_local_norm_applies_local_normalization(host, norm_method):
    norm = spec_local_norm(host, method=norm_method, top=5)
    local_norm = local_normalization(host.xaxis, host.flux, method=norm_method, top=5)

    assert np.allclose(norm.flux, local_norm)


def test_manual_normalization():
    x = np.arange(1, 100)
    y = np.arange(1, 100)

    local_norm = local_normalization(x, y, splits=5, plot=False, top=5, method="linear")
    cont = continuum(x, y, splits=3, plot=False, top=2, method="linear")

    assert np.allclose(cont, y)
    assert np.allclose(local_norm, np.ones_like(y))


@pytest.mark.xfail()
def test_chi2_model_norms(host, tcm_model, norm_method):
    host.wav_select(2110.5, 2114.5)  # cut to avoid Nans from doppler shifts
    wave = host.xaxis
    obs = host.xaxis
    models = tcm_model(wave)

    chi2norm = chi2_model_norms(wave, obs, models, method=norm_method, splits=15, top=10)

    assert False


@pytest.mark.parametrize("splits", [13, 27, 50])
def test_shortening_array(splits):
    x = np.arange(2033)
    rem = len(x) % splits
    z = x[:-rem]

    while len(x) % splits != 0:
        x = x[:-1]

    assert len(x) == len(z)
    assert np.allclose(x, z)


@pytest.mark.parametrize("splits, top, size", [
    (10, 5, (10,)),  # zero remainder
    (11, 4, (11,)),
    (51, 5, (51,)),
    (100, 7, (100,))  # zero remainder
])
def test_get_continuum_points(splits, top, size):
    x = np.arange(200)
    x1, x2 = get_continuum_points(x, x, splits=splits, top=top)

    assert np.all(x1 == x2)
    assert x1.shape == size


@pytest.mark.parametrize("scale", [1, 2, 3, 10])
def test_continuum_scalar(scale):
    x = np.linspace(2000, 2100, 2000)
    y = scale * np.ones(2000)
    cont = continuum(x, y, method="scalar")
    assert np.allclose(np.mean(cont), scale)
    assert np.allclose(np.mean(y / cont), 1)


@pytest.mark.parametrize("x1, x0", [(0.1, .5), (0.002, .7), (0.08, 1)])
def test_continuum_linear(x1, x0):
    x = np.linspace(2000, 2100, 2000)
    y = x1 * x + x0

    cont = continuum(x, y, method="linear")
    assert np.allclose(np.mean(y / cont), 1)
    assert np.allclose(cont, y)


@pytest.mark.parametrize("x1, x0", [(0.1, .5), (0.002, .7), (0.08, 1)])
def test_continuum_exponential(x1, x0):
    x = np.linspace(2000, 2100, 2000)
    y = np.exp(x1 * x) + x0

    cont = continuum(x, y, method="exponential")
    assert np.allclose(np.mean(y / cont), 1)
    assert np.allclose(cont, y, 2)


from mingle.utilities.norm import arbitrary_rescale
@pytest.mark.parametrize("start, stop, step", [
    (0.8, 1.2, 0.1),
    (1.5, 0.7, -0.05)])
def test_arbitrary_rescale_arb_norm_is_np_arange(start, stop, step):
    """Checking it is an array."""
    model = np.random.random((4, 3, 5))
    new_model, arb_norm = arbitrary_rescale(model, start, stop, step)
    assert np.all(arb_norm == np.arange(start, stop, step))
    # check shape of new_model
    assert new_model.shape == (*model.shape, len(arb_norm))


@pytest.mark.parametrize("start, stop, step, expected", [
    (0.8, 1.2, 0.1, [0.8, 0.9, 1.0, 1.1]),
    (1.5, 1.1, -0.05, [1.5, 1.45, 1.40, 1.35, 1.3, 1.25, 1.2, 1.15])])
def test_arbitrary_rescale_arb_norm_examples(start, stop, step, expected):
    """testing with specfic examples."""
    model = np.random.random((3, 2))
    __, arb_norm = arbitrary_rescale(model, start, stop, step)
    assert np.allclose(arb_norm, expected)


@pytest.mark.parametrize("shape", [
    (1, 2, 3),
    (2,),
    (4, 2, 3, 4, 1, 8),
    (0,)
])
def test_arbitrary_rescale_on_different_size_models(shape):
    """Testing with specific examples."""
    model = np.random.random(shape)
    new_model, arb_norm = arbitrary_rescale(model, 0.8, 1.2, 0.05)
    assert new_model.shape == (*model.shape, len(arb_norm))