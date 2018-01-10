import os

import numpy as np
import pytest
from spectrum_overload import Spectrum

import simulators
from simulators.iam_module import (continuum_alpha, iam_analysis,
                                   iam_helper_function, iam_wrapper,
                                   setup_iam_dirs, renormalization)

from simulators.iam_script import parse_args


@pytest.mark.parametrize("star, obs, chip", [
    ("HD30501", 1, 1),
    ("HD4747", "a", 4)])
def test_iam_helper_function(star, obs, chip):
    obs_name, params, output_prefix = iam_helper_function(star, obs, chip)

    assert isinstance(obs_name, str)
    assert isinstance(output_prefix, str)
    assert str(star) in obs_name
    assert str(obs) in obs_name
    assert str(chip) in obs_name
    assert os.path.join(star, "iam", star) in output_prefix
    assert "iam_chisqr" in output_prefix
    assert params["name"] == star.lower()


@pytest.mark.xfail()
def test_iam_wrapper(host, comp, tmpdir):
    simulators.paths["output_dir"] = str(tmpdir)
    host_params = [5600, 4.5, 0.0]
    comp_params = [[2300, 4.5, 0.0], [2400, 4.5, 0.0]]

    host.wav_select(2110, 2115)
    comp.wav_select(2110, 2115)

    obs_spec = host.copy()
    obs_spec.flux += comp.flux
    obs_spec.header.update({"OBJECT": "Test_object"})
    setup_iam_dirs("Test_object")

    result = iam_wrapper(0, host_params, comp_params, obs_spec=obs_spec,
                         gammas=[0, 1, 2], rvs=[-1, 1], norm=True,
                         save_only=True, chip=1, prefix=tmpdir.join("TEST_file"))
    assert result is None


@pytest.mark.xfail()
def test_iam_wrapper_without_prefix(host, comp, tmpdir):
    simulators.paths["output_dir"] = str(tmpdir)
    host_params = [5600, 4.5, 0.0]
    comp_params = [[2300, 4.5, 0.0], [2400, 4.5, 0.0]]

    host.wav_select(2110, 2115)
    comp.wav_select(2110, 2115)

    obs_spec = host.copy()
    obs_spec.flux += comp.flux
    obs_spec.header.update({"OBJECT": "Test_object", "MJD-OBS": 56114.31674297})
    setup_iam_dirs("Test_object")

    result = iam_wrapper(0, host_params, comp_params, obs_spec=obs_spec,
                         gammas=[0, 1, 2], rvs=[-1, 1], norm=True,
                         save_only=True, chip=1)
    assert result is None


@pytest.mark.parametrize("chip", [None, 1, 2, 3, 4])
def test_continuum_alpha(chip):
    x = np.linspace(2100, 2180, 100)
    model1 = Spectrum(xaxis=x, flux=np.ones(len(x)))
    model2 = model1 * 2
    alpha = continuum_alpha(model1, model2, chip)

    assert np.allclose(alpha, [2])


def test_setup_dirs_creates_dirs(tmpdir):
    simulators.paths["output_dir"] = str(tmpdir)
    star = "TestStar"
    assert not tmpdir.join(star.upper(), "iam").check(dir=True)
    assert not tmpdir.join(star.upper(), "iam", "plots").check(dir=True)
    assert not tmpdir.join(star.upper(), "iam", "grid_plots").check(dir=True)
    assert not tmpdir.join(star.upper(), "iam", "fudgeplots").check(dir=True)
    result = setup_iam_dirs(star)

    assert tmpdir.join(star.upper(), "iam").check(dir=True)
    assert tmpdir.join(star.upper(), "iam", "plots").check(dir=True)
    assert tmpdir.join(star.upper(), "iam", "grid_plots").check(dir=True)
    assert tmpdir.join(star.upper(), "iam", "fudgeplots").check(dir=True)
    assert result is None


def test_iam_script_parser():
    parsed = parse_args(["HD30501", "01"])
    assert parsed.star == "HD30501"
    assert parsed.obsnum == "01"
    assert parsed.suffix is None
    assert parsed.small is False
    assert parsed.n_jobs == 1
    assert parsed.error_off is False
    assert parsed.area_scale is True
    assert parsed.renormalize is False
    assert parsed.disable_wav_scale is False


def test_iam_script_parser_toggle():
    args = ["HDswitches", "02", "-c", "4", "-j", "3", "--suffix", "_test",
            "-n", "-s", "-a", "--disable_wav_scale", "--error_off"]
    parsed = parse_args(args)
    assert parsed.star == "HDswitches"
    assert parsed.obsnum == "02"
    assert parsed.suffix == "_test"
    assert parsed.small is True
    assert parsed.area_scale is False
    assert parsed.n_jobs == 3
    assert parsed.chip == "4"
    assert parsed.renormalize is True
    assert parsed.disable_wav_scale is True
    assert parsed.error_off is True


def setup_renomlization_model(spectrum, model_shape):
    flux = spectrum.flux
    while flux.ndim < len(model_shape):
        flux = flux[:, np.newaxis]
    models = flux * np.ones(model_shape)
    return models


@pytest.mark.parametrize("method", ["scalar", "linear"])
@pytest.mark.parametrize("model_shape", [(1, 5, 7), (1, 5), (1, 4, 2, 8)])
def test_renormalization_on(host, method, model_shape):
    models = setup_renomlization_model(host, model_shape)
    result = renormalization(host, models, normalize=True, method=method)

    assert result.ndim == models.ndim
    assert result.ndim == len(model_shape)
    assert result.shape == models.shape


@pytest.mark.parametrize("method", ["scalar", "linear"])
@pytest.mark.parametrize("model_shape", [(1, 5, 7), (1, 59), (1, 4, 2, 8)])
def test_renormalization_off(host, method, model_shape):
    expected_shape = tuple(1 for _ in model_shape)
    expected_shape = (len(host.flux), *expected_shape[1:])
    models = setup_renomlization_model(host, model_shape)

    result = renormalization(host, models, normalize=False, method=method)

    assert result.ndim == len(model_shape)
    assert result.shape == expected_shape


@pytest.mark.parametrize("method", ["", None, "quadratic", "poly"])
def test_renormalization_with_invalid_method(host, method):
    with pytest.raises(ValueError):
        renormalization(host.flux, host.flux, method=method, normalize=True)
