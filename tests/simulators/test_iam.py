import os

import numpy as np
import pytest
from simulators.iam_module import (continuum_alpha, iam_helper_function, iam_wrapper,
                                   setup_iam_dirs, renormalization, observation_rv_limits, prepare_iam_model_spectra,
                                   )
from mingle.utilities.param_utils import target_params
from simulators.iam_script import parse_args
from spectrum_overload import Spectrum


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
def test_iam_wrapper(sim_config, host, comp, tmpdir):
    simulators = sim_config
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
def test_iam_wrapper_without_prefix(sim_config, host, comp, tmpdir):
    simulators = sim_config
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


def test_setup_dirs_creates_dirs(sim_config, tmpdir):
    simulators = sim_config
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
    assert parsed.n_jobs == 1
    assert parsed.error_off is False
    assert parsed.area_scale is True
    assert parsed.renormalize is False
    assert parsed.disable_wav_scale is False


def test_iam_script_parser_toggle():
    args = ["HDswitches", "02", "-c", "4", "-j", "3", "--suffix", "_test",
            "-n", "-a", "--disable_wav_scale", "--error_off"]
    parsed = parse_args(args)
    assert parsed.star == "HDswitches"
    assert parsed.obsnum == "02"
    assert parsed.suffix == "_test"
    assert parsed.area_scale is False
    assert parsed.n_jobs == 3
    assert parsed.chip == "4"
    assert parsed.renormalize is True
    assert parsed.disable_wav_scale is True
    assert parsed.error_off is True


@pytest.mark.parametrize("flag, result", [
    ("-v", False),   # Not and -x flag
    ("-x", True),
    ("--strict_mask", True)])
def test_iam_parser_toggle_strict_mask(flag, result):
    args = ["HDswitches", "02", flag]
    parsed = parse_args(args)
    assert parsed.star == "HDswitches"
    assert parsed.obsnum == "02"
    assert parsed.strict_mask is result


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
    with pytest.warns(UserWarning) as record:
        result = renormalization(host, models, normalize=False, method=method)
    assert len(record) == 1
    assert record[0].message.args[0] == "Not Scalar Re-normalizing to observations!"

    assert result.ndim == len(model_shape)
    assert result.shape == expected_shape


@pytest.mark.parametrize("method", ["", None, "quadratic", "poly"])
def test_renormalization_with_invalid_method(host, method):
    with pytest.raises(ValueError):
        renormalization(host.flux, host.flux, method=method, normalize=True)


@pytest.mark.parametrize("params,  expected", [
    ({"temp": 5000, "logg": 3.5, "fe_h": 0.0, "comp_temp": 2300},
     ([5000, 3.5, 0.0], [2300, 3.5, 0.0])),
    ({"temp": 5000, "logg": 4.5, "fe_h": -0.5, "comp_temp": 2400, "comp_logg": 5, "comp_fe_h": 0.5},
     ([5000, 4.5, -0.5], [2400, 5, 0.5])),
    ({"temp": 4500, "logg": 4.5, "fe_h": 0.0, "comp_temp": 100, "comp_logg": 3},
     ([4500, 4.5, 0.0], [100, 3, 0.0]))
])
def test_target_parameters_from_dict_iam_mode(params, expected):
    result = target_params(params, mode="iam")
    assert np.all(result[0] == expected[0])
    assert np.all(result[1] == expected[1])


@pytest.mark.parametrize("params,  expected", [
    ({"temp": 5000, "logg": 3.5, "fe_h": 0.0, "comp_temp": 2300},
     ([5000, 3.5, 0.0])),
    ({"temp": 5000, "logg": 4.5, "fe_h": -0.5, "comp_temp": 2400, "comp_logg": 5, "comp_fe_h": 0.5},
     ([5000, 4.5, -0.5])),
    ({"temp": 4500, "logg": 4.5, "fe_h": 0.0, "comp_temp": 100, "comp_logg": 3},
     ([4500, 4.5, 0.0]))
])
def test_target_parameters_from_dict_bhm_mode(params, expected):
    result = target_params(params, mode="bhm")
    assert np.all(result[0] == expected)
    assert np.all(result[1] == [])


def test_target_parameters_comp_not_in_file(params_1):
    host, comp = target_params(params_1, mode="iam")
    assert np.all(host == [5340, 4.65, -0.22])
    assert np.all(comp == [1733, 4.65, -0.22])


def test_target_parameters_with_comp_vals_in_file(params_2):
    host, comp = target_params(params_2, mode="iam")
    assert np.all(host == [5340, 4.65, -0.22])
    assert np.all(comp == [1733, 5.3, -0.4])


@pytest.mark.parametrize("mode", [None, "tcm", ""])
def test_target_parameters_invalid_mode(params_1, mode):
    with pytest.raises(ValueError):
        target_params(params_1, mode=mode)


def test_observation_rv_limits_with_zeros(comp):
    """Test limits given for zero RVs equal to  min delta of 1km/s"""
    limits = observation_rv_limits(comp, 0, 0)
    assert np.all(limits == [np.min(comp.xaxis) - 1.1, np.max(comp.xaxis) + 1.1])


def test_observation_rv_limits(comp):
    """Test that the limits extend parameter range."""
    limits = observation_rv_limits(comp, 5, 20)
    assert limits[0] <= np.min(comp.xaxis)
    assert limits[1] >= np.max(comp.xaxis)


@pytest.mark.parametrize("limits", [
    [2110, 2111],
    [2080, 2195]])
def test_prepare_iam_model_spectra(limits):
    """Assert spectra with correct parameters returned."""
    params1 = [5200, 4.5, 0.0]
    params2 = [2300, 5, 0.0]
    x, y = prepare_iam_model_spectra(params1, params2, limits=limits)
    assert isinstance(x, Spectrum)
    assert isinstance(y, Spectrum)
    # Check correct models are loaded
    assert x.header["PHXTEFF"] == params1[0]
    assert x.header["PHXLOGG"] == params1[1]
    assert x.header["PHXM_H"] == params1[2]
    assert y.header["PHXTEFF"] == params2[0]
    assert y.header["PHXLOGG"] == params2[1]
    assert y.header["PHXM_H"] == params2[2]
    assert x.xaxis[0] >= limits[0]
    assert y.xaxis[0] >= limits[0]
    assert x.xaxis[-1] <= limits[1]
    assert y.xaxis[-1] <= limits[1]


@pytest.mark.parametrize("limits", [
    [2110, 2111]])
def test_prepare_iam_model_spectra_with_warnings(limits):
    """Assert spectra with correct parameters are returned."""
    params1 = [5200, 4.5, 0.0]
    params2 = [2300, 5, 0.0]
    with pytest.warns(UserWarning) as record:
        x, y = prepare_iam_model_spectra(params1, params2, limits=limits,
                                         area_scale=False, wav_scale=False)
    assert len(record) == 2
    # Check that the message matches
    assert record[0].message.args[0] == "Not using area_scale. This is incorrect for paper."
    assert record[1].message.args[0] == "Not using wav_scale. This is incorrect for paper."

    assert isinstance(x, Spectrum)
    assert isinstance(y, Spectrum)
    # Check correct models are loaded
    assert x.header["PHXTEFF"] == params1[0]
    assert x.header["PHXLOGG"] == params1[1]
    assert x.header["PHXM_H"] == params1[2]
    assert y.header["PHXTEFF"] == params2[0]
    assert y.header["PHXLOGG"] == params2[1]
    assert y.header["PHXM_H"] == params2[2]
    assert x.xaxis[0] >= limits[0]
    assert y.xaxis[0] >= limits[0]
    assert x.xaxis[-1] <= limits[1]
    assert y.xaxis[-1] <= limits[1]
