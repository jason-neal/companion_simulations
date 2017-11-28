import os

import pytest

import simulators
from mingle.utilities.phoenix_utils import closest_model_params, generate_close_params
from simulators.bhm_module import (bhm_helper_function, get_model_pars, setup_bhm_dirs)
from simulators.bhm_script import parse_args


def test_get_model_pars_close_method_returns_close_params():
    pars = get_model_pars({"temp": 5200, "logg": 4.5, "fe_h": 0.0}, method="close")
    assert pars == list(generate_close_params(closest_model_params(5200, 4.5, 0.0)))


def test_get_model_pars_all_notimplemented():
    with pytest.raises(NotImplementedError):
        get_model_pars({"temp": 5200, "logg": 4.5, "fe_h": 0.0}, method="all")


def test_get_model_pars_value_error_for_method():
    with pytest.raises(ValueError):
        get_model_pars({"temp": 5200, "logg": 4.5, "fe_h": 0.0}, method="some")


@pytest.mark.parametrize("star, obs, chip", [
    ("HD30501", 1, 1),
    ("HD4747", "a", 4)])
def test_bhm_helper_function(star, obs, chip):
    obs_name, params, output_prefix = bhm_helper_function(star, obs, chip)

    assert isinstance(obs_name, str)
    assert isinstance(output_prefix, str)
    assert simulators.paths["spectra"] in obs_name
    assert "-mixavg-tellcorr_" in obs_name
    assert str(star) in obs_name
    assert str(obs) in obs_name
    assert str(chip) in obs_name
    assert os.path.join(star, "bhm", star) in output_prefix
    assert "bhm_chisqr_results" in output_prefix
    assert params["name"] == star.lower()


def test_setup_bhm_dirs_creates_dirs(tmpdir):
    simulators.paths["output_dir"] = tmpdir
    star = "TestStar"
    assert not os.path.exists(os.path.join(tmpdir, star.upper()))
    assert not os.path.exists(os.path.join(tmpdir, star.upper(), "bhm", "plots"))
    # assert not os.path.exists(os.path.join(tmpdir, star.upper(), "bhm", "grid_plots"))
    # assert not os.path.exists(os.path.join(tmpdir, star.upper(), "bhm", "fudgeplots"))
    result = setup_bhm_dirs(star)

    assert os.path.exists(os.path.join(tmpdir, star.upper()))
    assert os.path.exists(os.path.join(tmpdir, star.upper(), "bhm", "plots"))
    # assert os.path.exists(os.path.join(tmpdir, star.upper(), "bhm", "grid_plots"))
    # assert os.path.exists(os.path.join(tmpdir, star.upper(), "bhm", "fudgeplots"))
    assert result is None


def test_bhm_script_parser():
    args = ["HD30501", "01"]
    parsed = parse_args(args)
    assert parsed.star == "HD30501"
    assert parsed.obsnum == "01"
    assert parsed.chip is None
    assert parsed.suffix is ""
    assert parsed.error_off is False
    assert parsed.disable_wav_scale is False


def test_bhm_script_parser_toggle():
    args = ["HDswitches", "1a", "-c", "4", "--suffix", "_test", "--disable_wav_scale", "--error_off"]
    parsed = parse_args(args)
    assert parsed.star == "HDswitches"
    assert parsed.obsnum == "1a"
    assert parsed.chip == "4"
    assert parsed.suffix is "_test"
    assert parsed.error_off is True
    assert parsed.disable_wav_scale is True
