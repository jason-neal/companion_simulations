import os

import pytest

import simulators
from simulators.tcm_module import (tcm_helper_function, setup_tcm_dirs)
from simulators.tcm_script import parse_args


@pytest.mark.parametrize("star, obs, chip", [
    ("HD30501", 1, 1),
    ("HD4747", "a", 4)])
def test_tcm_helper_function(sim_config, star, obs, chip):
    simulators = sim_config
    obs_name, params, output_prefix = tcm_helper_function(star, obs, chip)

    assert isinstance(obs_name, str)
    assert isinstance(output_prefix, str)
    assert simulators.paths["spectra"] in obs_name
    assert "-mixavg-tellcorr_" in obs_name
    assert str(star) in obs_name
    assert str(obs) in obs_name
    assert str(chip) in obs_name
    assert os.path.join(star, "tcm", star) in output_prefix
    assert "tcm_chisqr_results" in output_prefix
    assert params["name"] == star.lower()


def test_setup_tcm_dirs_creates_dirs(sim_config, tmpdir):
    simulators = sim_config
    simulators.paths["output_dir"] = str(tmpdir)
    star = "TestStar"
    assert not tmpdir.join(star.upper()).check()
    assert not tmpdir.join(star.upper(), "tcm", "plots").check()
    result = setup_tcm_dirs(star)

    assert tmpdir.join(star.upper()).check(dir=True)
    assert tmpdir.join(star.upper(), "tcm", "plots").check(dir=True)
    assert result is None


def test_tcm_script_parser():
    parsed = parse_args([])
    assert parsed.chip is None
    assert parsed.error_off is False
    assert parsed.disable_wav_scale is False


def test_tcm_script_parser_toggle():
    args = ["--chip", "2", "--error_off", "--disable_wav_scale"]
    parsed = parse_args(args)
    assert parsed.chip is "2"
    assert parsed.error_off is True
    assert parsed.disable_wav_scale is True


@pytest.mark.parametrize("flag, result", [
    ("-v", False),  # Not and -x flag
    ("-x", True),
    ("--strict_mask", True)])
def test_tcm_parser_toggle_strict_mask(flag, result):
    args = [flag]
    parsed = parse_args(args)
    assert parsed.strict_mask is result
