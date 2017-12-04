import os

import pytest

import simulators
from simulators.tcm_module import (tcm_helper_function, setup_tcm_dirs)
from simulators.tcm_script import parse_args


@pytest.mark.parametrize("star, obs, chip", [
    ("HD30501", 1, 1),
    ("HD4747", "a", 4)])
def test_tcm_helper_function(star, obs, chip):
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


def test_setup_tcm_dirs_creates_dirs(tmpdir):
    simulators.paths["output_dir"] = str(tmpdir)
    star = "TestStar"
    assert not tmpdir.join(star.upper()).check()
    assert not tmpdir.join(star.upper(), "tcm", "plots").check()
    # assert not tmpdir.join(star.upper(), "tcm", "grid_plots").check(dir=True)
    # assert not tmpdir.join(star.upper(), "tcm", "fudgeplots").check(dir=True)
    result = setup_tcm_dirs(star)

    assert tmpdir.join(star.upper()).check(dir=True)
    assert tmpdir.join(star.upper(), "tcm", "plots").check(dir=True)
    # assert tmpdir.join(star.upper(), "tcm", "grid_plots").check(dir=True)
    # assert tmpdir.join(star.upper(), "tcm", "fudgeplots").check(dir=True)
    assert result is None


def test_tcm_script_parser():
    parsed = parse_args([])
    # assert parsed.star == "HD30501"
    # assert parsed.obsnum == "01"
    assert parsed.chip is None
    assert parsed.small is False
    assert parsed.error_off is False
    assert parsed.disable_wav_scale is False
    assert parsed.parallel is False


def test_tcm_script_parser_toggle():
    args = ["--chip", "2", "-p", "-s", "--error_off", "--disable_wav_scale"]
    parsed = parse_args(args)
    # assert parsed.star == "HD30501"
    # assert parsed.obsnum == "01"
    assert parsed.chip is "2"
    assert parsed.small is True
    assert parsed.error_off is True
    assert parsed.disable_wav_scale is True
    assert parsed.parallel is True
