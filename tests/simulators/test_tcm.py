import os

import pytest

import simulators
from simulators.tcm_module import (tcm_helper_function, save_full_tcm_chisqr, setup_tcm_dirs)


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


def test_setup_dirs_creates_dirs(tmpdir):
    simulators.paths["output_dir"] = tmpdir
    star = "TestStar"
    assert not os.path.exists(os.path.join(tmpdir, star.upper()))
    assert not os.path.exists(os.path.join(tmpdir, star.upper(), "tcm", "plots"))
    # assert not os.path.exists(os.path.join(tmpdir, star.upper(), "tcm", "grid_plots"))
    # assert not os.path.exists(os.path.join(tmpdir, star.upper(), "tcm", "fudgeplots"))
    result = setup_tcm_dirs(star)

    assert os.path.exists(os.path.join(tmpdir, star.upper()))
    assert os.path.exists(os.path.join(tmpdir, star.upper(), "tcm", "plots"))
    # assert os.path.exists(os.path.join(tmpdir, star.upper(), "tcm", "grid_plots"))
    # assert os.path.exists(os.path.join(tmpdir, star.upper(), "tcm", "fudgeplots"))
    assert result is None