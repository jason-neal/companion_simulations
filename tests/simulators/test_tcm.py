import os

import pytest

import simulators
from simulators.tcm_module import (tcm_helper_function, save_full_tcm_chisqr)


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
    assert os.path.join(star, star) in output_prefix
    assert "tcm_chisqr_results" in output_prefix
    assert params["name"] == star.lower()

