import os

import pandas as pd
import pytest

import simulators
from mingle.utilities.phoenix_utils import closest_model_params, generate_close_params
from simulators.bhm_module import (bhm_helper_function, get_model_pars, save_full_bhm_chisqr)


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
    assert os.path.join(star, star) in output_prefix
    assert "bhm_chisqr_results" in output_prefix
    assert params["name"] == star.lower()


