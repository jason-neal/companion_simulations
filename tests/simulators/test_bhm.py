import pytest
import os
from simulators.bhm_module import (bhm_helper_function, deconstruct_array, get_model_pars)
import simulators

@pytest.mark.xfail()
def test_save_pd_cvs(tmpdir):
    assert 0


def test_get_model_pars():
    pars = get_model_pars({"temp": 5200, "logg": 4.5, "fe_h": 0.0})
    assert pars == False
    assert False


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


def test_deconstruct_array():
        deconstruct_array()
        assert False

        def deconstruct_array_a(array, values):
            """Index of other arrays to apply these values to."""
            print("array shape", array.shape)
            print("array[:5]", array[:5])
            print("values.shape", values.shape)
            values2 = values * np.ones_like(array)
            print("values2.shape", values2.shape)
            print("values2.shape", values2[:5])
            for i in enumerate(array):
                indx = [0]
            gam = [0]
            chi2 = [0]
            return indx, gam, chi2
