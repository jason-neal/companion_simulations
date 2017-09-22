import pytest
import numpy as np
from spectrum_overload.Spectrum import Spectrum
from utilities.simulation_utilities import check_inputs
from simulators.iam_module import (iam_helper_function, iam_analysis,
                                   parallel_iam_analysis, iam_wrapper,
                                   save_full_iam_chisqr, continuum_alpha)
from utilities.phoenix_utils import load_starfish_spectrum


@pytest.fixture
def host():
    """Host spectrum fixture."""
    mod_spec = load_starfish_spectrum([5200, 4.50, 0.0], limits=[2100, 2105], normalize=True)
    return mod_spec


@pytest.fixture
def comp():
    """Noramlized Companion spectrum fixture."""
    mod_spec = load_starfish_spectrum([2600, 4.50, 0.0], limits=[2100, 2105], normalize=True)
    return mod_spec

@pytest.mark.parametrize("star,obs,chip", [
    ("HD30501", 1, 1),("HD4747", "a", 4)])
def test_iam_helper_function(star, obs, chip):
    obs_name, params, output_prefix = iam_helper_function(star, obs, chip)

    assert isinstance(obs_name, str)
    assert isinstance(output_prefix, str)
    assert str(star) in obs_name
    assert str(obs) in obs_name
    assert str(chip) in obs_name
    assert "{0}/{0}".format(star) in output_prefix
    assert "iam_chisqr" in output_prefix
    assert params["name"] == star.lower()


@pytest.mark.xfail()
def test_iam_check_inputs():
    assert 0


@pytest.mark.xfail()
def test_save_full_iam_chisqr():
    assert 0


@pytest.mark.xfail()
def test_iam_analysis_same_as_parallel():
    assert parallel_iam_analysis() == iam_analysis()


def test_iam_wrapper(host, comp):
    host_params = [5600, 4.5, 0.0]
    comp_params = [[2300, 4.5, 0.0], [2400, 4.5, 0.0]]

    obs_spec = host
    obs_spec.flux += comp.flux

    result = iam_wrapper(0, host_params, comp_params, obs_spec=obs_spec,
                         gammas=[0, 1, 2], rvs=[-1, 1], norm=False,
                         save_only=False, chip=1, prefix="Testtestest")
    assert 0


@pytest.mark.parametrize("chip", [None, 1, 2, 3, 4])
def test_continuum_alpha(chip):
    x = np.linspace(2100, 2180, 100)
    model1 = Spectrum(xaxis=x, flux=np.ones(len(x)))
    model2 = model1 * 2
    alpha = continuum_alpha(model1, model2, chip)

    assert np.allclose(alpha, 2)