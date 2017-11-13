import numpy as np
import pytest

from simulators.iam_module import (continuum_alpha, iam_analysis,
                                   iam_helper_function, iam_wrapper,
                                   parallel_iam_analysis)
from spectrum_overload import Spectrum


@pytest.mark.parametrize("star,obs,chip", [
    ("HD30501", 1, 1),
    ("HD4747", "a", 4)])
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

    host.wav_select(2110, 2115)
    comp.wav_select(2110, 2115)

    obs_spec = host.copy()
    obs_spec.flux += comp.flux
    obs_spec.header.update({"OBJECT": "Test_object"})
    result = iam_wrapper(0, host_params, comp_params, obs_spec=obs_spec,
                         gammas=[0, 1, 2], rvs=[-1, 1], norm=True,
                         save_only=True, chip=1, prefix="Testtestest")
    assert result is None


@pytest.mark.parametrize("chip", [None, 1, 2, 3, 4])
def test_continuum_alpha(chip):
    x = np.linspace(2100, 2180, 100)
    model1 = Spectrum(xaxis=x, flux=np.ones(len(x)))
    model2 = model1 * 2
    alpha = continuum_alpha(model1, model2, chip)

    assert np.allclose(alpha, [2])
