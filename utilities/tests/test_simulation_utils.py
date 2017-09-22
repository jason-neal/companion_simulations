import numpy as np
import pytest

from utilities.simulation_utilities import check_inputs, spec_max_delta, max_delta

c = 299792.458


@pytest.mark.parametrize("input,expected", [
    (None, np.ndarray([0])),
    ([0], np.array([0])),
    (1, np.array([1])),
    (range(5), np.array([0,1,2,3,4]))
])
def test_check_inputs(input, expected):
    assert np.allclose(check_inputs(input), expected)

from spectrum_overload.Spectrum import Spectrum


@pytest.mark.parametrize("xaxis,rv,gamma", [
    ([1, 2, 3, 4, 5], 3, 5),
    ([1.1, 1.2, 1.3, 1.4, 1.5], 0, -7.1)
])
def test_spec_max_delta_applies_max_delta_on_xaxis(xaxis, rv, gamma):
    spec = Spectrum(xaxis=xaxis, flux=np.ones(len(xaxis)))

    assert spec_max_delta(spec, rv, gamma) == max_delta(xaxis, rv, gamma)


@pytest.mark.parametrize("wav, rv, gamma, expected", [
    ([1], 3, 5, 5/c),
    ([2], 1, -7.1, 2*7.1/c),
    (2, 1, 7.1, 2*7.1/c),
    ([1, 2], 0, 0, 0)
])
def test_spec_max_delta(wav, rv, gamma, expected):

    assert 2*round(expected, 3) == max_delta(wav, rv, gamma)
