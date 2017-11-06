import numpy as np
import pytest

from spectrum_overload import Spectrum
from utilities.simulation_utilities import (check_inputs, max_delta,
                                            spec_max_delta)

c = 299792.458


@pytest.mark.parametrize("xaxis,rv,gamma", [
    ([1, 2, 3, 4, 5], 3, 5),
    ([1.1, 1.2, 1.3, 1.4, 1.5], 0, -7.1)
])
def test_spec_max_delta_applies_max_delta_on_xaxis(xaxis, rv, gamma):
    spec = Spectrum(xaxis=xaxis, flux=np.ones(len(xaxis)))

    assert spec_max_delta(spec, rv, gamma) == max_delta(xaxis, rv, gamma)


@pytest.mark.parametrize("wav, rv, gamma, expected", [
    ([1], 3, 5, 5 / c),
    ([2], 1, -7.1, 2 * 7.1 / c),
    (2, 1, 7.1, 2 * 7.1 / c),
    ([1, 2], 0, 0, 0)
])
def test_spec_max_delta(wav, rv, gamma, expected):
    assert 2 * round(expected, 3) == max_delta(wav, rv, gamma)


@pytest.mark.parametrize("rv, gamma", [
    (np.array([1, 2, 3, 4]), np.array([])),
    (np.array([]), np.array([1, 2, 3, 4])),
    ([], np.array([1, 2, 3, 4])),
    (np.array([1, 2, 3, 4]), [])])
def test_max_delta_with_empty_arrays(rv, gamma):
    wav = np.arange(20)
    with pytest.raises(ValueError) as excinfo:
        max_delta(wav, rv, gamma)

    assert 'Empty variable vector' in str(excinfo.value)


@pytest.mark.parametrize("inputs,expected", [
    (range(5), np.array([0, 1, 2, 3, 4])),
    ("None", np.ndarray([0])),
    (None, np.ndarray([0])),
    ([0], np.array([0])),
    (1, np.array([1])),
    (0, np.array([0]))
])
def test_check_inputs(inputs, expected):
    assert np.allclose(check_inputs(inputs), expected)


@pytest.mark.parametrize("inputs", [[], np.array([]), {}, ()])
def test_check_inputs_raises_empty_error(inputs):
    with pytest.raises(ValueError) as excinfo:
        check_inputs(inputs)
    assert "Empty variable" in str(excinfo.value)
