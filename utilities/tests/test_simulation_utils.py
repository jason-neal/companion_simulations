import numpy as np
import pytest

from utilities.simulation_utilities import check_inputs


@pytest.mark.parametrize("input,expected", [
    (None, np.ndarray([0])),
    ([0], np.array([0])),
    (1, np.array([1])),
    (range(5), np.array([0,1,2,3,4]))
])
def test_check_inputs(input, expected):
    assert np.allclose(check_inputs(input), expected)
