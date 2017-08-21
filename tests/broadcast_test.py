"""broadcast_test.py."""
import numpy as np
import pytest

# Test that the two componet model with alpha = [0] and rvs=[0] are equal!
from models.broadcasted_models import one_comp_model, two_comp_model
from utilities.phoenix_utils import load_starfish_spectrum
from models.broadcasted_models import check_broadcastable

@pytest.fixture
def host():
    """Host spectrum fixture."""
    mod_spec = load_starfish_spectrum([5600, 5, -0.5], limits=[2100, 2150], hdr=True, normalize=True)

    return mod_spec



@pytest.fixture
def comp():
    """Companion spectrum fixture."""
    mod_spec = load_starfish_spectrum([2600, 4.5, 0.0], limits=[2100, 2150], hdr=True, normalize=True)

    return mod_spec



def test_models_are_same_with_no_companion(host, comp):
    """To compare models give equvalient ouptut.

    If alpha= 0 and rvs = 0.
    s"""
    ocm = one_comp_model(host.xaxis, host.flux, [1, 2, 3])
    tcm = two_comp_model(host.xaxis, host.flux, comp.flux, [0], [0], [1, 2, 3])

    assert np.allclose(ocm, tcm)


def test_broadcasting_with_transpose():
    """Test transpose method for calculations."""
    # Doesn't check actual codes
    small = np.random.rand(1, 2)
    large = np.random.rand(1, 2, 4, 5, 2)

    assert ((small.T * large.T).T == small[:, :, None, None, None] * large).all()
    assert ((large.T * small.T).T == large * small[:, :, None, None, None]).all()
    assert ((large.T * small.T).T == small[:, :, None, None, None] * large).all()
def test_check_broadcastable():
    # turn scalar or list into 2d array with 1s on the right
    assert check_broadcastable(2).shape == (1, 1)
    assert check_broadcastable([2]).shape == (1, 1)
    assert check_broadcastable([[[2]]]).shape == (1, 1, 1)
    assert check_broadcastable([1, 2]).shape == (2, 1)
    assert check_broadcastable([[1], [2]]).shape == (2, 1)
