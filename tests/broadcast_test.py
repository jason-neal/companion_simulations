"""broadcast_test.py."""
import numpy as np
import pytest

# Test that the two componet model with alpha = [0] and rvs=[0] are equal!
from models.broadcasted_models import one_comp_model, two_comp_model
from spectrum_overload.Spectrum import Spectrum


@pytest.fixture
def host():
    """Host spectrum fixture."""
    return Spectrum()
    


@pytest.fixture
def comp():
    """Companion spectrum fixture."""
    return Spectrum()



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
