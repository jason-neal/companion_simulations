"""broadcast_test.py"""

import numpy as np
import pytest

# Test that the two componet model with alpha = [0] and rvs=[0] are equal!
from models.broadcasted_models import one_comp_model, two_comp_model
from spectrum_overload.Spectrum import Spectrum


@pytest.fixture
def host():
    pass


@pytest.fixture
def comp():
    pass


def test_models_are_same_with_no_companion(host, comp):

    ocm = one_comp_model(host.xaxis, host.flux, [1, 2, 3])
    tcm = two_comp_model(host.xaxis, host.flux, comp.flux, [0], [0], [1, 2, 3])

    assert np.allclose(ocm, tcm)


def test_broadcasting_with_transpose():
    # Doesn't check actual codes
    a = np.random.rand(1, 2)
    b = np.random.rand(1, 2, 4, 5, 2)

    assert ((a.T * b.T).T == a[:, :, None, None, None] * b).all()
