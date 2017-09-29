"""broadcast_test.py."""
from __future__ import division, print_function

import numpy as np
import pytest
import scipy as sp

# Test that the two componet model with alpha = [0] and rvs=[0] are equal!
from models.broadcasted_models import (check_broadcastable,
                                       inherent_alpha_model, one_comp_model,
                                       two_comp_model,
                                       two_comp_model_with_transpose)
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


def test_models_are_same_with_no_companion(host):
    """To compare models give equvalient ouptut.

    If alpha= 0 and rvs = 0.
    """
    ocm = one_comp_model(host.xaxis, host.flux, [1, 2, 3])
    ocm_eval = ocm(host.xaxis)
    tcm = two_comp_model(host.xaxis, host.flux, np.ones_like(host.flux), 0, [0], [1, 2, 3])
    tcm_eval = tcm(host.xaxis).squeeze()

    assert ocm_eval.shape == tcm_eval.shape
    ocm_eval[np.isnan(ocm_eval)] = 0
    tcm_eval[np.isnan(tcm_eval)] = 0

    assert np.allclose(ocm_eval, tcm_eval)


@pytest.mark.parametrize("alpha,equal", [
    (0.1, False),
    (0, True)
    ])
def test_no_tcm_companion(host, alpha, equal):
    """To compare models give equvalient ouptut.

    If alpha= 0 and rvs = 0.
    s"""

    tcm = two_comp_model(host.xaxis, host.flux, np.ones_like(host.flux), alpha, 0, [1, 2, 3])
    tcm2 = two_comp_model(host.xaxis, host.flux, np.zeros_like(host.flux), alpha, 0, [1, 2, 3])
    tcm_eval = tcm(host.xaxis).squeeze()
    tcm2_eval = tcm2(host.xaxis).squeeze()

    assert tcm_eval.shape == tcm2_eval.shape

    tcm_eval[np.isnan(tcm_eval)] = 0
    tcm2_eval[np.isnan(tcm2_eval)] = 0
    assert np.allclose(tcm_eval, tcm2_eval) is equal


@pytest.mark.xfail()
def test_tcm_with_transpose(host, comp):
    """To compare models give equvalient ouptut.

    If alpha= 0 and rvs = 0.
    s"""
    tcm = two_comp_model(host.xaxis, host.flux, comp.flux, 0, [0], [1, 2, 3])
    tcm_T = two_comp_model_with_transpose(host.xaxis, host.flux, comp.flux, 0, [0], [1, 2, 3])

    tcm_eval = tcm(host.xaxis)
    tcm_T_eval = tcm_T(host.xaxis)

    assert tcm_eval.shape == tcm_T_eval.shape
    assert np.allclose(tcm_eval, tcm_T_eval)


def test_broadcasting_with_transpose():
    """Test transpose method for calculations."""
    # Doesn't check actual codes
    small = np.random.rand(1, 2)
    large = np.random.rand(1, 2, 4, 5, 2)

    assert ((small.T * large.T).T == small[:, :, None, None, None] * large).all()
    assert ((large.T * small.T).T == large * small[:, :, None, None, None]).all()
    assert ((large.T * small.T).T == small[:, :, None, None, None] * large).all()


def test_shape_of_tcm(host, comp):
    gammas = np.arange(2)
    rvs = np.arange(3)
    alphas = np.arange(4) / 16

    tcm = two_comp_model(host.xaxis, host.flux, comp.flux, alphas, rvs, gammas)
    assert isinstance(tcm, sp.interpolate.interp1d)

    tcm_eval = tcm(host.xaxis)  # Evaluate at host.xaxis
    assert tcm_eval.shape == (len(host.xaxis), len(alphas), len(rvs), len(gammas))


def test_shape_of_ocm(host):
    gammas = np.arange(2)

    ocm = one_comp_model(host.xaxis, host.flux, gammas)
    assert isinstance(ocm, sp.interpolate.interp1d)

    ocm_eval = ocm(host.xaxis)  # Evaluate at host.xaxis
    assert ocm_eval.shape == (len(host.xaxis), len(gammas))


def test_check_broadcastable():
    # turn scalar or list into 2d array with 1s on the right
    assert check_broadcastable(2).shape == (1, 1)
    assert check_broadcastable([2]).shape == (1, 1)
    assert check_broadcastable([[[2]]]).shape == (1, 1, 1)
    assert check_broadcastable([1, 2]).shape == (2, 1)
    assert check_broadcastable([[1], [2]]).shape == (2, 1)


def test_inherinent_model_same_as_alpha_0(host, comp):

    tcm = two_comp_model(host.xaxis, host.flux, comp.flux, 0, [0, 2, 4], [1, 2, 3])
    iam = inherent_alpha_model(host.xaxis, host.flux, comp.flux, [0, 2, 4], [1, 2, 3])

    host.wav_select(2100.5, 2104.5)   # cut to avoid Nans from doppler shifts
    tcm_value = tcm(host.xaxis)
    iam_value = iam(host.xaxis)
    # print(tcm_value)
    # print(iam_value)
    assert tcm_value.squeeze().shape == iam_value.shape
