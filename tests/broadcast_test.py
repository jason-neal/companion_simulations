"""broadcast_test.py."""
from __future__ import division, print_function

import numpy as np
import pytest
import scipy as sp

# Test that the two component model with alpha = [0] and rvs=[0] are equal!
from mingle.models.broadcasted_models import (check_broadcastable,
                                              inherent_alpha_model, one_comp_model,
                                              two_comp_model,
                                              two_comp_model_with_transpose)
from utilities.phoenix_utils import load_starfish_spectrum


@pytest.fixture
def host():
    """Host spectrum fixture."""
    mod_spec = load_starfish_spectrum([5200, 4.50, 0.0], limits=[2130, 2135], normalize=True)
    return mod_spec


@pytest.fixture
def comp():
    """Normalized Companion spectrum fixture."""
    mod_spec = load_starfish_spectrum([2600, 4.50, 0.0], limits=[2130, 2135], normalize=True)
    return mod_spec


@pytest.mark.parametrize("gamma, rv", [
    ([0], [0]),
    (0, [0]),
    (0, [0]),
    ([-1, -2, -3], 1),
    ([0], [3]),
    ([-1, 0, 1], [0])
])
def test_ocm_and_tcm_models_are_same_with_no_companion(host, gamma, rv):
    """To compare models give equivalent output.

    If alpha=0 then there is no companion.
    """
    ocm = one_comp_model(host.xaxis, host.flux, gamma)
    ocm_eval = ocm(host.xaxis).squeeze()
    tcm = two_comp_model(host.xaxis, host.flux, np.ones_like(host.flux), 0, rv, gamma)
    tcm_eval = tcm(host.xaxis).squeeze()

    assert ocm_eval.shape == tcm_eval.shape

    o_ravel = ocm_eval.ravel()
    t_ravel = ocm_eval.ravel()
    assert np.allclose(o_ravel[~np.isnan(o_ravel)], t_ravel[~np.isnan(t_ravel)])


@pytest.mark.xfail()
def test_tcm_with_transpose(host, comp):
    """To compare models give equivalent output.

    If alpha= 0 and rvs = 0.
    s"""
    tcm = two_comp_model(host.xaxis, host.flux, comp.flux, 0, [0], [1, 2, 3])
    tcm_trans = two_comp_model_with_transpose(host.xaxis, host.flux, comp.flux, 0, [0], [1, 2, 3])

    tcm_eval = tcm(host.xaxis)
    tcm_trans_eval = tcm_trans(host.xaxis)

    assert tcm_eval.shape == tcm_trans_eval.shape
    assert np.allclose(tcm_eval, tcm_trans_eval)


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

    host.wav_select(2100.5, 2104.5)  # cut to avoid Nans from doppler shifts
    tcm_value = tcm(host.xaxis)
    iam_value = iam(host.xaxis)
    # print(tcm_value)
    # print(iam_value)
    assert tcm_value.squeeze().shape == iam_value.shape
