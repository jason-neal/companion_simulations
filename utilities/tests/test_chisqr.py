import numpy as np
import pytest
from scipy.stats import chisquare

from utilities.chisqr import chi_squared, reduced_chi_squared, spectrum_chisqr


@pytest.mark.parametrize("observed, expected", [
    ([1.1, 2.2, 3.1, 3.9, 4.8], [1, 2, 3, 4, 5]),
    (5, 6)])
def test_chi_squared_without_errors_equals_scipy(observed, expected):
    observed = np.asarray(observed)
    expected = np.asarray(expected)
    assert chi_squared(observed, expected) == chisquare(observed, expected).statistic


@pytest.mark.parametrize("observed, expected, error", [
    ([1.1, 2.2, 3.1, 3.9, 4.8], [1, 2, 3, 4, 5], [0.1, 0.1, 0.2]),
    ([5, 3], [6, 2], [0.2, 0.5, 0.6])])
def test_chi_squared_with_error_unequal_length(observed, expected, error):
    observed = np.asarray(observed)
    expected = np.asarray(expected)
    error = np.asarray(error)
    with pytest.raises(ValueError):
        chi_squared(observed, expected, error=error)


@pytest.mark.parametrize("observed", [
    ([1.1, 2.2, 3.1, 3.9, 4.8]),
    5])
def test_model_equal_observed_chi_squared_returns_zero(observed):
    observed = np.asarray(observed)
    assert chi_squared(observed, observed) == 0


@pytest.mark.parametrize("chi2, n, p, expected", [
    (7.1, 8, 1, 7.1 / 7),
    (19, 25, 3, 19 / 22),
    ([2, 3, 5], 3, 1, [1, 3 / 2, 5 / 2])])
def test_reduced_chi_squared_without_erros_equals_scipy(chi2, n, p, expected):
    if isinstance(chi2, list):
        chi2 = np.asarray(chi2)
        expected = np.asarray(expected)
    assert np.all(reduced_chi_squared(chi2, n, p) == expected)


@pytest.mark.xfail()
def test_spectrum_chi_squared(host):
    pass
    assert False


def test_spectrum_chi_squared_model_equal_observed_gives_zero(host):
    assert spectrum_chisqr(host, host) == 0


def test_spectrum_chi_squared_with_unequal_axis_throws_exception(host):
    """Assert won't work when xaxis different lenghts."""
    host2 = host.copy()
    host2.wav_select(host2.xaxis[50], host2.xaxis[-50])

    with pytest.raises(Exception):
        spectrum_chisqr(host, host2)
