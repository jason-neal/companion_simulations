import pytest
from spectrum_overload import Spectrum

from mingle.utilities.spectrum_utils import load_spectrum


@pytest.mark.xfail
def test_load_spectrum():
    load_spectrum()
    assert False
    assert isinstance(results, Spectrum)


@pytest.mark.xfail
def test_load_spectrum_with_failure():
    assert False


@pytest.mark.xfail
def test_select_observation():
    assert False


@pytest.mark.xfail
def test_spectrum_plotter(spectra, label=None, show=False):
    """Plot a Spectrum object."""
    assert False


@pytest.mark.xfail
def test_plot_spectra(obs, model):
    """Plot two spectra."""
    assert False

