import pytest
from spectrum_overload import Spectrum

from mingle.utilities.spectrum_utils import load_spectrum, select_observation


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

@pytest.mark.parametrize("chip", [0, None, 5, 42])
def test_select_observation_with_bad_chip(chip):
    with pytest.raises(ValueError):
        select_observation("HD30501", "1", chip)

@pytest.mark.xfail
def test_spectrum_plotter(spectra, label=None, show=False):
    """Plot a Spectrum object."""
    assert False


@pytest.mark.xfail
def test_plot_spectra(obs, model):
    """Plot two spectra."""
    assert False

