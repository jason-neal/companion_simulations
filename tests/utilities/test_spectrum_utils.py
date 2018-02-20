import os

import numpy as np
import pytest
from spectrum_overload import Spectrum

from mingle.utilities.spectrum_utils import load_spectrum, select_observation


@pytest.mark.parametrize("fname", ["HD30501-1-mixavg-tellcorr_1.fits", "HD30501-1-mixavg-h2otellcorr_1.fits"])
def test_load_spectrum(fname):
    fname = os.path.join("tests", "testdata", "handy_spectra", fname)
    results = load_spectrum(fname)
    assert isinstance(results, Spectrum)
    assert results.header["OBJECT"].upper() == "HD30501"
    assert np.all(results.xaxis > 2110)  # nm
    assert np.all(results.xaxis < 2130)  # nm
    assert np.all(results.flux < 2)
    assert np.all(results.flux >= 0)


def test_load_no_filename_fits():
    """Not a valid file."""
    with pytest.raises(ValueError):
        load_spectrum("")


@pytest.mark.parametrize("chip", [0, None, 5, 42])
def test_select_observation_with_bad_chip(chip):
    with pytest.raises(ValueError):
        select_observation("HD30501", "1", chip)


@pytest.mark.xfail()
def test_spectrum_plotter(spectra, label=None, show=False):
    """Plot a Spectrum object."""
    assert False


@pytest.mark.xfail()
def test_plot_spectra(obs, model):
    """Plot two spectra."""
    assert False
