"""Testing phoenix utilities."""

import numpy as np
import pytest

from spectrum_overload.Spectrum import Spectrum
# from utilities.crires_utilities import crires_resolution
from utilities.phoenix_utils import (load_phoenix_spectrum,
                                     load_starfish_spectrum, phoenix_area)


@pytest.mark.parametrize("limits, normalize", [([2100, 2150], True), ([2050, 2150], False)])
def test_load_phoenix_spectra(limits, normalize):
    test_spectrum = "utilities/tests/test_data/lte02300-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    spec = load_phoenix_spectrum(test_spectrum, limits=limits, normalize=normalize)

    assert isinstance(spec, Spectrum)

    if normalize:
        assert np.all(spec.flux < 2)
    else:
        assert np.all(spec.flux > 1e10)

    assert np.all(limits[0] < spec.xaxis)
    assert np.all(spec.xaxis < limits[-1])


@pytest.mark.parametrize("teff,limits,normalize", [(2300, [2100, 2150], True), (2300, [2100, 2150], False), (5000, [2050, 2150], False)])
def test_load_starfish_spectra(teff, limits, normalize):
    spec = load_starfish_spectrum([teff, 5, 0], limits=limits, normalize=normalize)

    assert isinstance(spec, Spectrum)
    if normalize:
        assert np.all(spec.flux < 2)
    else:
        assert np.all(spec.flux > 1e10)

    assert np.all(limits[0] < spec.xaxis)
    assert np.all(spec.xaxis < limits[-1])


def test_phoenix_and_starfish_load_differently_without_limits():
    test_spectrum = "utilities/tests/test_data/lte02300-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    spec1 = load_phoenix_spectrum(test_spectrum, limits=None)
    spec2 = load_starfish_spectrum([2300, 5, 0], limits=None)
    assert len(spec1) > len(spec2)   # Spec1 is full spectrum
    assert isinstance(spec1, Spectrum)
    assert isinstance(spec2, Spectrum)

@pytest.mark.xfail()   # Starfish does resampling
@pytest.mark.parametrize("limits", [[2090, 2135], [2450, 2570]])
def test_phoenix_and_starfish_load_equally_with_limits(limits):
    test_spectrum = "utilities/tests/test_data/lte02300-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    spec1 = load_phoenix_spectrum(test_spectrum, limits=limits)
    spec2 = load_starfish_spectrum([2300, 5, 0], limits=limits)
    assert spec1.xaxis[0] == spec2.xaxis[0]
    assert spec1.xaxis[-1] == spec2.xaxis[-1]

    assert np.allclose(spec1.flux, spec2.flux)
    assert np.allclose(spec1.xaxis, spec2.xaxis)
    assert spec1.header == spec2.header
    assert isinstance(spec1, Spectrum)
    assert isinstance(spec2, Spectrum)


def test_phoenix_area():
    test_header = {"PHXREFF": 1e11}   # scales effective radius down by 1e-11
    assert np.allclose(phoenix_area(test_header), np.pi)

    with pytest.raises(ValueError):
        phoenix_area(None)

    with pytest.raises(KeyError):
        phoenix_area({"Not_PHXREFF": 42})
