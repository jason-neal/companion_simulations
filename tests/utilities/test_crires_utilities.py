import numpy as np
import pytest
from astropy.constants import c
from spectrum_overload import Spectrum

from mingle.utilities.crires_utilities import barycorr_crires, barycorr_crires_spectrum


# Test with and without offsets
# Test with invalid headers

@pytest.mark.xfail
def test_barycorr_crires_spectrum(host):
    barycorr_crires_spectrum(host, extra_offset=None)
    assert False


@pytest.mark.xfail
def test_barycorr_crires_spectrum_with_obs_spec(host):
    new_spec = barycorr_crires_spectrum(host, extra_offset=None)
    assert False


@pytest.mark.xfail
def test_barycorr_crires(host):
    barycorr_crires(host.xaxis, host.flux, host.header, extra_offset=0)
    assert False


@pytest.mark.parametrize("extra", [None, 0.0])
def test_barycorr_crires_spectrum_with_invalid_header_no_offset_returns_same(host, extra):
    new_spec = barycorr_crires_spectrum(host, extra_offset=extra)
    assert np.allclose(new_spec.xaxis, host.xaxis)

    assert np.allclose(new_spec.flux, host.flux)

@pytest.mark.xfail
@pytest.mark.parametrize("extra", [None, 0.0, -10., 50.])
def test_barycorr_crires_is_implemented_on_spectrum(host, extra):
    host_copy = host.copy()
    new_spec = barycorr_crires_spectrum(host, extra_offset=extra)
    wlprime, new_flux = barycorr_crires(host.xaxis, host.flux, host.header, extra_offset=extra)
    extra = 0.0 if extra is None else extra

    assert isinstance(new_spec, Spectrum)
    assert np.allclose(new_spec.xaxis, host.xaxis)
    assert np.allclose(new_spec.flux[~np.isnan(new_spec.flux)], new_flux[~np.isnan(new_flux)])
    assert np.allclose(new_flux[~np.isnan(new_flux)], np.interp(host.xaxis, wlprime, host.flux)[~np.isnan(new_flux)])
    if extra is not None:
        manual_shift = host_copy.xaxis * (1 + extra / c.value)
        assert np.allclose(wlprime, manual_shift)
    assert np.allclose(new_spec.flux, new_flux)


@pytest.mark.parametrize("extra", [None, 0.0])
def test_barycorr_crires_on_None_or_zero_returns_unchanged(host, extra):
    wlprime, new_flux = barycorr_crires(host.xaxis, host.flux, {}, extra_offset=extra)

    assert np.allclose(wlprime, host.xaxis)
    assert np.allclose(new_flux, host.flux)


@pytest.mark.parametrize("extra", [-20, -5, 0, 6, 12, 35])
def test_barycorr_extra_offset_is_reversable(host, extra):
    wav, flux = barycorr_crires(host.xaxis, host.flux, {}, extra_offset=extra)
    wav2, flux2 = barycorr_crires(wav, flux, {}, extra_offset=-extra)

    assert np.allclose(host.xaxis, wav2)
    assert np.allclose(host.flux[~np.isnan(flux2)], flux2[~np.isnan(flux2)], rtol=4)

# # TODO: Add reversible option to add the berv?
# @pytest.mark.parametrize("extra", [-20, -5, 0, 6, 12, 35])
# def test_barycorr_is_reversible(host, extra):
#     wavprime, flux = barycorr_crires(host.xaxis, host.flux, {}, extra_offset=extra)
#     __, flux2 = barycorr_crires(host.xaxis, flux, {}, extra_offset=extra, reverse=True)
#     wav2prime, __ = barycorr_crires(wavprime, flux, {}, extra_offset=extra, reverse=True)
#
#     assert np.allclose(host.xaxis, wav2prime)
#    assert np.allclose(host.flux[~np.isnan(flux2)], flux2[~np.isnan(flux2)], rtol=4)
