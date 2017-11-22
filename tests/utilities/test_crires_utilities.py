import pytest

from mingle.utilities.crires_utilities import barycorr_crires, barycorr_crires_spectrum


def test_barycorr_crires_spectrum(host):
    barycorr_crires_spectrum(host, )
    assert False

@pytest.mark.parametrize("extra", [None, 0.0])
def test_barycorr_crires_spectrum_with_invalid_header_no_offset_returns_same(host, extra):
    new_spec = barycorr_crires_spectrum(host, extra_offset=extra)
    assert new_spec.xaxis == host.xaxis
    assert new_spec.flux == host.flux


# Test with and without offsets
# Test with invalid headers

@pytest.mark.parametrize("extra", [None, 0.0, -10, 50])
def test_barycorr_crires_is_implemented_on_spectrum(host, extra):
    new_spec = barycorr_crires_spectrum(host, extra_offset=extra)
    new_xaxis, new_flux = barycorr_crires(host.xaxis, host.flux, host.header, extra_offset=extra)
    assert np.allclose(new_spec.xaxis, new_xaxis)
    assert np.allclose(new_spec.flux, new_flux)


# Test with and without offsets


def test_barycorr_crires(host):
    barycorr_crires(barycorr_crires(host.xaxis, host.flux, host.header, extra_offset=0))
    assert False
