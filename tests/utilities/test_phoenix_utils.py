"""Testing phoenix utilities."""

from types import GeneratorType

import numpy as np
import pytest
from spectrum_overload import Spectrum

import simulators
from mingle.utilities.phoenix_utils import closest_model_params
from mingle.utilities.phoenix_utils import (gen_new_param_values,
                                            generate_close_params_with_simulator,
                                            load_phoenix_spectrum,
                                            load_starfish_spectrum, phoenix_area)


@pytest.mark.parametrize("limits, normalize", [([2100, 2150], True), ([2050, 2150], False)])
def test_load_phoenix_spectra(limits, normalize):
    test_spectrum = "tests/testdata/lte02300-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    spec = load_phoenix_spectrum(test_spectrum, limits=limits, normalize=normalize)

    assert isinstance(spec, Spectrum)

    if normalize:
        assert np.all(spec.flux < 2)
    else:
        assert np.all(spec.flux > 1e10)

    assert np.all(limits[0] < spec.xaxis)
    assert np.all(spec.xaxis < limits[-1])


@pytest.mark.parametrize("teff, limits, normalize",
                         [(2300, [2100, 2150], True), (2300, [2100, 2150], False), (5000, [2050, 2150], False)])
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
    test_spectrum = "tests/testdata/lte02300-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    spec1 = load_phoenix_spectrum(test_spectrum, limits=None)
    spec2 = load_starfish_spectrum([2300, 5, 0], limits=None)
    assert len(spec1) > len(spec2)  # Spec1 is full spectrum
    assert isinstance(spec1, Spectrum)
    assert isinstance(spec2, Spectrum)


@pytest.mark.xfail()  # Starfish does resampling
@pytest.mark.parametrize("limits", [[2090, 2135], [2450, 2570]])
def test_phoenix_and_starfish_load_equally_with_limits(limits):
    test_spectrum = "tests/testdata/lte02300-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
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
    test_header = {"PHXREFF": 1e11}  # scales effective radius down by 1e-11
    assert np.allclose(phoenix_area(test_header), np.pi)

    with pytest.raises(ValueError):
        phoenix_area(None)

    with pytest.raises(KeyError):
        phoenix_area({"Not_PHXREFF": 42})


def test_gen_new_param_values():
    # temp, logg, metal
    x, y, z = gen_new_param_values(100, 1, 0, small=False)
    print("X=", x)
    assert np.all(x == [-400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 600])
    assert np.all(y == [0, 0.5, 1, 1.5, 2])
    assert np.all(z == [-1, -0.5, 0, 0.5, 1])


def test_gen_new_param_values_with_host():
    # temp, logg, metal
    x, y, z = gen_new_param_values(5000, 2, 1, small="host")

    assert np.all(x == np.array([4900, 5000, 5100]))
    assert np.all(y == np.array([1.5, 2, 2.5]))
    assert np.all(z == np.array([0.5, 1, 1.5]))


def test_generate_close_params_with_simulator_single_return():
    start_params = [5000, 4.5, 0]
    # start, stop, step of [0, 1, 1]  -> [0] while [0, 0, 1] -> []
    test_dict = {"teff_1": [0, 1, 1], "feh_1": [0, 1, 1], "logg_1": [0, 1, 1],
                 "teff_2": [0, 1, 1], "feh_2": [0, 1, 1], "logg_2": [0, 1, 1]}
    expected = [[5000, 4.5, 0]]
    simulators.sim_grid = test_dict
    host_params = generate_close_params_with_simulator(start_params, "host")
    assert isinstance(host_params, GeneratorType)
    host_params = list(host_params)
    print("host_params", host_params)
    comp_params = generate_close_params_with_simulator(start_params, "companion")
    assert isinstance(comp_params, GeneratorType)
    comp_params = list(comp_params)

    assert host_params == expected
    assert comp_params == expected


def test_load_starfish_with_header():
    params = [2300, 5, 0]
    limits = [2100, 2200]
    spec_no_header = load_starfish_spectrum(params, hdr=False, limits=limits)
    assert spec_no_header.header == {}

    # Test some values from phoenix header
    spec_with_header = load_starfish_spectrum(params, hdr=True, limits=limits)
    print(spec_with_header.header)
    assert spec_with_header.header["air"] == False
    assert spec_with_header.header["PHXEOS"] == "ACES"


def test_load_starfish_header_with_area_scale():
    spec_no_scale = load_starfish_spectrum([2300, 5, 0], hdr=True, area_scale=False)
    assert spec_no_scale.header.get("emit_area") is None  # Change spec header when scaling
    spec = load_starfish_spectrum([2300, 5, 0], hdr=True, area_scale=True)
    print("header", spec.header)

    assert spec.header["emit_area"] is not None  # Change spec header when scaling


def test_load_starfish_no_header_with_area_scale():
    with pytest.raises(ValueError):
        load_starfish_spectrum([2300, 5, 0], hdr=False, area_scale=True)


def test_load_starfish_flux_rescale():
    params = [2300, 5, 0]
    limits = [2100, 2200]
    spec = load_starfish_spectrum(params, hdr=False, limits=limits, flux_rescale=False)
    spec_rescaled = load_starfish_spectrum(params, hdr=False, limits=limits, flux_rescale=True)
    rescale_value = 1e7
    spec /= rescale_value
    assert np.allclose(spec.xaxis, spec_rescaled.xaxis)
    assert np.allclose(spec.flux, spec_rescaled.flux)


@pytest.mark.parametrize("input, expected", [
    ([2535, 4.53, -0.21], [2500, 4.5, 0.0]),
    ([7887, 2.91, -1.21], [7900, 3.0, -1.0]),
    ([10187, 0.51, -2.76], [7900, 0.5, -3.0])
])
def closest_model_params(input, expected):
    assert closest_model_params(*input) == expected



from mingle.utilities.phoenix_utils import phoenix_name, phoenix_regex
def test_phoenix_name():

    print(phoenix_name(2000, 2.5, 0.5))
    print(os.path.join("Z+0.5", ("lte02000-2.50+0.5."
                        "PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")))
    assert phoenix_name(2000, 2.5, 0.5, Z=True) == os.path.join("Z+0.5", ("lte02000-2.50+0.5."
                        "PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"))


def test_find_phoenix_model_names():
    base_dir = os.path.join("tests", "testdata")
    original_model = "lte02500-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    found = find_phoenix_model_names(base_dir, original_model)
    print("found models", found)
    assert "lte02300-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits" in found[0]
    assert len(found) == 1  # because only have one file suitable file in testdata atm.


assert phoenix_name(5700, 6.0, -1.5, Z=False) == ("lte05700-6.00-1.5."
                        "PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")

def test_phoenix_name():
    tail = "PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    assert phoenix_name(2000, 2.5, 0.5, Z=True) == os.path.join("Z+0.5",
                                                                "lte02000-2.50+0.5.{}".format(tail))

    assert phoenix_name(5700, 6.0, -1.5, Z=False) == "lte05700-6.00-1.5.{}".format(tail)


def test_phoenix_name_alpha_notimplemented():
    with pytest.raises(NotImplementedError):
        phoenix_name(5000, 1.5, 0.0, alpha=0.2)


def test_phoenix_regex():
    assert phoenix_regex(2000, 2.5, 0.5, Z=True) == os.path.join("Z+0.5", "*02000-2.50+0.5.PHOENIX*.fits")

    assert phoenix_regex(12000, 3, 0, Z=False) == "*12000-3.00-0.0.PHOENIX*.fits"
