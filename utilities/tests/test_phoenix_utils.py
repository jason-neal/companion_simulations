"""Testing phoenix utilities."""

import numpy as np
import pytest

import simulators
from spectrum_overload.Spectrum import Spectrum
# from utilities.crires_utilities import crires_resolution
from utilities.phoenix_utils import (gen_new_param_values,
                                     generate_close_params_with_simulator,
                                     load_phoenix_spectrum,
                                     load_starfish_spectrum, phoenix_area)

print(simulators.paths)
print(simualtors)

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
    print("simulators paths in without limits",simulators.paths)
    test_spectrum = "utilities/tests/test_data/lte02300-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    spec1 = load_phoenix_spectrum(test_spectrum, limits=None)
    spec2 = load_starfish_spectrum([2300, 5, 0], limits=None)
    assert len(spec1) > len(spec2)   # Spec1 is full spectrum
    assert isinstance(spec1, Spectrum)
    assert isinstance(spec2, Spectrum)


@pytest.mark.xfail()   # Starfish does resampling
@pytest.mark.parametrize("limits", [[2090, 2135], [2450, 2570]])
def test_phoenix_and_starfish_load_equally_with_limits(limits):
    print("simulators paths in 'with limits' ",simulators.paths)
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


def gen_new_param_values():
    x, y, z = gen_new_param_values(100, 1, 0, small=False)

    assert x == [-400, 300, 200, 100, 0, 100, 200, 300, 400, 500, 600]
    assert y == [-0, 0.5, 1, 1.5, 2]
    assert z == [-1, -0.5, 0, 0.5, 1]


def gen_new_param_values_with_host():
    x, y, z = gen_new_param_values(5000, 2, 1, small="host")

    assert np.all(x == np.array([4900, 5000, 5100]))
    assert np.all(y == np.array([1.5, 2, 2.5]))
    assert np.all(z == np.array([0.5, 1, 1.5]))


# def test_generate_close_params():
#    assert False


def generate_close_params_with_simulator_single_return():
    start_params = [5000, 4.5, 0]
    test_dict = {"teff_1": [0, 0, 1], "feh_1": [0, 0, 1], "logg_1": [0, 0, 1],
                 "teff_2": [0, 0, 1], "feh_2": [0, 0, 1], "logg_2": [0, 0, 1]}
    expected = [[5000, 4.5, 0]]
    simulators.sim_grid = test_dict
    host_params = generate_close_params_with_simulator(start_params, "host")
    comp_params = generate_close_params_with_simulator(start_params, "companion")
    assert list(host_params) == expected
    assert list(comp_params) == expected
    assert False
