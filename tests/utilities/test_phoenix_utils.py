"""Testing phoenix utilities."""

import itertools
import os
from types import GeneratorType

import numpy as np
import pytest
from spectrum_overload import Spectrum

from mingle.utilities.phoenix_utils import closest_model_params
from mingle.utilities.phoenix_utils import (gen_new_param_values,
                                            generate_close_params_with_simulator,
                                            load_phoenix_spectrum,
                                            load_starfish_spectrum, phoenix_area)
from mingle.utilities.phoenix_utils import get_phoenix_limits
from mingle.utilities.phoenix_utils import phoenix_name, phoenix_regex, \
    find_closest_phoenix_name, find_phoenix_model_names
from mingle.utilities.phoenix_utils import set_model_limits


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


@pytest.mark.xfail(strict=True)  # Starfish does resampling
@pytest.mark.parametrize("limits", [[2090, 2135], [2450, 2570]])
def test_phoenix_and_starfish_load_equally_with_limits(limits):
    test_spectrum = "tests/testdata/lte02300-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    spec1 = load_phoenix_spectrum(test_spectrum, limits=limits)
    spec2 = load_starfish_spectrum([2300, 5, 0], limits=limits)

    spec2 = spec2.interpolate1d_to(spec1)  # resamle to same axis
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


def test_generate_close_params_with_simulator_single_return(sim_config):
    simulators = sim_config
    start_params = [5000, 4.5, 0]
    # start, stop, step of [0, 1, 1]  -> [0] while [0, 0, 1] -> []
    test_dict = {"teff_1": [0, 1, 1], "feh_1": [0, 1, 1], "logg_1": [0, 1, 1],
                 "teff_2": [0, 1, 1], "feh_2": [0, 1, 1], "logg_2": [0, 1, 1]}
    expected = [[5000, 4.5, 0]]
    simulators.sim_grid = test_dict
    host_params = generate_close_params_with_simulator(start_params, "host")
    assert isinstance(host_params, GeneratorType)
    host_params = list(host_params)
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


def test_find_closest_phoenix_name():
    data_dir = os.path.join("tests", "testdata")
    name = find_closest_phoenix_name(data_dir, 2305, 4.89, 0.01, alpha=None)
    assert name == []

    # z dir False
    name = find_closest_phoenix_name(data_dir, 2305, 4.89, 0.01, alpha=None, Z=False)
    assert "lte02300-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits" in name[0]


def test_find_phoenix_model_names():
    base_dir = os.path.join("tests", "testdata")
    original_model = "lte02500-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    found = find_phoenix_model_names(base_dir, original_model)
    print("found models", found)
    assert "lte02300-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits" in found[0]
    assert len(found) == 1  # because only have one file suitable file in testdata atm.


@pytest.mark.parametrize("target", ["not_host", "", 7])
def test_gen_close_params_with_simulator_invalid_target(target):
    with pytest.raises(ValueError):
        generator = generate_close_params_with_simulator([5000, 4.5, 0.5], target=target)
        generator.__next__()  # Need to act on it to make code work (and fail)


@pytest.mark.parametrize("teff, logg, feh, expected_num", [
    ([-100, 101, 100], [-0.5, 0.51, 0.5], [-0.5, 0.51, 0.5], 27),
    ([0, 100, 100], [0, 1, 1], [0, 1, 1], 1),
    ([-500, 501, 100], [0, 1.01, 0.5], [0, 1, 1], 33)])
def test_gen_close_params_with_simulator_gets_comp_set_to_sims(sim_config, teff, logg, feh, expected_num):
    simulators = sim_config
    simulators.sim_grid["teff_2"] = teff
    simulators.sim_grid["logg_2"] = logg
    simulators.sim_grid["feh_2"] = feh

    result = generate_close_params_with_simulator([5000, 4.5, 0.5], target="companion")
    assert isinstance(result, GeneratorType)
    result = list(result)
    print(result)
    assert len(list(result)) == expected_num


@pytest.mark.parametrize("teff, logg, feh, expected_num", [
    ([-100, 101, 100], [-0.5, 0.51, 0.5], [-0.5, 0.51, 0.5], 27),
    ([0, 100, 100], [0, 1, 1], [0, 1, 1], 1),
    ([-500, 501, 100], [0, 1.01, 0.5], [0, 1, 1], 33)])
def test_gen_close_params_with_simulator_gets_host_set_to_sims(sim_config, teff, logg, feh, expected_num):
    simulators = sim_config
    simulators.sim_grid["teff_1"] = teff
    simulators.sim_grid["logg_1"] = logg
    simulators.sim_grid["feh_1"] = feh
    result = generate_close_params_with_simulator([5000, 4.5, 0.5], target="host")
    assert isinstance(result, GeneratorType)
    result = list(result)
    assert len(result) == expected_num


@pytest.mark.parametrize("teff, logg, feh", [
    (2900, 4.5, 0.0),
    (5600, 2.5, 0.5),
    (7200, 1.5, -1.5)
])
def test_gen_close_params_simulators_with_none_configured(sim_config, teff, logg, feh):
    simulators = sim_config
    for key in ["teff_1", "teff_2", "logg_1", "logg_2", "feh_1", "feh_2"]:
        simulators.sim_grid[key] = None
    gen_params = list(gen_new_param_values(teff, logg, feh, small=True))
    host_close_none = list(generate_close_params_with_simulator([teff, logg, feh],
                                                                target="host"))
    comp_close_none = list(generate_close_params_with_simulator([teff, logg, feh],
                                                                target="companion"))
    par_result = []
    for t, l, m in itertools.product(*gen_params):
        par_result.append([t, l, m])
    assert par_result == host_close_none
    assert par_result == comp_close_none


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


@pytest.mark.parametrize("parrange, lengths", [
    ([[4000, 4200], [1, 3], [0, 0.5]], (3, 5, 2)),
    ([[4000, 4800], [3, 6], [-0.5, 0.5]], (9, 7, 3)),
    ([[4700, 5100], [4, 4], [-4, -2.5]], (4, 1, 4)),
])
def test_simulators_parrange_affects_returned_parameters(sim_config, parrange, lengths):
    simulators = sim_config
    simulators.starfish_grid["parrange"] = parrange
    teff = np.arange(4000, 5001, 100)
    logg = np.arange(1, 6.01, 0.5)
    feh = np.arange(-4, 1.51, 0.5)
    new_teff, new_logg, new_feh = set_model_limits(teff, logg, feh, simulators.starfish_grid["parrange"])

    assert len(new_teff) == lengths[0]
    assert len(new_logg) == lengths[1]
    assert len(new_feh) == lengths[2]


@pytest.mark.parametrize("limits, lengths", [
    ([[4000, 4200], [1, 3], [0, 0.5]], (3, 5, 2)),
    ([[4000, 4800], [3, 6], [-0.5, 0.5]], (9, 7, 3)),
    ([[4700, 5100], [4, 4], [-4, -2.5]], (4, 1, 4)),
])
def test_set_model_limits(limits, lengths):
    teff = np.arange(4000, 5001, 100)
    logg = np.arange(1, 6.01, 0.5)
    feh = np.arange(-4, 1.51, 0.5)
    new_teff, new_logg, new_feh = set_model_limits(teff, logg, feh, limits)

    assert len(new_teff) == lengths[0]
    assert len(new_logg) == lengths[1]
    assert len(new_feh) == lengths[2]


@pytest.mark.parametrize("name, expected", [
    ("phoenix", [[2300, 12000], [0, 6], [-4, 1]]),
    ("cifist", [[1200, 7000], [2.5, 5], [0, 0]])
])
def test_phoenix_limits(name, expected):
    assert get_phoenix_limits(name) == expected


@pytest.mark.parametrize("name", ["cond", "dusty", "all", "", " "])
def test_phoenix_limits_errors_on_invalid_name(name):
    with pytest.raises(ValueError):
        get_phoenix_limits(name)
