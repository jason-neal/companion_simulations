"""Testing utilities."""

import os

import pytest

import simulators
from mingle.utilities import param_file
from mingle.utilities.crires_utilities import crires_resolution
from mingle.utilities.io import get_filenames


def test_crires_resolution():
    """Simple manual test of crires resolution Rule of thumb."""
    header = {"INSTRUME": "CRIRES", "HIERARCH ESO INS SLIT1 WID": 0.2, "result": 100000}

    header2 = {"INSTRUME": "CRIRES", "HIERARCH ESO INS SLIT1 WID": 0.4, "result": 50000}

    res_1 = crires_resolution(header)
    res_2 = crires_resolution(header2)

    assert isinstance(res_1, int)
    assert isinstance(res_2, int)
    assert res_1 == header["result"]
    assert res_2 == header2["result"]
    assert res_1 > res_2


@pytest.mark.parametrize("input, expected", [
    ("[20, 15, 2]", [20., 15., 2.]),
    ("[hello, 15, ]", ["hello", "15", ""]),
    ("[# comment, 'goes', 'here']", ["# comment", "goes", "here"]),
])
def test_parse_list_string(input, expected):
    param_file.parse_list_string(input) == expected


def test_parse_paramfile():
    test_param_file = "test_params.dat"

    params = param_file.parse_paramfile(test_param_file, "tests/testdata")

    assert params["name"] == "hd4747"
    assert params["spt"] == "g9v"
    assert params["comp_temp"] == 1733.
    assert params["tau"] == 50463.10


def test_parse_paramfile_errors():
    test_param_file = "noexistent_paramfile.txt"

    with pytest.raises(Exception):
        param_file.parse_paramfile(test_param_file, "tests/testdata")


def test_get_host_params(sim_config):
    """Find host star parameters from param file."""
    simulators = sim_config
    star = "test"
    simulators.paths["parameters"] = "tests/testdata"

    params = param_file.get_host_params(star)

    assert len(params) == 3
    assert params == (5340, 4.65, -0.22)


def test_load_paramfile_returns_parse_paramfile(sim_config):
    simulators = sim_config
    star = "test"
    test_param_file = "test_params.dat"
    simulators.paths["parameters"] = "tests/testdata"

    params = param_file.parse_paramfile(test_param_file, "tests/testdata")
    params2 = param_file.load_paramfile(star)

    assert params == params2
    assert isinstance(params2, dict)


def test_get_filenames_with_one_regex():
    print("cwd", os.getcwd())
    results = get_filenames("tests/testdata/handy_spectra", "detect*")

    assert "detector_masks.json" in results
    assert "detector_snrs.json" in results
    assert len(results) == 2


def test_get_filenames_with_two_regex():
    results = get_filenames("tests/utilities", "test_*", "*_util*")

    assert "test_phoenix_utils.py" in results
    assert "test_simulation_utils.py" in results
    assert "test_spectrum_utils.py" in results
    assert "test_utilities.py" in results
    assert len(results) == 5
