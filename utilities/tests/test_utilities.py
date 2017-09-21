"""Testing crires utilities."""

import pytest

# from utilities.crires_utilities import crires_resolution
from utilities.crires_utilities import crires_resolution
import utilities.param_file as paramfile


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
    paramfile.parse_list_string(input) == expected


def test_parse_paramfile():
    test_param_file = "test_paramfile.txt"

    params = paramfile.parse_paramfile(test_param_file, "utilities/tests/test_data")

    assert params["name"] == "hd4747"
    assert params["spt"] == "g9v"
    assert params["comp_temp"] == 1733.
    assert params["tau"] == 50463.10


def test_parse_paramfile_errors():
    test_param_file = "noexistent_paramfile.txt"

    with pytest.raises(Exception):
        paramfile.parse_paramfile(test_param_file, "utilities/tests/test_data")
