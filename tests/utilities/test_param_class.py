from mingle.utilities.phoenix_utils import ParamGeneratorBase, PHOENIXACESTemp, PHOENIXACESAlpha
import pytest


def test_ParamGeneratorBase():
    param = ParamGeneratorBase()

    assert isinstance(param, ParamGeneratorBase)


def test_PHOENIXTemp():
    teff = PHOENIXACESTemp()
    assert isinstance(teff, PHOENIXACESTemp)
    assert isinstance(teff, ParamGeneratorBase)

    assert False


@pytest.mark.parameterize("invalid_teff", [2000, 2301, 7050, 7100, 50000])
def test_no_valid_PHOENIXTemp(invalid_teff):
    teff = PHOENIXACESTemp()
    assert not teff.is_valid(invalid_teff)


@pytest.mark.parameterize("valid_teff", [2300, 4000, 7200, 7800, 10800])
def test_no_valid_PHOENIXTemp(valid_teff):
    teff = PHOENIXACESTemp()
    assert teff.is_valid(valid_teff)
