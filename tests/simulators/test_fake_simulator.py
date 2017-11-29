import numpy as np
import pytest
from simulators.fake_simulator import main as fake_main
from simulators.fake_simulator import parse_args


def test_fake_simulator_parser():
    args = ["HD30501", "01", "-p", "2300, 4.5, -3.0", "--params2", "2100, 3.5, 0.0", "-g", "10"]
    parsed = parse_args(args)
    assert parsed.star == "HD30501"
    assert parsed.sim_num == "01"
    assert parsed.params1 == "2300, 4.5, -3.0"
    assert parsed.params2 == "2100, 3.5, 0.0"
    assert parsed.gamma == 10
    assert parsed.rv is None
    assert isinstance(parsed.gamma, float)
    assert parsed.replace is False
    assert parsed.noplots is False
    assert parsed.test is False
    assert parsed.mode == "iam"
    assert parsed.noise is None


def test_fake_simulator_parser_toggle():
    args = ["HDTEST", "02", "-t", "-r", '-v', "10", "-m", "bhm", "-s", "100", "-n"]
    parsed = parse_args(args)
    assert parsed.star == "HDTEST"
    assert parsed.sim_num == "02"
    assert parsed.params1 is None
    assert parsed.params2 is None
    assert parsed.rv == 10
    assert parsed.gamma is None
    assert parsed.replace is True
    assert parsed.noplots is True
    assert parsed.test is True
    assert parsed.mode == "bhm"
    assert parsed.noise == "100"


def test_fake_sim_main_with_no_params1_returns_error():
    with pytest.raises(ValueError):
        fake_main("hdtest", 1, params1=None, params2=[5800, 4.0, -0.5], rv=7, gamma=5, mode="iam")

    with pytest.raises(ValueError):
        fake_main("hdtest2", 2, params1=None, gamma=7, mode="bmh")

