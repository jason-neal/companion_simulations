import pytest

from mingle.utilities.errors import get_snrinfo, spectrum_error


@pytest.mark.parametrize("star, obsnum, chip", [
    ("HD30501", "1", "1"),
    ("HD30501", 1, 1),
])
def test_get_snrinfo(star, obsnum, chip):
    # Has value of 200 and length one
    assert len(get_snrinfo(star, obsnum, chip)) == 1
    assert get_snrinfo(star, obsnum, chip) == [200]


@pytest.mark.parametrize("star, obsnum, chip", [
    ("HD30501", "1", "5"),
    ("HD30501", "1", "0"),
    ("HD30501", "4", "1"),
    ("HD0", "1", "1"),
    ("FHe37823", "1", "1")
])
def test_get_snrinfo_with_bad_key(star, obsnum, chip):
    """If KeyError then it returns None."""
    assert get_snrinfo(star, obsnum, chip) is None


@pytest.mark.parametrize("star, obsnum, chip, expected", [
    ("HD30501", "2a", "2", 1 / 200),
    ("HD30501", 3, 4, 1 / 200),
])
def test_spectrum_error_expected(star, obsnum, chip, expected):
    assert spectrum_error(star, obsnum, chip) == expected


@pytest.mark.parametrize("star, obsnum, chip", [
    ("HD30501", "1", "1"),
    ("HD30501", 1, 1),
])
def test_spectrum_error_inverse_snr(star, obsnum, chip):
    assert spectrum_error(star, obsnum, chip) == 1 / get_snrinfo(star, obsnum, chip)[0]


@pytest.mark.parametrize("star, obsnum, chip", [
    ("HD30501", "1", "1"),
    ("HD30501", 1, 1),
])
def test_spectrum_error_error_off(star, obsnum, chip):
    assert spectrum_error(star, obsnum, chip, error_off=True) is None


@pytest.mark.parametrize("N, j", [
    (4, 1),
    (3, 2),
    (0, 1),
])
def test_betasigma_params_from_config(sim_config, N, j):
    simulators = sim_config
    # Default
    assert simulators.betasigma["N"] == 5
    assert simulators.betasigma["j"] == 2

    # Check they can be changed.
    simulators.betasigma["N"] = N
    assert simulators.betasigma["N"] == N
    simulators.betasigma["j"] = j
    assert simulators.betasigma["j"] == j
