import pytest

from utilities.masking import get_maskinfo, spectrum_masking


@pytest.mark.parametrize("star, obsnum, chip, expected", [
    ("HD30501", "1", "1", [[2112.5418, 2123.4991]]),
    ("HD30501", 1, 2, [[2127.7725, 2137.4103]]),
    ("HD30501", "2b", 3, [[2141.9467, 2151.7604]])
])
def test_get_maskinfo(star, obsnum, chip, expected):
    # Has value of 200
    assert len(get_maskinfo(star, obsnum, chip)) == 1
    assert len(get_maskinfo(star, obsnum, chip)[0]) == 2
    assert get_maskinfo(star, obsnum, chip) == expected


@pytest.mark.parametrize("star, obsnum, chip", [
    ("HD30501", "1", "5"),
    ("HD30501", "1", "0"),
    ("HD30501", "4", "1"),
    ("HD0", "1", "1"),
    ("FHe37823", "1", "1")
])
def test_get_maskinfo_with_bad_key(star, obsnum, chip):
    # Has value of 200
    # This also checks if the datafile exists
    assert get_maskinfo(star, obsnum, chip) == []  # empty mask list


@pytest.mark.parametrize("star, obsnum, chip", [
    ("HD30501", "1", "1"),
    ("HD30501", "2a", "2"),
    ("HD30501", "2b", "3"),
    ("HD30501", "3", "4"),
])
def test_spectrum_masking(host, star, obsnum, chip):
    info = get_maskinfo(star, obsnum, chip)[0]
    assert host.xaxis[0] < info[0]
    assert host.xaxis[-1] > info[1]

    host = spectrum_masking(host, star, obsnum, chip)

    assert not host.xaxis[0] < info[0]
    assert not host.xaxis[-1] > info[1]


def test_spectrum_masking_50pixel_detector_4(host):
    start_len = len(host)
    host = spectrum_masking(host, "HD0", "1", "4")  # Detector 4 but with no recored masks.
    end_len = len(host)
    assert start_len - end_len == 50
