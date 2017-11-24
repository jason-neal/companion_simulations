
from bin.coadd_chi2_db import parse_args


def test_coadd_chi2_bd_parser_defaults():
    parsed = parse_args(["HDdefault", "0", ])
    assert parsed.star == "HDdefault"
    assert parsed.obsnum == "0"
    assert parsed.suffix is ""
    assert parsed.chunksize == 1000
    assert isinstance(parsed.chunksize, int)
    assert parsed.replace is False
    assert parsed.verbose is False
    assert parsed.move is False


def test_coadd_chi2_bd_parser():
    parsed = parse_args(["HDswitches", "1a", "--suffix", "_test",
                         "-v", "-r", "-c", "50000", "-m"])
    assert parsed.star == "HDswitches"
    assert parsed.obsnum == "1a"
    assert parsed.suffix is "_test"
    assert parsed.chunksize == 50000
    assert isinstance(parsed.chunksize, int)
    assert parsed.replace is True
    assert parsed.verbose is True
    assert parsed.move is True
