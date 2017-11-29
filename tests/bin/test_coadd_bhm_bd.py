from bin.coadd_bhm_db import parse_args


def test_coadd_bhm_db_parser_defaults():
    args = ["HDdefault", "0", ]
    parsed = parse_args(args)
    assert parsed.star == "HDdefault"
    assert parsed.obsnum == "0"
    assert parsed.suffix is ""
    assert parsed.chunksize == 1000
    assert isinstance(parsed.chunksize, int)
    assert parsed.replace is False
    assert parsed.verbose is False
    assert parsed.move is False


def test_coadd_bhm_db_parser():
    args = ["HDswitches", "1a", "--suffix", "_test", "-v", "-r", "-c", "50000", "-m"]
    parsed = parse_args(args)
    assert parsed.star == "HDswitches"
    assert parsed.obsnum == "1a"
    assert parsed.suffix is "_test"
    assert parsed.chunksize == 50000
    assert isinstance(parsed.chunksize, int)
    assert parsed.replace is True
    assert parsed.verbose is True
    assert parsed.move is True
