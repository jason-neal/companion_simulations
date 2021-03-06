import os

import pytest

import simulators
from mingle.utilities.phoenix_utils import closest_model_params, generate_close_params
from simulators.bhm_module import (bhm_helper_function, get_bhm_model_pars, setup_bhm_dirs)
from simulators.bhm_script import parse_args


def test_get_bh_model_pars_close_method_returns_close_params():
    pars = get_bhm_model_pars({"temp": 5200, "logg": 4.5, "fe_h": 0.0}, method="close")
    assert pars == list(generate_close_params(closest_model_params(5200, 4.5, 0.0)))

import itertools
def test_get_bh_model_pars_from_config():
    simulators.sim_grid["teff_1"] = [-100, 101, 100]
    simulators.sim_grid["logg_1"] = [-1, 0.51, 0.5]
    simulators.sim_grid["feh_1"] = [-0.5, 0.51, 0.5]

    pars = get_bhm_model_pars({"temp": 5200, "logg": 4.5, "fe_h": 0.0}, method="config")

    expected = []
    for t, l, m in itertools.product([5100, 5200, 5300], [3.5, 4, 4.5, 5], [-0.5, 0, 0.5]):
         expected.append([t, l, m])
    print(pars)
    print(expected)
    print(len(pars), len(expected))
    assert pars == expected



def test_get_bh_model_pars_value_error_for_method():
    with pytest.raises(ValueError):
        get_bhm_model_pars({"temp": 5200, "logg": 4.5, "fe_h": 0.0}, method="some")


@pytest.mark.parametrize("star, obs, chip", [
    ("HD30501", 1, 1),
    ("HD4747", "a", 4)])
def test_bhm_helper_function(sim_config, star, obs, chip):
    simulators = sim_config
    obs_name, params, output_prefix = bhm_helper_function(star, obs, chip)

    assert isinstance(obs_name, str)
    assert isinstance(output_prefix, str)
    assert simulators.paths["spectra"] in obs_name
    assert "-mixavg-tellcorr_" in obs_name
    assert str(star) in obs_name
    assert str(obs) in obs_name
    assert str(chip) in obs_name
    assert os.path.join(star, "bhm", star) in output_prefix
    assert "bhm_chisqr_results" in output_prefix
    assert params["name"] == star.lower()


def test_setup_bhm_dirs_creates_dirs(sim_config, tmpdir):
    simulators = sim_config
    simulators.paths["output_dir"] = str(tmpdir)
    star = "TestStar"
    assert not tmpdir.join(star.upper()).check(dir=True)
    assert not tmpdir.join(star.upper(), "bhm", "plots").check(dir=True)
    # assert not tmpdir.join(star.upper(), "bhm", "grid_plots").check(dir=True)
    # assert not tmpdir.join(star.upper(), "bhm", "fudgeplots").check(dir=True)
    result = setup_bhm_dirs(star)

    assert tmpdir.join(star.upper()).check(dir=True)
    assert tmpdir.join(star.upper(), "bhm", "plots").check(dir=True)
    # assert tmpdir.join(star.upper(), "bhm", "grid_plots").check(dir=True)
    # assert tmpdir.join(star.upper(), "bhm", "fudgeplots").check(dir=True)
    assert result is None


def test_bhm_script_parser():
    args = ["HD30501", "01"]
    parsed = parse_args(args)
    assert parsed.star == "HD30501"
    assert parsed.obsnum == "01"
    assert parsed.chip is None
    assert parsed.suffix is ""
    assert parsed.error_off is False
    assert parsed.disable_wav_scale is False


def test_bhm_script_parser_toggle():
    args = ["HDswitches", "1a", "-c", "4", "--suffix", "_test", "--disable_wav_scale", "--error_off"]
    parsed = parse_args(args)
    assert parsed.star == "HDswitches"
    assert parsed.obsnum == "1a"
    assert parsed.chip == "4"
    assert parsed.suffix is "_test"
    assert parsed.error_off is True
    assert parsed.disable_wav_scale is True


from simulators.common_setup import obs_name_template


@pytest.mark.parametrize("mode, end", [
    ("tell_corr", ".fits"),
    ("h2o_tell_corr", ".fits"),
    ("berv_corr", "_bervcorr.fits"),
    ("h2o_berv_corr", "_bervcorr.fits"),
    ("berv_mask", "_bervcorr_masked.fits"),
    ("h2o_berv_mask", "_bervcorr_masked.fits")])
def test_obs_name_template(sim_config, mode, end):
    simulators = sim_config
    simulators.spec_version = mode
    star = "HD00001"
    obsnum = "1"
    chip = 7

    template = obs_name_template()
    assert "tellcorr" in template
    assert "mixavg" in template
    assert end in template

    fname = template.format(star, obsnum, chip)

    assert fname.startswith("{}-{}-".format(star, obsnum))
    assert end in fname
    if "h2o" in mode:
        assert "-h2otellcorr" in template
        assert "-h2otellcorr" in fname
