import pytest

from simulators.common_setup import obs_name_template, sim_helper_function


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


@pytest.mark.parametrize("mode", [
    "tellcorr", "", "bad"])
def test_obs_name_template_invalid_mode(sim_config, mode):
    simulators = sim_config
    simulators.spec_version = mode

    with pytest.raises(ValueError, match="spec_version {} is not valid".format(mode)):
        obs_name_template()


def test_obs_name_template_warns_on_None_mode(sim_config):
    simulators = sim_config
    simulators.spec_version = None

    with pytest.warns(UserWarning,
                      match="No spec_version specified in config.yaml. Defaulting to berv_mask template."):
        obs_name_template()


@pytest.mark.parametrize("mode", ["iam", "bhm", "tcm"])
@pytest.mark.parametrize("star, obs, chip", [
    ("HD30501", 1, 1),
    ("HD4747", "a", 4)])
def test_sim_helper_function(star, obs, chip, mode):
    obs_name, params, output_prefix = sim_helper_function(star, obs, chip, skip_params=False, mode=mode)

    assert isinstance(obs_name, str)
    assert isinstance(output_prefix, str)
    assert str(star) in obs_name
    assert str(obs) in obs_name
    assert str(chip) in obs_name
    assert os.path.join(star, mode, star) in output_prefix
    assert "{}_chisqr".format(mode) in output_prefix
    assert params["name"] == star.lower()

@pytest.mark.parametrize("star, obs, chip", [
    ("HD30501", 1, 1),
    ("HD4747", "a", 4)])
def test_sim_helper_function_no_params(star, obs, chip):
    obs_name, params, output_prefix = sim_helper_function(star, obs, chip, mode="iam", skip_params=True)

    assert params == {}  # Empty when skipping params
    assert isinstance(obs_name, str)
    assert isinstance(output_prefix, str)


import os

@pytest.mark.parametrize("mode", ["abc", "1234"])
def test_sim_helper_function_bad_mode(mode):
    """Assert Error raised"""
    with pytest.raises(ValueError, match="Mode {} for sim_helper_function not in 'iam, tcm, bhm'".format(mode)):
        sim_helper_function("HD30501", 1, 1, mode=mode)