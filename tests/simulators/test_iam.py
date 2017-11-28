import os

import numpy as np
import pytest
from spectrum_overload import Spectrum

import simulators
from simulators.iam_module import (continuum_alpha, iam_analysis,
                                   iam_helper_function, iam_wrapper,
                                   parallel_iam_analysis, setup_iam_dirs)


@pytest.mark.parametrize("star, obs, chip", [
    ("HD30501", 1, 1),
    ("HD4747", "a", 4)])
def test_iam_helper_function(star, obs, chip):
    obs_name, params, output_prefix = iam_helper_function(star, obs, chip)

    assert isinstance(obs_name, str)
    assert isinstance(output_prefix, str)
    assert str(star) in obs_name
    assert str(obs) in obs_name
    assert str(chip) in obs_name
    assert os.path.join(star, "iam", star) in output_prefix
    assert "iam_chisqr" in output_prefix
    assert params["name"] == star.lower()


@pytest.mark.xfail()
def test_save_full_iam_chisqr():
    assert 0


@pytest.mark.xfail()
def test_iam_analysis_same_as_parallel():
    assert parallel_iam_analysis() == iam_analysis()


def test_iam_wrapper(host, comp, tmpdir):
    simulators.paths["output_dir"] = tmpdir
    host_params = [5600, 4.5, 0.0]
    comp_params = [[2300, 4.5, 0.0], [2400, 4.5, 0.0]]

    host.wav_select(2110, 2115)
    comp.wav_select(2110, 2115)

    obs_spec = host.copy()
    obs_spec.flux += comp.flux
    obs_spec.header.update({"OBJECT": "Test_object"})
    setup_iam_dirs("Test_object")

    result = iam_wrapper(0, host_params, comp_params, obs_spec=obs_spec,
                         gammas=[0, 1, 2], rvs=[-1, 1], norm=True,
                         save_only=True, chip=1, prefix=tmpdir.join("TEST_file"))
    assert result is None


def test_iam_wrapper_without_prefix(host, comp, tmpdir):
    simulators.paths["output_dir"] = tmpdir
    host_params = [5600, 4.5, 0.0]
    comp_params = [[2300, 4.5, 0.0], [2400, 4.5, 0.0]]

    host.wav_select(2110, 2115)
    comp.wav_select(2110, 2115)

    obs_spec = host.copy()
    obs_spec.flux += comp.flux
    obs_spec.header.update({"OBJECT": "Test_object","MJD-OBS": 56114.31674297})
    setup_iam_dirs("Test_object")

    result = iam_wrapper(0, host_params, comp_params, obs_spec=obs_spec,
                         gammas=[0, 1, 2], rvs=[-1, 1], norm=True,
                         save_only=True, chip=1)
    assert result is None

@pytest.mark.parametrize("chip", [None, 1, 2, 3, 4])
def test_continuum_alpha(chip):
    x = np.linspace(2100, 2180, 100)
    model1 = Spectrum(xaxis=x, flux=np.ones(len(x)))
    model2 = model1 * 2
    alpha = continuum_alpha(model1, model2, chip)

    assert np.allclose(alpha, [2])


def test_setup_dirs_creates_dirs(tmpdir):
    simulators.paths["output_dir"] = tmpdir
    star = "TestStar"
    assert not os.path.exists(tmpdir.join(star.upper(), "iam"))
    assert not os.path.exists(tmpdir.join(star.upper(), "iam", "plots"))
    assert not os.path.exists(tmpdir.join(star.upper(), "iam", "grid_plots"))
    assert not os.path.exists(tmpdir.join(star.upper(), "iam", "fudgeplots"))
    result = setup_iam_dirs(star)

    assert os.path.exists(tmpdir.join(star.upper(), "iam"))
    assert os.path.exists(tmpdir.join(star.upper(), "iam", "plots"))
    assert os.path.exists(tmpdir.join(star.upper(), "iam", "grid_plots"))
    assert os.path.exists(tmpdir.join(star.upper(), "iam", "fudgeplots"))
    assert result is None


from simulators.iam_script import parse_args


def test_iam_script_parser():
    parsed = parse_args(["HD30501", "01"])
    assert parsed.star == "HD30501"
    assert parsed.obsnum == "01"
    assert parsed.suffix is None
    assert parsed.small is False
    assert parsed.n_jobs == 1
    assert parsed.error_off is False
    assert parsed.area_scale is True
    assert parsed.renormalize is False
    assert parsed.disable_wav_scale is False
    assert parsed.parallel is False


def test_iam_script_parser_toggle():
    args = ["HDswitches", "02", "-c", "4", "-j", "3", "--suffix", "_test",
                         "-n", "-p", "-s", "-a", "--disable_wav_scale", "--error_off"]
    parsed = parse_args(args)
    assert parsed.star == "HDswitches"
    assert parsed.obsnum == "02"
    assert parsed.suffix == "_test"
    assert parsed.small is True
    assert parsed.area_scale is False
    assert parsed.n_jobs == 3
    assert parsed.chip == "4"
    assert parsed.renormalize is True
    assert parsed.disable_wav_scale is True
    assert parsed.error_off is True
    assert parsed.parallel is True
