import numpy as np
import pytest

from simulators.fake_simulator import fake_bhm_simulation, fake_iam_simulation
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
    assert parsed.noise == 100.0
    assert isinstance(parsed.noise, float)


def test_fake_sim_main_with_no_params1_returns_error():
    with pytest.raises(ValueError):
        fake_main("hdtest", 1, params1=None, params2=[5800, 4.0, -0.5], rv=7, gamma=5, mode="iam")

    with pytest.raises(ValueError):
        fake_main("hdtest2", 2, params1=None, gamma=7, mode="bmh")

@pytest.mark.parametrize("starname, obsnum",
                         [("teststar", 1),
                          ("SECONDTEST", 9)])
@pytest.mark.parametrize("mode", ["iam", "bhm"])
def test_fake_simulator_main_runs_and_creates_files(sim_config, tmpdir, starname, obsnum, mode):
    """NO gamma or rv provided"""
    simulators = sim_config
    simulators.paths["output_dir"] = str(tmpdir)
    simulators.paths["parameters"] = str(tmpdir)
    simulators.paths["spectra"] = str(tmpdir)
    starname_up = starname.upper()

    def sim_filename(chip):
        return tmpdir.join("{0}-{1}-mixavg-tellcorr_{2}_bervcorr_masked.fits".format(starname_up, obsnum, chip))

    # Simulations for each chip don't exist yet?
    for chip in range(1, 5):
        expected_sim_file = sim_filename(chip)
        assert expected_sim_file.check(file=0)

    result = fake_main(starname, obsnum, "4500, 5.0, 0.5", "2300, 4.5, 0.0", mode=mode)

    # Simulations for each chip were created?
    for chip in range(1, 5):
        expected_sim_file = sim_filename(chip)
        assert expected_sim_file.check(file=1)

    assert result is None


@pytest.mark.parametrize("params", [(2500, 4.5, 0.0), (2800, 4.5, 0.5)])
@pytest.mark.parametrize("wav", [np.linspace(2147, 2160, 200)])
@pytest.mark.parametrize("rv", [5, 2, -6])
@pytest.mark.parametrize("gamma", [-5, 1, 7])
@pytest.mark.parametrize("limits", [[2070, 2180]])
def test_fake_iam_simulation_with_passing_wav(params, wav, rv, gamma, limits):
    fake_wav, fake_flux = fake_iam_simulation(wav, [5000, 4.5, 0.5], params2=params, gamma=gamma, rv=-rv, limits=limits)
    assert np.all(fake_wav < limits[1]) and np.all(fake_wav > limits[0])
    assert np.all(fake_wav == wav)
    assert fake_wav.shape == fake_flux.shape


@pytest.mark.parametrize("params", [(2500, 4.5, 0.0), (2800, 4.5, 0.5)])
@pytest.mark.parametrize("wav", [np.linspace(2130, 2145, 40)])
@pytest.mark.parametrize("rv", [5, 2, -6])
@pytest.mark.parametrize("gamma", [-5, 1, 7])
@pytest.mark.parametrize("limits", [[2070, 2180]])
def test_fake_iam_simulation_with_wav(params, wav, rv, gamma, limits):
    fake_wav, fake_flux = fake_iam_simulation(wav, [5000, 4.5, 0.5], params2=params, gamma=gamma, rv=-rv, limits=limits)
    assert np.all(fake_wav < limits[1]) and np.all(fake_wav > limits[0])
    assert np.all(fake_wav == wav)
    assert fake_wav.shape == fake_flux.shape


@pytest.mark.parametrize("params", [(2500, 4.5, 0.0), (2800, 4.5, 0.5)])
@pytest.mark.parametrize("limits", [[2030, 2180], [2100, 2140]])
def test_fake_iam_simulation_without_wav(params, limits):
    fake_wav, fake_flux = fake_iam_simulation(None, [5800, 5.0, -0.5], params2=params, gamma=5, rv=7, limits=limits)
    assert np.all(fake_wav < limits[1]) and np.all(fake_wav > limits[0])
    assert fake_wav.shape == fake_flux.shape


@pytest.mark.parametrize("wav", [np.linspace(2130, 2145, 40), np.linspace(2147, 2160, 200)])
@pytest.mark.parametrize("gamma", [-5, 1, 7])
@pytest.mark.parametrize("limits", [[2070, 2180]])
def test_fake_bhm_simulation_with_wav(wav, gamma, limits):
    fake_wav, fake_flux = fake_bhm_simulation(wav, [5000, 4.5, 0.5], gamma=gamma, limits=limits)
    assert np.all(fake_wav < limits[1]) and np.all(fake_wav > limits[0])
    assert np.all(fake_wav == wav)
    assert fake_wav.shape == fake_flux.shape


@pytest.mark.parametrize("limits", [[2030, 2180], [2100, 2140]])
def test_fake_bhm_simulation_without_wav(limits):
    fake_wav, fake_flux = fake_bhm_simulation(None, [5000, 4.5, 0.5], gamma=5, limits=limits)
    assert np.all(fake_wav < limits[1]) and np.all(fake_wav > limits[0])
    assert fake_wav.shape == fake_flux.shape
