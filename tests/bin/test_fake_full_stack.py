from bin.iam_fake_full_stack import make_fake_parameter_file
from mingle.utilities.param_file import parse_paramfile


def test_make_fake_parameter_file_creates_file(tmpdir, sim_config):
    """Test fake parameter file creates a file."""
    simulators = sim_config
    simulators.paths["parameters"] = str(tmpdir)
    info = {"star": "teststar", "temp": 9000, "logg": 6, "feh": 0}
    created_file = tmpdir.join("{}_params.dat".format(info["star"].upper()))
    # Check doesn't exist yet
    assert created_file.check(file=0)
    make_fake_parameter_file(info)
    # Check file exists
    assert created_file.check(file=1)


def test_make_fake_parameter_file_puts_values_in_file(tmpdir, sim_config):
    """Test fake parameter file is the same when read back in."""
    simulators = sim_config
    simulators.paths["parameters"] = str(tmpdir)
    info = {"star": "teststar", "temp": 9001, "logg": 6,
            "feh": 0.5, "comp_logg": 4, "comp_fe_h": 5}
    created_file = tmpdir.join("{}_params.dat".format(info["star"].upper()))
    assert created_file.check(file=0)
    make_fake_parameter_file(info)
    assert created_file.check(file=1)
    params = parse_paramfile(created_file)
    assert params == info
