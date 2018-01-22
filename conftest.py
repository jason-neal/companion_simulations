import pytest

from mingle.models.broadcasted_models import two_comp_model
from mingle.utilities.phoenix_utils import load_starfish_spectrum


def set_simulators(simulators):
    # Adjust simulator parameters for testing.
    simulators.starfish_grid["raw_path"] = "./tests/testdata/"
    simulators.starfish_grid["hdf5_path"] = "./tests/testdata/PHOENIX_CRIRES_50k_test.hdf5"
    simulators.starfish_grid["parname"] = ["temp", "logg", "Z"]
    simulators.starfish_grid["key_name"] = "t{0:.0f}g{1:.1f}z{2:.1f}"  # Specifies how the params are stored
    simulators.starfish_grid["parrange"] = [[2300, 12000], [1.0, 6.0], [-4.0, 1.5]]
    simulators.starfish_grid["wl_range"] = [21100, 21650]
    simulators.paths["parameters"] = "./tests/testdata/parameter_files/"
    simulators.paths["spectra"] = "./tests/testdata/handy_spectra/"
    simulators.paths["output_dir"] = "./tests/testdata/Analysis/"
    simulators.spec_version = "berv_mask"
    simulators.betasigma = {"N": 5, "j": 2}


@pytest.fixture
def sim_config():
    # Set reusable default test configuration.
    import simulators
    # Ideally would get/store initial values and reset with those.
    set_simulators(simulators)

    yield simulators
    # Clean up
    set_simulators(simulators)


@pytest.fixture
def host():
    """Host spectrum fixture."""
    mod_spec = load_starfish_spectrum([5200, 4.50, 0.0], limits=[2110, 2165], normalize=True)
    return mod_spec


@pytest.fixture
def comp():
    """Normalized Companion spectrum fixture."""
    mod_spec = load_starfish_spectrum([2600, 4.50, 0.0], limits=[2110, 2165], normalize=True)
    return mod_spec


@pytest.fixture(params=["scalar", "linear", "quadratic", "exponential"])
def norm_method(request):
    return request.param


@pytest.fixture()
def tcm_model(host, comp):
    return two_comp_model(host.xaxis, host.flux, comp.xaxis, alphas=[0.1, 0.2, 0.3],
                          rvs=[-0.25, 0.25], gammas=[1, 2, 3, 4])
