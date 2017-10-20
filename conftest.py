import pytest

from models.broadcasted_models import two_comp_model
from utilities.phoenix_utils import load_starfish_spectrum


@pytest.fixture
def host():
    """Host spectrum fixture."""
    mod_spec = load_starfish_spectrum([5200, 4.50, 0.0], limits=[2110, 2170], normalize=True)
    return mod_spec


@pytest.fixture
def comp():
    """Noramlized Companion spectrum fixture."""
    mod_spec = load_starfish_spectrum([2600, 4.50, 0.0], limits=[2110, 2170], normalize=True)
    return mod_spec


@pytest.fixture(params=["scalar", "linear", "quadratic", "exponential"])
def norm_method(request):
    return request.param


@pytest.fixture()
def tcm_model(host, comp):
    return two_comp_model(host.xaxis, host.flux, comp.xaxis, alphas=[0.1, 0.2, 0.3],
                          rvs=[-0.25, 0.25], gammas=[1, 2, 3, 4])
