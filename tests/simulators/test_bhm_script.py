import pytest

from simulators.bhm_script import (bhm_helper_function, get_model_pars,
                                   save_pd_cvs)


@pytest.mark.xfail()
def test_save_pd_cvs(tmpdir):
    assert 0


def test_get_model_pars():
    pars = get_model_pars({"temp": 5200, "logg": 4.5, "fe_h": 0.0})
    assert pars == False
    assert False
