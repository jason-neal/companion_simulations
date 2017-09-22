import pytest

from simulators.bhm_script import bhm_helper_function, get_model_pars, save_pd_cvs


@pytest.mark.xfail()
def test_save_pd_cvs(tmpdir):
    assert 0
