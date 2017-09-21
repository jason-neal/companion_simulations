import pytest

from simulators.inherint_alpha_model_HD211847 import iam_helper_function, check_inputs, save_full_iam_chisqr

@pytest.mark.parametrize("star,obs,chip", [
    ("HD30501", 1, 1),("HD4747", "a", 4)])
def test_iam_helper_function(star, obs, chip):
    obs_name, params, output_prefix = iam_helper_function(star, obs, chip)

    assert isinstance(obs_name, str)
    assert isinstance(output_prefix, str)
    assert str(star) in obs_name
    assert str(obs) in obs_name
    assert str(chip) in obs_name
    assert "{0}/{0}".format(star) in output_prefix
    assert "iam_chi2" in output_prefix
    assert params["name"] == star.lower()


@pytest.mark.xfail()
def test_iam_check_inputs():
    assert 0


@pytest.mark.xfail()
def test_save_full_iam_chisqr():
    assert 0