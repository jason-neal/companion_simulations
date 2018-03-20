import numpy as np
import pytest

from mingle.models.broadcasted_models import inherent_alpha_model


@pytest.mark.parametrize("rv1", [0, 5])
@pytest.mark.parametrize("rv2", [-50, 0, 3.5, 150])
def test_doppler_shift(comp, rv1, rv2):
    """Test that a dopplershift from alpha model on companion 2 is same as Spectm.doppler_shift."""
    host = comp.copy()
    host.flux = np.zeros_like(host.flux)

    # companion with empty "host"
    model_comp = inherent_alpha_model(host.xaxis, host.flux, comp.flux, rvs=rv2 - rv1, gammas=rv1)

    # shift companion
    doppler_comp = comp.copy()
    doppler_comp.doppler_shift(rv2)

    # remove nans after doppler
    doppler_flux = doppler_comp.flux[~np.isnan(doppler_comp.flux)]
    doppler_wav = doppler_comp.xaxis[~np.isnan(doppler_comp.flux)]

    # interpolating model
    model_comp = model_comp(doppler_wav).squeeze()  # to save another interpolation

    # removing nans from model
    nans = np.isnan(model_comp)
    model_comp = model_comp[~nans]
    doppler_flux = doppler_flux[~nans]

    assert np.allclose(model_comp, doppler_flux, rtol=5.e-2)
