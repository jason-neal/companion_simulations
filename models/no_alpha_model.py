
import copy
from Planet_spectral_simulations import spec_local_norm
"""Model that does not adjust the flux ratio of the two input spectra.

Unnormalized spectra are combined and then normalized together.

The rv shift occurs before the addition.
"""


def no_alpha(rv, host, companion, limits, new_x=None, normalize=True):
    """Entangled spectrum model with entrinsic flux ratio.

    Parameters
    ----------
    rv: float
        rv offset of the companion spectrum.
    host: Specturm
        Unnomalized model Spectrum of host.
    companion: Spectrum
        Unnomalized model Spectrum of companion.

    limits: list of float
        Wavelength limits to apply after rv shift.

    new_x: array, None
        New xaxis to return results for. e.g. observation.xaxis.


    returns:
    no_alpha_spec: Spectrum
        Spectra resulting from no alpha scaling of amplitude.

    """
    # this copy solved my nan issue.
    companion = copy.copy(companion)
    host = copy.copy(host)

    companion.doppler_shift(rv)

    no_alpha_spec = host + companion

    if new_x is not None:
        no_alpha_spec.spline_interpolate_to(new_x)

    if limits is not None:
        no_alpha_spec.wav_select(*limits)

    if normalize:
            """Apply normalization to joint spectrum."""
            if limits is None:
                print("Warning! Limits should be given when using normalization")
                print("specturm for normalize", spec)
            no_alpha_spec = spec_local_norm(no_alpha_spec)
    return no_alpha_spec
