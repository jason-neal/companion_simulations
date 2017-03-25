"""Gernate simulated observations from templates."""
import itertools
from collection import defaultdict

from utilities.simulation_utilities import combine_spectra
from models.alpha_model import alpha_model

def generate_observations(model_1, model_2, rv, alpha, resolutions, snrs):
    """Create an simulated obervation for combinations of resolution and snr.

    Parameters
    ---------
    model_1: Dict{resolution: Spectrum}
        Host spectum model convolved to different resolutions.
    model_2: Dict{resolution: Spectrum}
        Companion spectum model convolved to different resolutions.
    rv: float
       Rv offset applied to model_2.
    alpha: flaot
        Flux ratio I(model_2)/I(model_1).
    resolutions: list of int
        List of resolutions to simulate.
    snrs: List of int
        List of snr values to simulate.

    Returns
    -------
    observations: dict[resolution][snr]
        Simulated obserable spectra.

    """
    observations = defaultdict(dict)
    iterator = itertools.product(resolutions, snrs)
    for resolution, snr in iterator:
        # Preform tasks to simulate an observation
        spec_1 = model_1[resolution]

        spec_2 = model_2[resolution]
        spec_2.doppler_shift(rv)
        # model1 and model2 are already normalized and convovled to each resolution using
        # store_convolutions
        combined_model = combine_spectra(spec_1, spec_2, alpha)

        # combined_model.flux = add_noise2(combined_model.flux, snr)
        combined_model.add_noise(snr)

        observations[resolution][snr] = combined_model

    return observations


# Need to check and combine generate observations.
@memory.cache
def generate_observations2(model_1, model_2, rv, alpha, resolutions, snrs,
                          limits):
    """Create an simulated observation for combinations of resolution and snr.

    Paramters
    --------
    model_1: Dict{resolution: Spectrum}
        Host spectum model convolved to different resolutions.
    model_2: Dict{resolution: Spectrum}
        Companion spectum model convolved to different resolutions.
    rv: float
       Rv offset applied to model_2.
    alpha: float
        Flux ratio I(model_2)/I(model_1).
    resolutions: List[int]
        List of resolutions to simulate.
    snrs: List[int]
        List of snr values to simulate.
    limits:

    Returns
    -------
    observations: dict[resolution][snr]
        Simulated obserable spectra.

    """
    observations = defaultdict(dict)
    iterator = itertools.product(resolutions, snrs)
    for resolution, snr in iterator:
        # Preform tasks to simulate an observation
        # spec_1 = model_1[resolution]
        # spec_2 = model_2[resolution]
        # spec_2.doppler_shift(rv)
        # combined_model = combine_spectra(spec_1, spec_2, alpha)

        # Using alpha_model
        combined_model = alpha_model(alpha, rv, model_1[resolution],
                                     model_2[resolution], limits)

        # combined_model.flux = add_noise(combined_model.flux, snr)
        combined_model.add_noise(snr)

        observations[resolution][snr] = combined_model

    return observations


def generate_noise_observations(model_1, resolutions, snrs):
    """Create an simulated obervation for combinations of resolution and snr.

    Parameters
    ---------
    model_1: Dict{resolution: Spectrum}
        Spectrum objects convolved to different resolutions.
    resolutions: List[int]
        Resolutions to simulate.
    snrs: List[int]
        Signal-to-noise levels to simulate on the spectra.

    Returns
    -------
    observations: defaultdict{resolution: dict{snr: Spectrum}}
        Simulated obserable spectra with noise.

    """
    observations = defaultdict(dict)
    iterator = itertools.product(resolutions, snrs)
    for resolution, snr in iterator:
        # Preform tasks to simulate an observation
        spec_1 = model_1[resolution]

        # combined_model = combine_spectra(spec_1, spec_2, alpha)

        #spec_1.flux = add_noise2(spec_1.flux, snr)
        spec_1.add_noise(snr)      # Add noise added to Spectrum class

        observations[resolution][snr] = spec_1

    return observations
