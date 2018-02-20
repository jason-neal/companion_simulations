#!/usr/bin/env python
"Create some fake observations to test recovery"
import sys

from spectrum_overload import Spectrum

from mingle.models.broadcasted_models import inherent_alpha_model
from mingle.utilities.phoenix_utils import load_starfish_spectrum


def parse_args(args):
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Create synthetic observations.')
    parser.add_argument('simnum', help="Simulation Observation label")
    parser.add_argument('--suffix', default=None,
                        help='Suffix to add to database name.')
    parser.add_argument("-n", "--snr", default=200,
                        help="Noise to add")
    parser.add_argument('-p', '--plot', action="store_true",
                        help='Plot resulting spectrum.')
    parser.add_argument("-m", "--mode", help="Combination mode", choices=["tcm", "bhm", "iam"],
                        default="iam")
    return parser.parse_args(args)


def fake_obs(simnum, snr=200, suffix=None, plot=False, mode="iam"):
    snr = 200

    params1 = [5300, 4.5, 0.0]
    params2 = [2500, 4.5, 0.0]
    gamma = 5
    rv = -3
    normalization_limits = [2100, 2180]

    mod1_spec = load_starfish_spectrum(params1, limits=normalization_limits,
                                       hdr=True, normalize=False, area_scale=True,
                                       flux_rescale=True)

    mod2_spec = load_starfish_spectrum(params2, limits=normalization_limits,
                                       hdr=True, normalize=False, area_scale=True,
                                       flux_rescale=True)

    if broadcast:
        broadcast_result = inherent_alpha_model(mod1_spec.xaxis, mod1_spec.flux, mod2_spec.flux,
                                                rvs=rv, gammas=gamma)

        broadcast_values = broadcast_result(obs_spec.xaxis)
        result_spectrum = Spectrum(flux=broadcast_values, xaxis=mod1_spec.xaxis)
    else:
        # Manually redo the join
        mod2_spec.doppler_shift(rv)
        mod_combine = mod1_spec.copy()
        mod_combine += mod2_spec
        mod_combine.doppler_shift(gamma)
        mod_combine.normalize(method="exponential")
        mod_combine.interpolate(obs_spec.xaxis)
        result_spectrum = mod_combine
        # result_spectrum = Spectrum(flux=broadcast_values, xaxis=obs_spec.xaxis)

    result_spectrum.add_noise(snr)

    # Save as
    # Detector limits
    dect_limits = [(2112, 2123), (2127, 2137), (2141, 2151), (2155, 2165)]

    for ii, dect in enumerate(dect_limits):
        spec = result_spectrum.copy()
        spec.wav_select(*dect)
        spec.resample(1024)

        name = "HDsim-{0}_{1}_snr_{2}".format(sim_num, ii, snr)
        # spec.save...


if __name__ == "__main__":
    args = vars(parse_args(sys.argv[1:]))
    opts = {k: args[k] for k in args}

    sys.exit(fake_obs(**opts))
