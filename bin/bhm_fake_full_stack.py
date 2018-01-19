import argparse
import os

import simulators
from bin.coadd_bhm_analysis import main as anaylyse_main
from bin.coadd_bhm_db import main as db_main
from simulators.bhm_script import main as bhm_script_main
from simulators.fake_simulator import main as fake_generator


def make_fake_parameter_file(info):
    name = os.path.join(simulators.paths["parameters"], "{}_params.dat".format(info["star"].upper()))

    with open(name, "w") as f:
        for key, value in info.items():
            f.write("{0} \t= {1}\n".format(key, value))


def _parser():
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Best host modelling.')
    parser.add_argument("star", help='Star name.', type=str)
    parser.add_argument("num", help='Star observation number.', type=int)
    parser.add_argument("teff", help='Temperature of Star.', type=int)
    parser.add_argument("logg", help='Logg of Star.', type=float)
    parser.add_argument("feh", help='Feh of Star.', type=float)
    parser.add_argument('gamma', help='Gamma radial velocity', type=float, default=0)
    parser.add_argument('-s', '--suffix', type=str, default="",
                        help='Extra name identifier.')
    parser.add_argument('-n', '--noise',
                        help='SNR value. int', default=None, type=int)
    parser.add_argument('-r', '--replace',
                        help='Replace old fake spectra.', action="store_true")
    return parser.parse_args()


def main(star, num, teff, logg, feh, gamma=0, noise=False, suffix="", replace=False):
    chips = range(1, 5)

    starinfo = {"star": star, "temp": teff, "logg": logg, "fe_h": feh}
    make_fake_parameter_file(starinfo)

    params1 = "{}, {}, {}".format(teff, logg, feh)

    fake_generator(star=star, sim_num=num, params1=params1, gamma=gamma, noise=noise,
                   replace=replace, noplots=True, mode="bhm")

    # bhm_script
    for chip in chips:
        bhm_script_main(star=star, obsnum=num, chip=chip, suffix=suffix)

    # Generate db
    db_main(star=star, obsnum=num, suffix=suffix, move=True, replace=True)

    # Selected Analysis
    # anaylyse_main(star=star, obsnum=num, suffix=suffix, mode="smallest_chi2")
    # anaylyse_main(star=star, obsnum=num, suffix=suffix, mode="compare_spectra")
    anaylyse_main(star=star, obsnum=num, suffix=suffix, mode="all")
    # anaylyse_main(star=star, obsnum=num, suffix=suffix, mode="contrast")

    print("Noise level =", noise)


if __name__ == "__main__":
    args = vars(_parser())
    opts = {k: args[k] for k in args}

    main(**opts)
    print("bhm fake analysis")
    print("Original opts", opts)
