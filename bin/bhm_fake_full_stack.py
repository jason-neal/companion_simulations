import argparse
import os

from joblib import Parallel, delayed

import simulators
from bin.coadd_bhm_analysis import main as analyse_main
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
    parser.add_argument('-j', '--n_jobs',
                        help='Number of parallel jobs.', type=int, default=4)
    parser.add_argument('-b', '--betasigma',
                        help='Use beta_sigma SNR estimate.', action="store_true")
    parser.add_argument("--renormalize", help="renormalize before chi2", action="store_true")
    parser.add_argument("-m", "--norm_method", help="Re-normalization method flux to models. Default=scalar",
                        choices=["scalar", "linear"], default="scalar")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--no-plots", help="Do only the simulation and db creation.", action="store_true")
    group.add_argument("--only-plots", help="Only do the plots/analysis for these simulations.", action="store_true")
    return parser.parse_args()


def main(star, num, teff, logg, feh, gamma=0, noise=False, suffix="",
         replace=False, n_jobs=4, betasigma=False,
         renormalize=False, norm_method="scalar", no_plots=False, only_plots=False):
    chips = range(1, 5)

    if not only_plots:
        starinfo = {"star": star, "temp": teff, "logg": logg, "fe_h": feh,
                    "gamma": gamma, "name": star}
        make_fake_parameter_file(starinfo)

        params1 = "{}, {}, {}".format(teff, logg, feh)

        fake_generator(star=star, sim_num=num, params1=params1, gamma=gamma, noise=noise,
                       replace=replace, noplots=True, mode="bhm")

        Parallel(n_jobs=n_jobs)(
            delayed(bhm_script_main)(star=star, obsnum=num, chip=chip, suffix=suffix,
                                     renormalize=renormalize, norm_method=norm_method,
                                     betasigma=betasigma)
            for chip in chips)

        # Generate db
        db_main(star=star, obsnum=num, suffix=suffix, move=True, replace=True)

    if not no_plots:
        # Do Analysis
        try:
            analyse_main(star=star, obsnum=num, suffix=suffix, mode="all")
        except Exception as e:
            print(e)

    print("Noise level =", noise)
    return 0


if __name__ == "__main__":
    args = vars(_parser())
    opts = {k: args[k] for k in args}

    main(**opts)
    print("bhm fake analysis")
    print("Original opts", opts)
