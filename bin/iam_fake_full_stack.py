import argparse
import os
import logging
from joblib import Parallel, delayed

import simulators
from bin.coadd_analysis_script import main as anaylsis_main
from bin.coadd_chi2_db import main as db_main
from simulators.fake_simulator import main as fake_generator
from simulators.iam_script import main as iam_script_main


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
    parser.add_argument("obsnum", help='Star observation number.', type=int)
    parser.add_argument("teff", help='Temperature of Star.', type=int)
    parser.add_argument("logg", help='Logg of Star.', type=float)
    parser.add_argument("feh", help='Feh of Star.', type=float)
    parser.add_argument("teff2", help='Temperature of companion.', type=int)
    parser.add_argument("logg2", help='Logg of  companion.', type=float)
    parser.add_argument("feh2", help='Feh of companion.', type=float)
    parser.add_argument('rv', help='rv radial velocity of companion', type=float, default=0)
    parser.add_argument('gamma', help='Gamma radial velocity', type=float, default=0)
    parser.add_argument('-i', "--independent", help='Independent rv of companion', action="store_true")
    parser.add_argument('-s', '--suffix', type=str, default="",
                        help='Extra name identifier.')
    parser.add_argument('-n', '--noise',
                        help='SNR value int', type=int, default=None)
    parser.add_argument('-r', '--replace',
                        help='Replace old fake spectra.', action="store_true")
    parser.add_argument('-a', '--area_scale',
                        help='Disable area_scaling.', action="store_false")
    parser.add_argument('-j', '--n_jobs',
                        help='Number of parallel jobs.', type=int, default=4)
    parser.add_argument('-f', '--fudge',
                        help='Fudge factor.', default=None)
    parser.add_argument("--renormalize", help="renormalize before chi2", action="store_true")
    parser.add_argument("-m", "--norm_method", help="Re-normalization method flux to models. Default=scalar",
                        choices=["scalar", "linear"], default="scalar")
    return parser.parse_args()


def main(star, obsnum, teff, logg, feh, teff2, logg2, feh2, gamma=0, rv=0,
         noise=False, suffix="", replace=False, independent=False,
         fudge=None, area_scale=True, n_jobs=4,
         renormalize=False, norm_method="scalar"):
    chips = range(1, 5)

    starinfo = {"star": star, "temp": teff, "logg": logg, "fe_h": feh, "comp_temp": teff2}
    make_fake_parameter_file(starinfo)

    params1 = "{}, {}, {}".format(teff, logg, feh)
    params2 = "{}, {}, {}".format(teff2, logg2, feh2)

    fake_generator(star=star, sim_num=obsnum, params1=params1, params2=params2, rv=rv, gamma=gamma, noise=noise,
                   replace=replace, noplots=True, mode="iam", independent=independent, fudge=fudge,
                   area_scale=area_scale)

    Parallel(n_jobs=n_jobs)(
        delayed(iam_script_main)(star=star, obsnum=obsnum, chip=chip, suffix=suffix,
                                 area_scale=area_scale, betasigma=True,
                                 renormalize=renormalize, norm_method=norm_method)
        for chip in chips)

    # Generate db
    db_main(star=star, obsnum=obsnum, suffix=suffix, move=True, replace=True)

    # Selected Analysis
    try:
        anaylsis_main(star=star, obsnum=obsnum, suffix=suffix, mode="all")
    except:
        pass

    print("Noise level =", noise)


if __name__ == "__main__":
    args = vars(_parser())
    opts = {k: args[k] for k in args}

    main(**opts)
    print("iam fake analysis")
    print("Original opts", opts)
