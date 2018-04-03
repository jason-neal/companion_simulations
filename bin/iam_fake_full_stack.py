import argparse
import os

import numpy as np
from joblib import Parallel, delayed

import simulators
from bin.coadd_analysis_script import main as anaylsis_main
from bin.coadd_chi2_db import main as db_main
from simulators.fake_simulator import main as fake_generator
from simulators.iam_script import main as iam_script_main


os.makedirs(simulators.paths["spectra"], exist_ok=True)  # Check is valid location
os.makedirs(simulators.paths["parameters"], exist_ok=True)  # Check is valid location


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
    parser.add_argument('gamma', help='Gamma radial velocity', type=float, default=0)
    parser.add_argument('rv', help='rv radial velocity of companion', type=float, default=0)
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
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--no-plots", help="Do only the simulation and db creation.", action="store_true")
    group.add_argument("--only-plots", help="Only do the plots/analysis for these simulations.", action="store_true")
    return parser.parse_args()


def main(star, obsnum, teff, logg, feh, teff2, logg2, feh2, gamma=0, rv=0, noise=False, suffix="", replace=False,
         fudge=None, area_scale=True, n_jobs=4, renormalize=False, norm_method="scalar", no_plots=False,
         only_plots=False):
    chips = range(1, 5)

    # Check RV and gamma are inside their defined bounds
    rv_grid = np.arange(*simulators.sim_grid["rvs"])
    gamma_grid = np.arange(*simulators.sim_grid["gammas"])
    assert gamma > np.min(gamma_grid) and gamma < np.max(gamma_grid)
    assert rv > np.min(rv_grid) and rv < np.max(rv_grid)
    if not only_plots:
        starinfo = {"star": star, "temp": teff, "logg": logg, "fe_h": feh,
                    "comp_temp": teff2, "comp_logg": logg2, "comp_fe_h": feh2,
                    "gamma": gamma, "rv": rv, "name": star}
        make_fake_parameter_file(starinfo)

        params1 = "{}, {}, {}".format(teff, logg, feh)
        params2 = "{}, {}, {}".format(teff2, logg2, feh2)

        fake_generator(star=star, sim_num=obsnum, params1=params1, params2=params2, rv=rv, gamma=gamma, noise=noise,
                       replace=replace, noplots=True, mode="iam", fudge=fudge,
                       area_scale=area_scale)

        Parallel(n_jobs=n_jobs)(
            delayed(iam_script_main)(star=star, obsnum=obsnum, chip=chip, suffix=suffix,
                                     area_scale=area_scale, betasigma=True,
                                     renormalize=renormalize, norm_method=norm_method)
            for chip in chips)

        # Generate db
        db_main(star=star, obsnum=obsnum, suffix=suffix, move=True, replace=True)

    if not no_plots:
        # Selected Analysis
        try:
            anaylsis_main(star=star, obsnum=obsnum, suffix=suffix, mode="all")
        except Exception as e:
            print(e)

    print("Noise level =", noise)
    return 0


if __name__ == "__main__":
    args = vars(_parser())
    opts = {k: args[k] for k in args}

    main(**opts)
    print("iam fake analysis")
    print("Original opts", opts)
