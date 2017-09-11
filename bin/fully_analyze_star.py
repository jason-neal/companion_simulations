#!/usr/bin/python
"""Fully Run analysis for a given star.

"""
import os
import sys
import argparse
import subprocess

chips = range(1, 5)
star_observations = {"HD30501": ["1", "2a", "2b", "3"],
                     "HD211847": ["1", "2"],
                     "HD4747": ["1"]}


def _parser():
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Companion Simulation Analysis.')
    parser.add_argument("star", help="Name of star.")
    parser.add_argument('--full_chi_calculation', help='Do all chi2 steps (slow).', action="store_true")
    parser.add_argument('--suffix', help='Suffix to add to filenames.', default="")
    return parser.parse_args()


script_name = "inherint_alpha_model_HD21184.py"


def main(star, full_chi_calculation=False, suffix=""):
    cwd = os.getcwd()
    if cwd.endswith("Analysis"):
        prefix_dir = ""
    elif cwd.endswith("companion_simulations"):
        prefix_dir = "Analysis"
    else:
        raise RuntimeError("The cwd is not correct. Check where you are running cwd={}".format(cwd))

    observations = star_observations[star]
    for obs_num in observations:
        for chip in chips:
            db_name = "{0}/{0}-{1}_{2}_iam_chisqr_results{3}.db".format(star, obs_num, chip, suffix)
            db_name = os.path.join(prefix_dir, db_name)

            # Run single componet models


            if full_chi_calculation:
                if os.getcwd().endswith("Analysis"):
                    subprocess.call("python ../{3} {0} {1} -c {2} -s".format(star, obs_num, chip, script_name), shell=True)
                    # subprocess.call("python ../iam_chi2_calculator.py {0} {1} -c {2} -s".format(star, obs_num, chip), shell=True)

                    subprocess.call("python make_chi2_db.py '{0}/{0}-{1}_{2}_iam_chisqr_results_part*{3}.csv' -m ", shell=True)
                else:
                    subprocess.call("python {3} {0} {1} -c {2} -s".format(star, obs_num, chip, script_name), shell=True)
                    # subprocess.call("python iam_chi2_calculator.py {0} {1} -c {2} -s".format(star, obs_num, chip), shell=True)
                    subprocess.call("python make_chi2_db.py 'Analysis/{0}/{0}-{1}_{2}_iam_chisqr_results_part*{3}.csv' -m ".format(star, obs_num), shell=True)


            # Run analysis code to make plots
            if os.getcwd().endswith("Analysis"):
                subprocess.call("python ../bin/analysis_iam_chi2.py {0}".format(db_name), shell=True)
                subprocess.call("python ../bin/create_min_chi2_table.py -s {0} --suffix {1}".format(star, suffix), shell=True)

            else:
                subprocess.call("python bin/analysis_iam_chi2.py {0}".format(db_name), shell=True)
                subprocess.call("python bin/create_min_chi2_table.py -s {0} --suffix {1}".format(star, suffix), shell=True)


if __name__ == "__main__":
    args = vars(_parser())
    opts = {k: args[k] for k in args}
    sys.exit(main(**opts))
