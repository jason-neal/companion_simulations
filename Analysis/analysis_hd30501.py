#!/usr/bin/python
"""Run HD30501 analysis.


"""
import argparse
import os
import subprocess
import sys

star = "HD30501"
observations = ["1", "2a", "2b", "3"]
chips = range(1, 5)

# TODO common function to determine observatiosn and chips for different stars


def _parser():
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='HD30501 Analysis.')
    parser.add_argument('--full_chi_calculation', help='Do all chi2 steps (slow).', action="store_true")
    return parser.parse_args()


def main(full_chi_calculation=False):
    cwd = os.getcwd()
    if cwd.endswith("Analysis"):
        prefix_dir = ""
    elif cwd.endswith("companion_simulations"):
        prefix_dir = "Analysis"
    else:
        raise RuntimeError("The cwd is not correct. Check where you are running cwd={}".format(cwd))

    for obs_num in observations:
        for chip in chips:
            db_name = "{0}/{0}-{1}_{2}_iam_chisqr_results.db".format(star, obs_num, chip)
            db_name = os.path.join(prefix_dir, db_name)

            # Run single componet models

            # run db creator
            if full_chi_calculation:
                subprocess.call("python ../iam_chi2_calculator.py {0} {1} -c {2} -s".format(star, obs_num, chip), shell=True)

                # make database


            # Run analysis code to make plots
            if os.getcwd().endswith("Analysis"):
                subprocess.call("python ../bin/analysis_iam_chi2.py {0}".format(db_name), shell=True)
            else:
                subprocess.call("python bin/analysis_iam_chi2.py {0}".format(db_name), shell=True)

if __name__ == "__main__":
    args = vars(_parser())
    opts = {k: args[k] for k in args}
    sys.exit(main(**opts))
