#!/usr/bin/python
"""Run HD30501 analysis.


"""
import argparse
import subprocess
observations = ["1", "2a", "2b", "3"]
chips = range(1, 5)


def _parser():
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='HD30501 Analysis.')
    parser.add_argument('--full_chi_calculation', help='Do all chi2 steps (slow).', action="store_true")
    return parser.parse_args()


def main(full_chi_calculation=False):
    for obs in observations:
        for chip in chips:
            name = "HD30501/HD30501-{0}_{1}_iam_chisqr_results.db".format(obs, chip)

            # Run single componet models

            # run db creator
            if full_chi_calculation:
                subprocess.call("python ../iam_chi2_calculator.py {0} {1} -c {2} -s".format(name, obs_num, chip), shell=True)

            # Run analysis code to make plots
            subprocess.call("python ../bin/analysis_iam_chi2.py {0}".format(name), shell=True)

if __name__ == "__main__":
    opts = vars(_parser())
    opts = {k: args[k] for k in args}
    sys.exit(main(**opts))
