#!/usr/bin/env python
"""create_min_chi2_table.py.

Create Table of minimum Chi_2 values and save to a table.
"""
import argparse
import sys

from joblib import Parallel, delayed

from simulators.iam_script import main


def parse_args(args):
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Do iam simulations on stars.')
    parser.add_argument('star', help='Star names', default=None)
    parser.add_argument('--suffix', help='Suffix to add to the file names.', default="")
    parser.add_argument("-n", "--n_jobs", help="Number of parallel Jobs", default=1, type=int)
    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    star = args.star
    n_jobs = args.pop("n_jobs", 1)

    logging.info("Performing simulations on", star)
    obsnums = {"HD30501": ["1", "2a", "2b", "3"], "HD211847": ["1", "2"], "HD4747": ["1"]}


    def parallelized_main(opts, chip):
        """Run main with different chips in parallel."""
        opts["chip"] = chip
        return main(**opts)


    iam_opts = {"parallel": False, "more_id": args.suffix, "small": True, }

    iam_opts["star"] = star
    for obs in obsnums[star]:
        iam_opts["obsnum"] = obs

        res = Parallel(n_jobs=n_jobs)(delayed(parallelized_main)(iam_opts, chip)
                                          for chip in range(1, 5))
    sys.exit(0)
