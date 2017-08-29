#!/usr/bin/python
"""Run HD30501 analysis.


"""
import subprocess
observations = ["1", "2a", "2b", "3"]
chips = range(1, 5)



for obs in observations:
    for chip in chips:
        name = "HD30501/HD30501-{0}_{1}_iam_chisqr_results.db".format(obs, chip)

        # Run single componnet models

        #run db cretor



        # Run analysis code to make plots
        subprocess.call("python ../bin/analysis_iam_chi2.py {0}".format(name), shell=True)
