#!/usr/bin/env python
# Plot the chisquare for alpha detection
import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    """Plot the chi squared."""
    path = "/home/jneal/Phd/Codes/Phd-codes/Simulations/saves/new"
    X = np.load(os.path.join(path, "RV_mesgrid.npy"))
    Y = np.load(os.path.join(path, "alpha_meshgrid.npy"))
    snrs = np.load(os.path.join(path, "snr_values.npy"))

    # chisqr_store = np.load(os.path.join(path, "chisquare_data.npy"))
# scipy_chisqr_store = np.load(os.path.join(path, "scipy_chisquare_data.npy"))
    # X, Y = np.meshgrid(RVs, alphas)
    chisqr_snr = dict()
    for snr in snrs:
        chisqr_snr[snr] = np.load(os.path.join(path,
                                               "scipy_chisquare_data_snr_{}_res50000.npy".format(snr)
                                               ))

    # plt.contourf(X, Y, np.log(chis
    # plt.contourf(X, Y, np.log(chisqr_store), 40)
    # plt.title("my chisquared ".format(snr))
    # plt.show()
    for snr in snrs:
        this_chisqr_snr = chisqr_snr[snr]
        plt.contourf(X, Y, np.log(this_chisqr_snr), 100)
        plt.title("Log Chi squared with snr of {}".format(snr))
        plt.show()
        print("snr = ", snr)
        print("min chisquared value", np.min(this_chisqr_snr),
              "location", np.argmin(this_chisqr_snr))
        print("min scipy chisquared value", np.min(this_chisqr_snr),
              "location", np.argmin(this_chisqr_snr))

    for snr in snrs:
        this_chisqr_snr = chisqr_snr[snr]
        plt.contourf(X, Y, this_chisqr_snr, 100)
        plt.title("Chi squared with snr of {}".format(snr))
        plt.show()
        print("snr = ", snr)
        print("min chisquared value", np.min(this_chisqr_snr),
              "location", np.argmin(this_chisqr_snr))
        print("min scipy chisquared value", np.min(this_chisqr_snr),
              "location", np.argmin(this_chisqr_snr))
        print("min scipy chisquared value",
              np.min(this_chisqr_snr, axis=(0, 1)), "location",
              np.argmin(this_chisqr_snr))

# print("chisqr_store[650]", chisqr_store[650])

    # print("division of chisqr and scipy chisqr",
    # chisqr_store/scipy_chisqr_store)
# main()


if __name__ == "__main__":
    main()
