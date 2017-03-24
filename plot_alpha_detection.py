#!/usr/bin/env python
# Plot the chisquare for alpha detection
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    """Plot the chi squared."""
    path = "/home/jneal/Phd/Codes/Phd-codes/Simulations/saves"
    X = np.load(os.path.join(path, "RV_mesgrid.npy"))
    Y = np.load(os.path.join(path, "alpha_meshgrid.npy"))
    snrs = np.load(os.path.join(path, "snr_values.npy"))
    resolutions = np.load(os.path.join(path, "Resolutions.npy"))
    # chisqr_store = np.load(os.path.join(path, "chisquare_data.npy"))
# scipy_chisqr_store = np.load(os.path.join(path, "scipy_chisquare_data.npy"))
    # X, Y = np.meshgrid(RVs, alphas)

    # unpickle parameter
    try:
        with open(os.path.join(path, "input_params.pickle"), "rb") as f:
            input_parameters = pickle.load(f)
    except:
        raise
        input_parameters = (999, 999)

    res_chisqr_snr = dict()
    res_error_chisqr_snr = dict()  # uses sigma on model instead of expected
    for resolution in resolutions:
        chisqr_snr = dict()
        error_chisqr_snr = dict()  # uses sigma on model instead of expected
        for snr in snrs:
            # chisqr_snr[snr] = np.load(os.path.join(path,
            #                          "scipy_chisquare_data_snr_{}.npy".format(snr)
            #                                       ))
            chisqr_snr[snr] = np.load(os.path.join(path,
                                      "scipy_chisquare_data_snr_{0}_res{1}.npy".format(snr, resolution)
                                                   ))
            error_chisqr_snr[snr] = np.load(os.path.join(path,
                                            "error_chisquare_data_snr_{0}_res{1}.npy".format(snr, resolution)
                                                         ))
        res_chisqr_snr[resolution] = chisqr_snr
        res_error_chisqr_snr[resolution] = error_chisqr_snr

    df_list = []   # To make data frame
    for resolution in resolutions:
        for snr in snrs:
            this_chisqr_snr = res_chisqr_snr[resolution][snr]
            this_error_chisqr_snr = res_error_chisqr_snr[resolution][snr]
            # Calculating the minimum location
            print("snr = ", snr)
            # print("min chisquared value", np.min(this_chisqr_snr),
            #      "location", np.argmin(this_chisqr_snr))
            # print("min scipy chisquared value", np.min(this_chisqr_snr),
            #      "location", np.argmin(this_chisqr_snr))
            print("\n Using expectation value chi sqaure")
            print("min scipy chisquared value",
                  np.min(this_chisqr_snr, axis=(0, 1)), "location",
                  np.argmin(this_chisqr_snr))
            print("RV = ", X.ravel()[np.argmin(this_chisqr_snr)], "APLHA = ", Y.ravel()[np.argmin(this_chisqr_snr)])
            print("\n Using sigma value chi sqaure")
            print("min scipy chisquared value",
                  np.min(this_error_chisqr_snr, axis=(0, 1)), "location",
                  np.argmin(this_error_chisqr_snr))
            print("RV = ", X.ravel()[np.argmin(this_error_chisqr_snr)],
                  "APLHA = ", Y.ravel()[np.argmin(this_error_chisqr_snr)], "\n")

            rv_at_min = X.ravel()[np.argmin(this_chisqr_snr)]
            alpha_at_min = Y.ravel()[np.argmin(this_chisqr_snr)]
            df_list.append([resolution, snr, rv_at_min, alpha_at_min, np.min(this_error_chisqr_snr, axis=(0, 1))])
    df = pd.DataFrame(df_list, columns=["Resolution", "SNR", "Recovered RV",
                                        "Recovered Alpha", "chi**2"])
    print(df)

    plt.plot(df.loc[:, "Resolution"], df.loc[:, "Recovered RV"], "o")
    plt.xlabel("Resolutions")
    plt.ylabel("RV")
    plt.show()
    plt.plot(df.loc[:, "SNR"], df.loc[:, "Recovered RV"], "o")
    plt.xlabel("snr")
    plt.ylabel("RV")
    plt.show()
    plt.plot(df.loc[:, "Resolution"], df.loc[:, "Recovered Alpha"], "o")
    plt.xlabel("Resolutions")
    plt.ylabel("Alpha")
    plt.show()
    plt.plot(df.loc[:, "SNR"], df.loc[:, "Recovered Alpha"], "o")
    plt.xlabel("snr")
    plt.ylabel("Alpha")
    plt.show()

    # df_list = []   # To make data frame
    for resolution in resolutions:
        for snr in snrs:
            this_chisqr_snr = res_chisqr_snr[resolution][snr]
            this_error_chisqr_snr = res_error_chisqr_snr[resolution][snr]
            log_chisqr = np.log(this_chisqr_snr)
            log_error_chisqr = np.log(this_error_chisqr_snr)
            plt.figure(figsize=(10, 9))
            plt.suptitle(("Log Chi squared with SNR = {0}, Resolution = {1}\n Correct RV={2},"
                          " Correct alpha={3}").format(snr, resolution, input_parameters[0], input_parameters[1]),
                         fontsize=16)
            plt.subplot(2, 1, 1)
            plt.contourf(X, Y, log_chisqr, 100)
            plt.ylabel("Flux ratio")
            plt.title("Scipy chisquared")
            plt.subplot(2, 1, 2)
            plt.contourf(X, Y, log_error_chisqr, 100)
            plt.title("Sigma chisquared")
            plt.ylabel("Flux ratio")
            plt.xlabel("RV (km/s)")
            plt.show()

            # plt.contourf(X, Y, this_chisqr_snr, 100)
            # plt.title("Chi squared with snr of {} and resolution {}".format(snr, resolution))
            # plt.show()

    plt.show()


if __name__ == "__main__":
    main()
