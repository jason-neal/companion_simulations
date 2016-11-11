#!/usr/bin/env python
#Test Noise Addition:

import numpy as np
import matplotlib.pyplot as plt


def add_noise(flux, SNR):
    "Using the formulation mu/sigma"
    mu = np.mean(flux)
    sigma = mu / SNR
    # Add normal distributed noise at the SNR level.
    noisey_flux = flux + np.random.normal(0, sigma, len(flux))
    return noisey_flux


def add_noise2(flux, SNR):
    "Using the formulation mu/sigma"
    #mu = np.mean(flux)
    sigma = flux / SNR
    # Add normal distributed noise at the SNR level.
    noisey_flux = flux + np.random.normal(0, sigma)
    return noisey_flux


def main():
    """ Visually test the addition of Noise using add_noise function
    """
    flux = np.ones(100)
    for i, snr in enumerate([50, 100, 200, 300]):
        # Test that the standard deviation of the noise is close to the snr level
        print("Applying a snr of {}".format(snr))
        noisey_flux = add_noise(flux, snr)
        std = np.std(noisey_flux)
        print("Standard deviation of signal = {}".format(std))
        SNR = 1 / std
        print("Estimated SNR from stddev = {}".format(SNR))
        plt.plot(noisey_flux + 0.05 * i, label="snr={}".format(snr))

    plt.legend()
    plt.show()








if __name__ == "__main__":
    main()
