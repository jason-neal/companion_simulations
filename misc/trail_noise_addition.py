#!/usr/bin/env python
# Test Noise Addition:
import copy

import numpy as np

import matplotlib.pyplot as plt
from spectrum_overload import Spectrum
from utilities.simulate_utilities import add_noise


def main():
    """Visually test the addition of Noise using add_noise function."""
    org_flux = np.ones(500)
    for i, snr in enumerate([50, 100, 200, 300, 50000]):
        # Test that the standard deviation of the noise is close to the snr level

        print("Applying a snr of {}".format(snr))
        noisey_flux = add_noise(org_flux, snr)

        spec = Spectrum(flux=copy.copy(org_flux))   # Copy becasue org_flux is mutable.
        spec.add_noise(snr)

        std = np.std(noisey_flux)
        print("Standard deviation of signal = {}".format(std))
        snr_est = 1 / std
        snr_spectrum = 1. / np.std(spec.flux)
        print("Estimated SNR from stddev                   = {}".format(snr_est))
        print("Estimated SNR from stddev of Spectrum class = {}".format(snr_spectrum))
        plt.plot(noisey_flux + 0.1 * i, label="snr={}".format(snr))
        plt.plot(spec.flux + 0.1 * i, "--", label="Spectrum snr={}".format(snr))

        # Calculate chisqr from one
        chi2 = np.sum((noisey_flux - 1)**2 / 1 ** 2)
        print("Chisqr for SNR          {} = \t\t{}".format(snr, chi2))
        chi2_spectrum = np.sum((spec.flux - 1)**2 / 1 ** 2)
        print("Chisqr for snr_spectrum {} = \t\t{}".format(snr, chi2_spectrum))

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
