#!/usr/bin/env python
# Test Noise Addition:
import copy
import numpy as np
import matplotlib.pyplot as plt
from utilities.add_noise import add_noise2
from spectrum_overload.Spectrum import Spectrum


def main():
    """Visually test the addition of Noise using add_noise function."""
    org_flux = np.ones(500)
    for i, snr in enumerate([50, 100, 200, 300, 50000]):
        # Test that the standard deviation of the noise is close to the snr level

        print("Applying a snr of {}".format(snr))
        noisey_flux = add_noise2(org_flux, snr)

        spec = Spectrum(flux=copy.copy(org_flux))   # Copy becasue org_flux is mutable when passed in here and was being changed each time.
        spec.add_noise(snr)

        std = np.std(noisey_flux)
        print("Standard deviation of signal = {}".format(std))
        SNR = 1 / std
        SNR_spectrum = 1. / np.std(spec.flux)
        print("Estimated SNR from stddev                   = {}".format(SNR))
        print("Estimated SNR from stddev of Spectrum class = {}".format(SNR_spectrum))
        plt.plot(noisey_flux + 0.1 * i, label="snr={}".format(snr))
        plt.plot(spec.flux + 0.1 * i, "--", label="Spectrum snr={}".format(snr))

        # Calculate chisqr from one
        chi2 = np.sum((noisey_flux - 1)**2 / 1 ** 2)
        print("Chisqr for SNR          {} = \t\t{}".format(snr, chi2))
        chi2_spectrum = np.sum((spec.flux - 1)**2 / 1 ** 2)
        print("Chisqr for SNR_spectrum {} = \t\t{}".format(snr, chi2_spectrum))
        del spec, noisey_flux

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
