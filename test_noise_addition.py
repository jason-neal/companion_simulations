# #!/usr/bin python
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


def main():
    """ Visually test the addition of Noise using add_noise function
    """
    flux = np.ones(100)
    for i, snr in enumerate([50, 100, 200, 300]):
        plt.plot(add_noise(flux, snr) + 0.05 * i, label="snr={}".format(snr))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
