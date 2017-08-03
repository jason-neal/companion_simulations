
# New Version of alpha_detection using Parallel and
# methodolgy from grid_chisquare.

from __future__ import division, print_function

import copy
import logging

import matplotlib.pyplot as plt
import numpy as np

from models.alpha_model import alpha_model
from Planet_spectral_simulations import load_PHOENIX_hd30501
from spectrum_overload.Spectrum import Spectrum
from utilities.chisqr import parallel_chisqr, spectrum_chisqr
# self written modules
from utilities.debug_utils import pv
from utilities.simulation_utilities import combine_spectra, spectrum_plotter

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')
debug = logging.debug


def main():
    """Chisquare determinination to detect minimum alpha value."""
    path = "/home/jneal/Phd/Codes/Phd-codes/Simulations/saves2"  # save path
    print("Loading Data")

    chip_limits = [2080, 2220]

    org_star_spec, org_bd_spec = load_PHOENIX_hd30501(limits=chip_limits, normalize=True)

    # org_star_spec = Spectrum(xaxis=w_mod, flux=I_star)
    # org_bd_spec = Spectrum(xaxis=w_mod, flux=I_bdmod, calibrated=True)

    ##################
    # Assume binary stars
    org_bd_spec = copy.copy(org_star_spec)
    ##########

    alphas = 10 ** np.linspace(-2, -0.2, 79)
    rvs = np.arange(10, 40, 0.1)

    # RV and alpha value of Simulations
    rv_val = 20.2
    alpha_val = 0.1  # Vary this to determine detection limit
    input_parameters = (rv_val, alpha_val)

    # # Testing alpha_model
    # model_spec = alpha_model(org_star_spec, org_bd_spec, alpha_val, rv_val)

    # plt.plot(org_star_spec.xaxis, org_bd_spec.flux, label="star")
    # plt.plot(model_spec.xaxis, model_spec.flux, label="alpha model")
    # plt.show()

    # Alpha model seems ok
    new_limits = [2100, 2200]
    # Create observation
    simlulated_obs = alpha_model(alpha_val, rv_val, org_star_spec, org_bd_spec, new_limits)

    # function to run parallel

    params = (org_star_spec, org_bd_spec)
    chisqr_parallel = parallel_chisqr(alphas, rvs, simlulated_obs, alpha_model,
                                      (org_star_spec, org_bd_spec, new_limits), n_jobs=4)
    chisqr_parallel = np.array(chisqr_parallel)
    print("Chisqr from parallel run")
    print(chisqr_parallel)
    print("Finished Chisqr parallel run")

    reshape1 = chisqr_parallel.reshape(len(alphas), len(rvs))
    # reshape2 = chisqr_parallel.reshape(len(rvs), len(alphas))

    # R, S = np.meshgrid(alphas, rvs)
    T, U = np.meshgrid(rvs, alphas)

    print("shape of reshape1", reshape1.shape)
    # print(reshape2.shape)
    # print(R.shape)
    # print(T.shape)
    # contour_plot_stuff(R, S, np.log10(reshape2), label="reshape2")
    contour_plot_stuff(T, U, np.log10(reshape1), label="reshape1")
    # contour_plot_stuff(R, S, np.log10(reshape1).T, label="reshape1.T")
    # contour_plot_stuff(T, U, np.log10(reshape2).T, label="reshape2.T")


def contour_plot_stuff(X, Y, Z, label=""):
    plt.contourf(X, Y, Z)
    plt.title(label)
    plt.show()


if __name__ == "__main__":
    main()
