import datetime
import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

import simulators
from mingle.models.broadcasted_models import inherent_alpha_model
from mingle.utilities.chisqr import chi_squared
from mingle.utilities.norm import chi2_model_norms, continuum, arbitrary_rescale, arbitrary_minimums
from mingle.utilities.param_file import parse_paramfile
from mingle.utilities.phoenix_utils import load_starfish_spectrum
from mingle.utilities.simulation_utilities import check_inputs, spec_max_delta


def iam_helper_function(star, obsnum, chip):
    """Specifies parameter files and output directories given observation parameters."""
    param_file = os.path.join(simulators.paths["parameters"], "{0}_params.dat".format(star))
    params = parse_paramfile(param_file, path=None)
    obs_name = os.path.join(
        simulators.paths["spectra"], "{0}-{1}-mixavg-tellcorr_{2}.fits".format(star, obsnum, chip))
    output_prefix = os.path.join(
        simulators.paths["output_dir"], star.upper(), "{0}-{1}_{2}_iam_chisqr_results".format(
            star.upper(), obsnum, chip))

    return obs_name, params, output_prefix


def setup_iam_dirs(star):
    os.makedirs(os.path.join(simulators.paths["output_dir"], star.upper()), exist_ok=True)
    os.makedirs(os.path.join(simulators.paths["output_dir"], star.upper(), "plots"), exist_ok=True)
    os.makedirs(os.path.join(simulators.paths["output_dir"], star.upper(), "grid_plots"), exist_ok=True)
    os.makedirs(os.path.join(simulators.paths["output_dir"], star.upper(), "fudgeplots"), exist_ok=True)
    return None


def iam_analysis(obs_spec, model1_pars, model2_pars, rvs=None, gammas=None,
                 verbose=False, norm=False, save_only=True, chip=None,
                 prefix=None, errors=None, area_scale=False, wav_scale=True):
    """Run two component model over all model combinations."""
    rvs = check_inputs(rvs)
    gammas = check_inputs(gammas)

    if isinstance(model1_pars, list):
        logging.debug("Number of close model_pars returned {0}".format(len(model1_pars)))
    if isinstance(model2_pars, list):
        logging.debug("Number of close model_pars returned {0}".format(len(model2_pars)))

    # Solution Grids to return
    iam_grid_chisqr_vals = np.empty((len(model1_pars), len(model2_pars)))

    args = [model2_pars, rvs, gammas, obs_spec]
    kwargs = {"norm": norm, "save_only": save_only, "chip": chip,
              "prefix": prefix, "verbose": verbose, "errors": errors,
              "area_scale": area_scale, "wav_scale": wav_scale}

    for ii, params1 in enumerate(tqdm(model1_pars)):
        iam_grid_chisqr_vals[ii] = iam_wrapper(ii, params1, *args, **kwargs)

    if save_only:
        return None
    else:
        return iam_grid_chisqr_vals  # Just output the best value for each model pair


def parallel_iam_analysis(obs_spec, model1_pars, model2_pars, rvs=None,
                          gammas=None, verbose=False, norm=False,
                          save_only=True, chip=None, prefix=None,
                          errors=None, area_scale=False, wav_scale=True):
    """Run two component model over all model combinations."""
    rvs = check_inputs(rvs)
    gammas = check_inputs(gammas)

    if isinstance(model1_pars, list):
        logging.debug("Number of close model_pars returned {0}".format(len(model1_pars)))
    if isinstance(model2_pars, list):
        logging.debug("Number of close model_pars returned {0}".format(len(model2_pars)))

    def filled_iam_wrapper(num, param):
        """Fill in all extra parameters for parallel wrapper."""
        return iam_wrapper(num, param, model2_pars, rvs, gammas,
                           obs_spec, norm=norm, save_only=save_only,
                           chip=chip, prefix=prefix, verbose=verbose, errors=errors,
                           area_scale=area_scale, wav_scale=wav_scale)

    print("Parallelized running\n\n\n ###################")
    # raise NotImplementedError("Need to fix this up")
    iam_grid_chisqr_vals = Parallel(n_jobs=-2)(
        delayed(filled_iam_wrapper)(ii, param) for ii, param in enumerate(model1_pars))
    # iam_grid_chisqr_vals = Parallel(n_jobs=-2)(
    #     delayed(iam_wrapper)(ii, param, model2_pars, rvs, gammas,
    #                          obs_spec, norm=norm, save_only=save_only,
    #                          chip=chip, prefix=prefix, verbose=verbose)
    #     for ii, param in enumerate(model1_pars))

    if prefix is None:
        prefix = ""
    prefix += "_parallel"
    args = [model2_pars, rvs, gammas, obs_spec]
    kwargs = {"norm": norm, "save_only": save_only, "chip": chip,
              "prefix": prefix, "verbose": verbose, "errors": errors,
              "area_scale": area_scale, "wav_scale": wav_scale}

    iam_grid_chisqr_vals = Parallel(n_jobs=-2)(
        delayed(iam_wrapper)(ii, param, *args, **kwargs)
        for ii, param in enumerate(model1_pars))
    # iam_grid_chisqr_vals = np.empty_like(model1_pars)
    # for ii, param in enumerate(model1_pars):
    #    iam_grid_chisqr_vals[ii] = iam_wrapper(ii, param, *args, **kwargs)

    return iam_grid_chisqr_vals  # Just output the best value for each model pair


def continuum_alpha(model1, model2, chip=None):
    """Inherent flux ratio between the continuum of the two models.

    Assumes already scaled by area.
    Takes mean alpha of chip or full
    """
    assert not np.any(np.isnan(model1.xaxis))
    assert not np.any(np.isnan(model1.flux))
    assert not np.any(np.isnan(model2.xaxis))
    assert not np.any(np.isnan(model2.flux))
    # Fit models with continuum
    cont1 = continuum(model1.xaxis, model1.flux, method="exponential")
    cont2 = continuum(model2.xaxis, model2.flux, method="exponential")

    # Masking for individual chips
    if chip is None:
        chip = -1  # Full Crires range

    all_limits = {-1: [2111, 2169], 1: [2111, 2124], 2: [2125, 2139], 3: [2140, 2152], 4: [2153, 2169]}
    chip_limits = all_limits[chip]

    mask1 = (model1.xaxis > chip_limits[0]) * (model1.xaxis < chip_limits[1])
    mask2 = (model2.xaxis > chip_limits[0]) * (model2.xaxis < chip_limits[1])

    continuum_ratio = cont2[mask2] / cont1[mask1]
    alpha_ratio = np.nanmean(continuum_ratio)

    return alpha_ratio


def iam_wrapper(num, params1, model2_pars, rvs, gammas, obs_spec, norm=False,
                verbose=True, save_only=True, chip=None, prefix=None, errors=None,
                area_scale=True, wav_scale=True):
    """Wrapper for iteration loop of iam. To use with parallelization."""
    if prefix is None:
        sf = os.path.join(
            simulators.paths["output_dir"], obs_spec.header["OBJECT"].upper(),
            "iam_{0}_{1}-{2}_part{6}_host_pars_[{3}_{4}_{5}].csv".format(
                obs_spec.header["OBJECT"].upper(), int(obs_spec.header["MJD-OBS"]), chip,
                params1[0], params1[1], params1[2], num))
        prefix = os.path.join(
            simulators.paths["output_dir"], obs_spec.header["OBJECT"].upper())  # for fudge
    else:
        sf = "{0}_part{4}_host_pars_[{1}_{2}_{3}].csv".format(
            prefix, params1[0], params1[1], params1[2], num)
    save_filename = sf

    if os.path.exists(save_filename) and save_only:
        print("'{0}' exists, so not repeating calculation.".format(save_filename))
        return None
    else:
        if not save_only:
            iam_grid_chisqr_vals = np.empty(len(model2_pars))
        for jj, params2 in enumerate(model2_pars):
            if verbose:
                print(("Starting iteration with parameters: "
                       "{0}={1},{2}={3}").format(num, params1, jj, params2))

            # ### Main Part ###
            rv_limits = observation_rv_limits(obs_spec, rvs, gammas)

            obs_spec = obs_spec.remove_nans()
            assert ~np.any(np.isnan(obs_spec.flux)), "Observation has nan"

            # Load phoenix models and scale by area and wavelength limit
            mod1_spec, mod2_spec = \
                prepare_iam_model_spectra(params1, params2, limits=rv_limits,
                                          area_scale=area_scale, wav_scale=wav_scale)

            # Estimated flux ratio from models
            inherent_alpha = continuum_alpha(mod1_spec, mod2_spec, chip)

            # Combine model spectra with iam model
            mod1_spec.plot(label=params1)
            mod2_spec.plot(label=params2)
            fudge_factor = 1.3
            mod2_spec.flux *= fudge_factor   # fudge factor
            mod2_spec.plot(label="fudged {0}".format(params2))
            plt.title("fudges models")
            plt.legend()

            fudge_prefix = os.path.basename(os.path.normpath(prefix))
            fname = os.path.join(simulators.paths["output_dir"],
                                 obs_spec.header["OBJECT"].upper(), "fudgeplots",
                                 "{1}_fudged_model_spectra_factor={0}_num={2}.png".format(fudge_factor, fudge_prefix,
                                                                                          num))
            plt.savefig(fname)
            warnings.warn("Using a fudge factor")

            iam_grid_func = inherent_alpha_model(mod1_spec.xaxis, mod1_spec.flux, mod2_spec.flux,
                                                 rvs=rvs, gammas=gammas)
            iam_grid_models = iam_grid_func(obs_spec.xaxis)

            # Continuum normalize all iam_gird_models
            def axis_continuum(flux):
                """Continuum to apply along axis with predefined variables parameters."""
                return continuum(obs_spec.xaxis, flux, splits=50, method="exponential", top=5)

            iam_grid_continuum = np.apply_along_axis(axis_continuum, 0, iam_grid_models)

            iam_grid_models = iam_grid_models / iam_grid_continuum

            # ### RE-NORMALIZATION to observations?
            # print("Shape of iam_grid_models before renormalization", iam_grid_models.shape)
            # print("Shape of obs_spec.flux  before renormalization", obs_spec.flux.shape)
            if norm:
                warnings.warn("Scalar Re-normalizing to observations! Try this off!")
                obs_flux = chi2_model_norms(obs_spec.xaxis, obs_spec.flux,
                                            iam_grid_models, method="scalar")
            else:
                warnings.warn("Not Scalar Re-normalizing to observations!")
                obs_flux = obs_spec.flux[:, np.newaxis, np.newaxis]

            plot_iam_grid_slices(obs_spec.xaxis, rvs, gammas, iam_grid_models,
                                 star=obs_spec.header["OBJECT"].upper(),
                                 xlabel="wavelength", ylabel="rv", zlabel="gamma",
                                 suffix="iam_grid_models", chip=chip)

            # Arbitrary_normalization of observation
            iam_grid_models, arb_norm = arbitrary_rescale(iam_grid_models,
                                                          *simulators.sim_grid["arb_norm"])
            print("Arbitrary Normalized iam_grid_model shape.", iam_grid_models.shape)

            # Calculate Chi-squared
            obs_flux = np.expand_dims(obs_flux, -1)  # expand on last axis to match rescale
            iam_norm_grid_chisquare = chi_squared(obs_flux, iam_grid_models, error=errors)

            # print("Broadcast chi-squared values with arb norm", iam_norm_grid_chisquare.shape)

            # Take minimum chi-squared value along Arbitrary normalization axis
            iam_grid_chisquare, arbitrary_norms = arbitrary_minimums(iam_norm_grid_chisquare, arb_norm)
            # print("Broadcast chi-squared values ", iam_grid_chisquare.shape)

            npix = obs_flux.shape[0]  # Number of pixels used

            plot_iam_grid_slices(rvs, gammas, arb_norm, iam_norm_grid_chisquare,
                                 star=obs_spec.header["OBJECT"].upper(),
                                 xlabel="rv", ylabel="gamma", zlabel="Arbitrary Normalization",
                                 suffix="iam_grid_chisquare", chip=chip)

            if not save_only:
                iam_grid_chisqr_vals[jj] = iam_grid_chisquare.ravel()[np.argmin(iam_grid_chisquare)]

            save_full_iam_chisqr(save_filename, params1, params2,
                                 inherent_alpha, rvs, gammas,
                                 iam_grid_chisquare, arbitrary_norms, npix, verbose=verbose)
        if save_only:
            return None
        else:
            return iam_grid_chisqr_vals


def observation_rv_limits(obs_spec, rvs, gammas):
    """Calculate wavelength limits needed to cover RV shifts used."""
    delta = spec_max_delta(obs_spec, rvs, gammas)
    obs_min, obs_max = min(obs_spec.xaxis), max(obs_spec.xaxis)
    return [obs_min - 1.1 * delta, obs_max + 1.1 * delta]


def prepare_iam_model_spectra(params1, params2, limits, area_scale=True, wav_scale=True):
    """Load spectra with same settings."""
    if not area_scale:
        warnings.warn("Not using area_scale. This is incorrect for paper.")
    if not wav_scale:
        warnings.warn("Not using wav_scale. This is incorrect for paper.")
    mod1_spec = load_starfish_spectrum(params1, limits=limits,
                                       hdr=True, normalize=False, area_scale=area_scale,
                                       flux_rescale=True, wav_scale=wav_scale)
    mod2_spec = load_starfish_spectrum(params2, limits=limits,
                                       hdr=True, normalize=False, area_scale=area_scale,
                                       flux_rescale=True, wav_scale=wav_scale)
    assert len(mod1_spec.xaxis) > 0 and len(mod2_spec.xaxis) > 0
    assert np.allclose(mod1_spec.xaxis, mod2_spec.xaxis)

    return mod1_spec, mod2_spec


def save_full_iam_chisqr(filename, params1, params2, alpha, rvs, gammas,
                         iam_grid_chisquare, arbitrary_norms, npix, verbose=False):
    """Save the iterations chisqr values to a cvs."""
    rv_grid, g_grid = np.meshgrid(rvs, gammas, indexing='ij')
    # assert A.shape == rv_grid.shape
    assert rv_grid.shape == g_grid.shape
    assert g_grid.shape == iam_grid_chisquare.shape

    data = {"rv": rv_grid.ravel(), "gamma": g_grid.ravel(),
            "chi2": iam_grid_chisquare.ravel(), "arbnorm": arbitrary_norms.ravel()}

    columns = ["rv", "gamma", "chi2", "arbnorm"]
    len_c = len(columns)

    df = pd.DataFrame(data=data, columns=columns)
    # Update all rows with same value.
    for par, value in zip(["teff_2", "logg_2", "feh_2"], params2):
        df[par] = value

    columns = ["teff_2", "logg_2", "feh_2"] + columns

    if "[{0}_{1}_{2}]".format(params1[0], params1[1], params1[2]) not in filename:
        # Need to add the model values.
        for par, value in zip(["teff_1", "logg_1", "feh_1"], params1):
            df[par] = value
        columns = ["teff_1", "logg_1", "feh_1"] + columns

    df["alpha"] = alpha
    df["npix"] = npix
    columns = columns[:-len_c] + ["alpha", "npix"] + columns[-len_c:]

    df = df.round(decimals={"logg_2": 1, "feh_2": 1, "alpha": 4,
                            "rv": 3, "gamma": 3, "chi2": 4})

    exists = os.path.exists(filename)
    if exists:
        df[columns].to_csv(filename, sep=',', mode="a", index=False, header=False)
    else:
        # Add header at the top only
        df[columns].to_csv(filename, sep=',', mode="a", index=False, header=True)

    if verbose:
        print("Saved chi-squared values to {0}".format(filename))
    return None


def plot_iam_grid_slices(x, y, z, grid, xlabel=None, ylabel=None, zlabel=None, suffix=None, star=None,
                         chip=None):
    """Slice up 3d grid and plot slices."""
    os.makedirs(os.path.join(simulators.paths["output_dir"], star.upper(), "grid_plots"), exist_ok=True)
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing="ij")

    if xlabel is None:
        xlabel = "x"
    if ylabel is None:
        ylabel = "y"
    if zlabel is None:
        zlabel = "z"

    if len(z) > 1:
        for ii, y_val in enumerate(y):
            plt.subplot(111)
            try:
                xii = x_grid[:, ii, :]
                zii = z_grid[:, ii, :]
                grid_ii = grid[:, ii, :]
                plt.contourf(xii, zii, grid_ii)
            except IndexError:
                print("grid.shape", grid.shape)
                print("shape of x, y, z", x.shape, y.shape, z.shape)
                print("shape of x_grid, y_grid, z_grid", x_grid.shape, y_grid.shape, z_grid.shape)
                print("index value", ii, "y_val ", y_val)
                raise

            plt.xlabel(xlabel)
            plt.ylabel(zlabel)
            plt.title("Grid slice for {0}={1}".format(ylabel, y_val))

            plot_name = os.path.join(simulators.paths["output_dir"], star, "grid_plots",
                                 "y_grid_slice_{0}_chip-{1}_{2}_{3}_{4}_{5}_{6}_{7}.png".format(star, chip, xlabel,
                                                                                                ylabel, zlabel, ii,
                                                                                                suffix,
                                                                                                datetime.datetime.now()))
            plt.savefig(plot_name)
            plt.close(plt.gcf())

    for jj, z_val in enumerate(z):
        plt.subplot(111)
        try:
            xjj = x_grid[:, :, jj]
            yjj = y_grid[:, :, jj]
            grid_jj = grid[:, :, jj]
            plt.contourf(xjj, yjj, grid_jj)
        except IndexError:
            print("shape of x, y, z", x.shape, y.shape, z.shape)
            print("shape of x_grid, y_grid, z_grid", x_grid.shape, y_grid.shape, z_grid.shape)
            print("index value", jj, "y_val ", z_val)
            raise

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.title("Grid slice for {0}={1}".format(zlabel, z_val))
        plot_name = os.path.join(simulators.paths["output_dir"], star, "grid_plots",
                                 "z__grid_slice_{0}_chip-{1}_{2}_{3}_{4}_{5}_{6}_{7}.png".format(star, chip, xlabel,
                                                                                                 ylabel, zlabel, jj,
                                                                                                 suffix,
                                                                                                 datetime.datetime.now()))
        plt.savefig(plot_name)
        plt.close(plt.gcf())
