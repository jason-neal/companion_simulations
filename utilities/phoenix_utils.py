"""Phoenix Utilities.

Some functions to deal with phoenix models
i.e. searching for models with certian parameters

Jason Neal, Janurary 2017
"""
import os
import copy
import glob
import logging
import itertools

import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from typing import List
from utilities.param_file import parse_paramfile


def find_closest_phoenix(data_dir, teff, logg, feh, alpha=None):
    """Find the closest PHOENIX-ACES model to the stellar parameters given.

    alpha parameter is  not implemented yet.
    Parameters
    ----------
    data_dir: str
        Path to the Phoenix-aces folders Z+-.../
    teff: float
    logg: float
    feh: float
    alpha: float (optional)

    Returns
    -------
    phoenix_model: str
        Path/Filename to the closest matching model.

    """
    if alpha is not None:
        raise NotImplemented("Alpha not implemented")

    teffs = np.concatenate((np.arange(2300, 7000, 100),
                            np.arange(7000, 12100, 200)))
    loggs = np.arange(0, 6.1, 0.5)
    fehs = np.concatenate((np.arange(-4, -2, 1), np.arange(-2, 1.1, 0.5)))
    alphas = np.arange(-0.2, 0.3, 0.2)  # use only these alpha values if nesessary

    closest_teff = teffs[np.abs(teffs - teff).argmin()]
    closest_logg = loggs[np.abs(loggs - logg).argmin()]
    closest_feh = fehs[np.abs(fehs - feh).argmin()]

    if alpha is not None:
        if abs(alpha) > 0.2:
            logging.warning("Alpha is outside acceptable range -0.2->0.2")
        closest_alpha = alphas[np.abs(alphas - alpha).argmin()]
        phoenix_glob = ("Z{2:+4.1f}.Alpha={3:+5.2f}/*{0:05d}-{1:4.2f}"
                        "{2:+4.1f}.Alpha={3:+5.2f}.PHOENIX*.fits"
                        "").format(closest_teff, closest_logg, closest_feh,
                                   closest_alpha)
    else:
        phoenix_glob = ("Z{2:+4.1f}/*{0:05d}-{1:4.2f}{2:+4.1f}.PHOENIX*.fits"
                        "").format(closest_teff, closest_logg, closest_feh)
    phoenix_glob = phoenix_glob.replace("+0.0", "-0.0")      # Replace positive 0 metalicity with negative 0
    files = glob.glob(os.path.join(data_dir, phoenix_glob))
    if len(files) > 1:
        logging.warning("More than one file returned")
    return files


def phoenix_from_params(data_dir, parameters):
    """Return cloeset phoenix model given a stellar parameter file.

    Obtain temp, metalicity, and logg from parameter file.
    Parameters
    ----------
    data_dir: str
        Directory to phoenix models.
    parameters: str or dict
        Parameter filename if a string is given. Dictionary of parametes if dict is provided.

    Returns
    -------
    phoenix_model: str
        Filename of phoenix model closest to given parameters.
    """
    logging.debug("phoenix_from_params Data dir = {}".format(data_dir))
    if isinstance(parameters, str):
        params = parse_paramfile(parameters)
    else:
        params = parameters

    if "alpha" not in params.keys():
        params["alpha"] = None
    logging.debug(params)
    return find_closest_phoenix(data_dir, parameters["teff"], parameters["logg"], parameters["fe_h"],
                                alpha=parameters["alpha"])


def find_phoenix_models(base_dir, ref_model, mode="temp"):
    """Find other phoenix models with similar temp and metalicities.

    Parameters
    ----------
    base_dir: str
        Path to phoenix modes HiResFITS folder.
    ref_model:
       Model to start from and search around.
    mode: str
        Mode to find models, "temp" means all metalicity and logg but
        just limit temperature to +/- 400 K, "small" - smaller range of
        +/- 1 logg and metalicity. "all" search all.
        "closest", find the closest matches the given parameters.

    Returns
    -------
    phoenix_models: list[str]
       List of filenames of phoenix models that match mode criteria.

    Notes
    -----
    # Phoenix parameters
    # Parameter   	Range	 Step size
    # Teff [K]	 2300 - 7000	100
    # 	        7000 - 12000	200
    # log(g)	   0.0 - 6.0	0.5
    # [Fe/H]	 -4.0 - -2.0	1.0
    # 	         -2.0 - +1.0	0.5
    # [α/M]	     -0.2 - +1.2	0.2

    """
    teffs = np.concatenate((np.arange(2300, 7000, 100),
                            np.arange(7000, 12100, 200)))
    loggs = np.arange(0, 6.1, 0.5)
    fehs = np.concatenate((np.arange(-4, -2, 1), np.arange(-2, 1.1, 0.5)))
    # alphas = np.arange(-0.2, 0.3, 0.2)  # use only these alpha values if nesessary

    ref_model = ref_model.split("/")[-1]  # Incase has folders in name
    ref_temp = int(ref_model[4:8])
    ref_logg = float(ref_model[9:13])
    ref_feh = float(ref_model[14:17])

    if mode == "all":
        glob_temps = teffs
        glob_loggs = loggs
        glob_fehs = fehs
    elif mode == "temp":
        glob_temps = teffs[((teffs > (ref_temp - 400)) & (teffs < (ref_temp + 400)))]
        glob_loggs = loggs
        glob_fehs = fehs
    elif mode == "small":
        glob_temps = teffs[((teffs > (ref_temp - 400)) & (teffs < (ref_temp + 400)))]
        glob_loggs = loggs[((loggs > (ref_logg - 1)) & (loggs < (ref_logg + 1)))]
        glob_fehs = fehs[((fehs > (ref_feh - 1)) & (fehs < (ref_feh + 1)))]

    file_list = []
    for t_, logg_, feh_ in itertools.product(glob_temps, glob_loggs, glob_fehs):
        phoenix_glob = ("/Z{2:+4.1f}/*{0:05d}-{1:4.2f}{2:+4.1f}.PHOENIX*.fits"
                        "").format(t_, logg_, feh_)
        logging.debug("Phoenix glob", phoenix_glob)
        model_to_find = base_dir + phoenix_glob
        files = glob.glob(model_to_find)
        file_list += files
    logging.debug("file list", file_list)
    phoenix_models = file_list
    # folder_file = ["/".join(f.split("/")[-2:]) for f in phoenix_models]

    return phoenix_models


#def find_phoenix_models2(base_dir: str, original_model: str) -> List[str]:    # mypy
def find_phoenix_models2(base_dir, original_model):
    """Find other phoenix models with similar temp and metalicities.

    Returns list of model name strings.

    """
    # "lte05200-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    try:
        model_name = os.path.split(original_model)[-1]
    except:
        model_name = original_model
    temp = int(model_name[3:8])
    logg = float(model_name[9:13])
    metals = float(model_name[13:17])

    new_temps = np.array(-400, -300, -200, -100, 0, 100, 200, 300, 400) + temp
    new_metals = np.array(-1, -0.5, 0, 0.5, 1) + metals
    new_loggs = np.array(-1, -0.5, 0, 0.5, 1) + logg

    # TODO: Deal with Z folders.
    # z = metalicities
    # "Z{new_metal}/lte0{new_temp}-{newlogg}-{new_metal}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    close_models = []
    for t, l, m in itertools.product(new_temps, new_loggs, new_metals):
        name = os.path.join(base_dir,
                            "lte{:05d}-{:1.20f}{:+1.10}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(t, l, m))

        if "+0.0" in name:   # Positive zero is not alowed in naming
            name = name.replace("+0.0", "-0.0")

        if os.path.isfile(name):
            close_models.append(name)
    return close_models


def local_normalization(wave, flux, splits=50, method="exponential", plot=False):
    r"""Local minimization for section of Phoenix spectra.

    Split spectra into many chunks and get the average of top 5\% in each bin.

    Fit to those points and normalize by that.
    """
    org_flux = copy.copy(flux)
    org_wave = copy.copy(wave)

    while len(wave) % splits != 0:
        # Shorten array untill can be evenly split up.
        wave = wave[:-1]
        flux = flux[:-1]

    flux_split = np.split(flux, splits)
    wav_split = np.split(wave, splits)

    wav_points = np.empty(splits)
    flux_points = np.empty(splits)

    for i, (w, f) in enumerate(zip(wav_split, flux_split)):
        wav_points[i] = np.median(w[np.argsort(f)[-20:]])  # Take the median of the wavelength values of max values.
        flux_points[i] = np.median(f[np.argsort(f)[-20:]])

    if method == "scalar":
        norm_flux = np.median(flux_split) * np.ones_like(org_wave)
    elif method == "linear":
        z = np.polyfit(wav_points, flux_points, 1)
        p = np.poly1d(z)
        norm_flux = p(org_wave)
    elif method == "quadratic":
        z = np.polyfit(wav_points, flux_points, 2)
        p = np.poly1d(z)
        norm_flux = p(org_wave)
    elif method == "exponential":
        z = np.polyfit(wav_points, np.log(flux_points), deg=1, w=np.sqrt(flux_points))

        def _exp_func(x, a, b):
            """Exponetial function from polyfit parameters."""
            return np.exp(a) * np.exp(b * x)

        norm_flux = _exp_func(org_wave, z[0], z[1])

    if plot:
        plt.subplot(211)
        plt.plot(wave, flux)
        plt.plot(wav_points, flux_points, "x-", label="points")
        plt.plot(org_wave, norm_flux, label='norm_flux')
        plt.legend()
        plt.subplot(212)
        plt.plot(org_wave, org_flux / norm_flux)
        plt.title("Normalization")
        plt.show()

    return org_flux / norm_flux


def spec_local_norm(spectrum, splits=50, method="quadratic", plot=False):
    r"""Apply local normalization on Spectrum object.

    Split spectra into many chunks and get the average of top 5\% in each bin.
    """
    norm_spectrum = copy.copy(spectrum)
    flux_norm = local_normalization(spectrum.xaxis, spectrum.flux, splits=splits, plot=plot)
    norm_spectrum.flux = flux_norm

    return norm_spectrum
