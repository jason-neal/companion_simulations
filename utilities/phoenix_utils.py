"""Phoenix Utilities.

Some functions to deal with phoenix models
i.e. searching for models with certian parameters

Jason Neal, January 2017
"""
import os
import copy
import glob
import logging
import itertools
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.modeling import models, fitting
from spectrum_overload.Spectrum import Spectrum
from utilities.param_file import parse_paramfile

import Starfish
from Starfish.grid_tools import HDF5Interface


def load_normalized_phoenix_spectrum(phoenix_name, limits=None):
    wav_dir = "/home/jneal/Phd/data/PHOENIX-ALL/PHOENIX/"
    wav_model = fits.getdata(os.path.join(wav_dir, "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"))
    wav_model /= 10   # turn into nanometers
    flux = fits.getdata(phoenix_name)
    spec = Spectrum(flux=flux, xaxis=wav_model)
    # Limit to K band
    spec.wav_select(2070, 2350)
    spec = spec_local_norm(spec)
    if limits is not None:
        spec.wav_select(*limits)
    return spec


def load_normalized_starfish_spectrum(params, limits=None, hdr=False):
    """Load spectrum from hdf5 grid file with normaliztion on.

    Helper function in which normalization is turned on.

    Parameters
    ----------
    params: list
        Model parameters [teff, logg, Z]
    limits: list or None
        wl limits, default is None
    hdr: bool
       Include hdr information

    Returns
    -------
    spec: spectrum
        Spectrum object for the given stellar parameters.

    """
    return load_starfish_spectrum(params, normalize=True, limits=limits, hdr=hdr)


def load_starfish_spectrum(params, limits=None, hdr=False, normalize=False):
    """Load spectrum from hdf5 grid file.

    parameters: list
        Model parameters [teff, logg, Z]
    hdr: bool
       Inlcude the model header. Default False.
    normalize: bool
        Locally normalize the spectrum. Default False.
    """
    myHDF5 = HDF5Interface()
    myHDF5.wl = myHDF5.wl / 10   # Turn into Nanometer

    if hdr:
        flux, myhdr = myHDF5.load_flux_hdr(np.array(params))
        spec = Spectrum(flux=flux, xaxis=myHDF5.wl, header=myhdr)
    else:
        flux = myHDF5.load_flux(np.array(params))
        spec = Spectrum(flux=flux, xaxis=myHDF5.wl)

    if normalize:
        spec = spec_local_norm(spec)

    if limits is not None:
        spec.wav_select(*limits)
    return spec


def closest_model_params(teff, logg, feh, alpha=None):
    """Find the closest PHOENIX-ACES model parameters to the stellar parameters given.

    Parameters
    ----------
    teff: float
    logg: float
    feh: float
    alpha: float (optional)

    Returns
    -------
    params: list of floats
        Parameters for the closest matching model.

    """
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

        return [closest_teff, closest_logg, closest_feh, closest_alpha]
    else:
        return [closest_teff, closest_logg, closest_feh]


# find_closest_phoenix_name   # Should change to this
def find_closest_phoenix_name(data_dir, teff, logg, feh, alpha=None):
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
        closest_teff, closest_logg, closest_feh, closest_alpha = closest_model_params(teff, logg, feh, alpha=alpha)
    else:
        closest_teff, closest_logg, closest_feh = closest_model_params(teff, logg, feh, alpha=None)

    if alpha is not None:
        if abs(alpha) > 0.2:
            logging.warning("Alpha is outside acceptable range -0.2->0.2")

        phoenix_glob = ("Z{2:+4.1f}.Alpha={3:+5.2f}/*{0:05d}-{1:4.2f}"
                        "{2:+4.1f}.Alpha={3:+5.2f}.PHOENIX*.fits"
                        "").format(closest_teff, closest_logg, closest_feh,
                                   closest_alpha)
    else:
        phoenix_glob = ("Z{2:+4.1f}/*{0:05d}-{1:4.2f}{2:+4.1f}.PHOENIX*.fits"
                        "").format(closest_teff, closest_logg, closest_feh)
    logging.debug("Old Phoenix_glob {}".format(phoenix_glob))
    phoenix_glob = phoenix_glob.replace("+0.0", "-0.0")      # Replace positive 0 metalicity with negative 0
    logging.debug("New Phoenix_glob {}".format(phoenix_glob))
    joint_glob = os.path.join(data_dir, phoenix_glob)
    logging.debug("Data dir = {}".format(data_dir))
    logging.debug("Glob path/file {}".format(os.path.join(data_dir, phoenix_glob)))
    logging.debug(" joint Glob path/file {}".format(joint_glob))

    files = glob.glob(os.path.join(data_dir, phoenix_glob))
    if len(files) > 1:
        logging.warning("More than one file returned")
    return files


def phoenix_name_from_params(data_dir, parameters):
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
    return find_closest_phoenix_name(data_dir, parameters["teff"], parameters["logg"], parameters["fe_h"],
                                     alpha=parameters["alpha"])


def generate_close_params(params):
    """teff, logg, Z"""
    temp, logg, metals = params[0], params[1], params[2]
    new_temps = np.arange(-400, 401, 100) + temp
    new_metals = np.arange(-1, 1.1, 0.5) + metals
    new_loggs = np.arange(-1, 1.1, 0.5) + logg

    for t, l, m in itertools.product(new_temps, new_loggs, new_metals):
        yield [t, l, m]


def find_phoenix_model_names(base_dir, ref_model, mode="temp"):
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


# def find_phoenix_model_names2(base_dir: str, original_model: str) -> List[str]:    # mypy
def find_phoenix_model_names2(base_dir, original_model):
    """Find other phoenix models with similar temp and metalicities.

    Returns list of model name strings.

    """
    # "lte05200-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    try:
        model_name = os.path.split(original_model)[-1]
    except:
        model_name = original_model
    logging.debug("original_name = {}".format(original_model))
    logging.debug("model_name = {}".format(model_name))
    temp = int(model_name[3:8])
    logg = float(model_name[9:13])
    metals = float(model_name[13:17])

    new_temps = np.arange(-1000, 1001, 100) + temp
    new_metals = np.arange(-1, 1.1, 0.5) + metals
    new_loggs = np.arange(-1, 1.1, 0.5) + logg

    close_models = []
    for t, l, m in itertools.product(new_temps, new_loggs, new_metals):
        if "Z" in original_model:
            name = os.path.join(base_dir,
                                "Z{0:+1.10}".format(m),
                                "lte{0:05d}-{1:1.02f}{2:+1.10}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(t, l, m))
        else:
            name = os.path.join(base_dir,
                                "lte{0:05d}-{1:1.02f}{2:+1.10}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(t, l, m))

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
        p = np.poly1d(z)
        norm_flux = np.exp(p(org_wave))   # Un-log the y values.

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
