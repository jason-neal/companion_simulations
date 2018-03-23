"""Phoenix Utilities.

Some functions to deal with phoenix models
i.e. searching for models with certain parameters

Jason Neal, January 2017
"""
import glob
import itertools
import logging
import os

import Starfish
import numpy as np
from Starfish.grid_tools import HDF5Interface
from astropy.io import fits
from logutils import BraceMessage as __
from spectrum_overload import Spectrum

# from typing import List
import simulators
from mingle.utilities.norm import spec_local_norm
from mingle.utilities.param_file import parse_paramfile
from mingle.utilities.param_utils import closest_model_params, gen_new_param_values


def load_phoenix_spectrum(phoenix_name, limits=None, normalize=False):
    wav_dir = simulators.starfish_grid["raw_path"]
    wav_model = fits.getdata(os.path.join(wav_dir, "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"))
    wav_model /= 10  # turn into nanometers
    flux = fits.getdata(phoenix_name)
    spec = Spectrum(flux=flux, xaxis=wav_model)

    if normalize:
        spec = spec_local_norm(spec, method="exponential")

    if limits is not None:
        spec.wav_select(*limits)
    return spec


def load_starfish_spectrum(params, limits=None, hdr=False, normalize=False,
                           area_scale=False, flux_rescale=False, wav_scale=True):
    """Load spectrum from hdf5 grid file.

    Parameters
    ----------
    params: list
        Model parameters [teff, logg, Z]
    limits: List[float, float] default=None
        Wavelength limits.
    hdr: bool
       Include the model header. Default False.
    normalize: bool
        Locally normalize the spectrum. Default False.
    area_scale: bool
        Multiply by stellar surface area pi*R**2 (towards Earth)
    flux_rescale: bool
        Convert from /cm to /nm by dividing by 1e7
    wav_scale: bool
        Multiply by wavelength to turn into [erg/s/cm^2]

    Returns
    -------
    spec: Spectrum
        The loaded spectrum as Spectrum object.
    """
    my_hdf5 = HDF5Interface()
    my_hdf5.wl = my_hdf5.wl / 10  # Turn into Nanometer

    if hdr:
        flux, myhdr = my_hdf5.load_flux_hdr(np.array(params))
        spec = Spectrum(flux=flux, xaxis=my_hdf5.wl, header=myhdr)
    else:
        flux = my_hdf5.load_flux(np.array(params))
        spec = Spectrum(flux=flux, xaxis=my_hdf5.wl)

    if flux_rescale:
        spec = spec * 1e-7  # convert flux unit from /cm to /nm

    if area_scale:
        if hdr:
            emitting_area = phoenix_area(spec.header)
            spec = spec * emitting_area
            spec.header["emit_area"] = (emitting_area, "pi*r^2")
        else:
            raise ValueError("No header provided for stellar area scaling")

    if wav_scale:
        # Convert into photon counts, (constants ignored)
        spec = spec * spec.xaxis

    if normalize:
        spec = spec_local_norm(spec, method="exponential")

    if limits is not None:
        if limits[0] > spec.xaxis[-1] or limits[-1] < spec.xaxis[0]:
            logging.warning("Warning: The wavelength limits do not overlap the spectrum."
                            "There is no spectrum left... Check your wavelength, or limits.")
        spec.wav_select(*limits)

    return spec


def load_btsettl_spectrum(params, limits=None, hdr=False, normalize=False, area_scale=False, flux_rescale=False):
    """Load spectrum from hdf5 grid file.

    Parameters
    ----------
    params: list
        Model parameters [teff, logg, Z]
    limits: List[float, float] default=None
        Wavelength limits.
    hdr: bool
       Include the model header. Default False.
    normalize: bool
        Locally normalize the spectrum. Default False.
    area_scale: bool
        Multiply by stellar surface area pi*R**2 (towards Earth)
    flux_rescale: bool
        Convert from /cm to /nm by dividing by 1e7

    Returns
    -------
    spec: Spectrum
        The loaded spectrum as Spectrum object.
    """
    # Starfish.grid["btsettl_hdf5_path"], instrument, ranges=Starfish.grid["parrange"]
    my_hdf5 = HDF5Interface(filename=Starfish.grid["btsettl_hdf5_path"], key_name=Starfish.grid["key_name"])
    my_hdf5.wl = my_hdf5.wl / 10  # Turn into Nanometer

    if hdr:
        flux, myhdr = my_hdf5.load_flux_hdr(np.array(params))
        spec = Spectrum(flux=flux, xaxis=my_hdf5.wl, header=myhdr)
    else:
        flux = my_hdf5.load_flux(np.array(params))
        spec = Spectrum(flux=flux, xaxis=my_hdf5.wl)

    if flux_rescale:
        spec = spec * 1e-7  # convert flux unit from /cm to /nm

    if area_scale:
        if hdr:
            emitting_area = phoenix_area(spec.header)
            spec = spec * emitting_area
            spec.header["emit_area"] = (emitting_area, "pi*r^2")
        else:
            raise ValueError("No header provided for stellar area scaling")
    if normalize:
        spec = spec_local_norm(spec, method="exponential")

    if limits is not None:
        if limits[0] > spec.xaxis[-1] or limits[-1] < spec.xaxis[0]:
            logging.warning("Warning: The wavelength limits do not overlap the spectrum."
                            "There is no spectrum left... Check your wavelength, or limits.")
        spec.wav_select(*limits)

    return spec


def phoenix_area(header):
    """Calculate Surface area for PHOENIX model.

    Input
    -----
    header: Header, dict-like
        PHOENIX header

    Return
    ------
    surface_area: float
        Stellar effective surface area in Gm**2.
    """
    if header is None:
        raise ValueError("Header should not be None.")
    # BUNIT 	'erg/s/cm^2/cm' 	Unit of flux
    # PHXREFF 	67354000000.0	[cm] Effective stellar radius
    radius = phoenix_radius(header)
    surface_area = np.pi * radius ** 2  # Towards Earth
    return surface_area


def phoenix_radius(header):
    """Get PHOENIX effective radius.

    Input
    -----
    header: Header, dict-like
        PHOENIX header

    Returns
    -------
    PHXREFF: float
        Effective radius PHXREFF area in Gm

    """
    radius = header["PHXREFF"] * 1e-11  # cm to Gm
    return radius


def closest_model_params(teff: Union[float, int], logg: Union[float, int], feh: Union[float, int], alpha: Optional[Union[float, int]] = None) -> List[Union[int64, float64]]:
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
    alphas = np.arange(-0.2, 0.3, 0.2)  # use only these alpha values if necessary

    closest_teff = teffs[np.abs(teffs - teff).argmin()]
    closest_logg = loggs[np.abs(loggs - logg).argmin()]
    closest_feh = fehs[np.abs(fehs - feh).argmin()]

    if alpha is not None:
        if abs(float(alpha)) > 0.2:
            logging.warning("Alpha is outside acceptable range -0.2->0.2")
        closest_alpha = alphas[np.abs(alphas - alpha).argmin()]

        return [closest_teff, closest_logg, closest_feh, closest_alpha]
    else:
        return [closest_teff, closest_logg, closest_feh]


def all_aces_params():
    teffs = np.concatenate((np.arange(2300, 7000, 100),
                            np.arange(7000, 12001, 200)))
    loggs = np.arange(0, 6.01, 0.5)
    fehs = np.concatenate((np.arange(-4, -2, 1), np.arange(-2, 1.01, 0.5)))
    alphas = np.arange(-0.2, 1.21, 0.2)
    return teffs, loggs, fehs, alphas


def all_btsettl_params(model="cifist2011_2015"):
    if model == "cifist2011_2015":
        teffs = np.arange(1200, 7000, 100)
        loggs = np.arange(2.5, 5.1, 0.5)
        fehs = np.arange(0, 0.1, 1)
        alphas = np.arange(0, 0.1, 0.2)
    else:
        NotImplementedError("all_btsettl_params not supported for model {0}".format(model))
    return teffs, loggs, fehs, alphas


# find_closest_phoenix_name   # Should change to this
def find_closest_phoenix_name(data_dir, teff, logg, feh, alpha=None, Z=True):
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
        if Z:
            phoenix_glob = ("Z{2:+4.1f}/*{0:05d}-{1:4.2f}{2:+4.1f}.PHOENIX*.fits"
                            "").format(closest_teff, closest_logg, closest_feh)
        else:
            phoenix_glob = ("*{0:05d}-{1:4.2f}{2:+4.1f}.PHOENIX*.fits"
                            "").format(closest_teff, closest_logg, closest_feh)
    phoenix_glob = phoenix_glob.replace("+0.0", "-0.0")  # Replace positive 0 metallicity with negative 0
    logging.debug(__("New Phoenix_glob {0}", phoenix_glob))
    joint_glob = os.path.join(data_dir, phoenix_glob)
    logging.debug(__("Data dir = {0}", data_dir))
    logging.debug(__("Glob path/file {0}", os.path.join(data_dir, phoenix_glob)))
    logging.debug(__("joint Glob path/file {0}", joint_glob))

    files = glob.glob(os.path.join(data_dir, phoenix_glob))
    if len(files) > 1:
        logging.warning("More than one file returned")
    return files


def phoenix_name_from_params(data_dir, paramfile):
    """Return closest phoenix model given a stellar parameter file.

    Obtain temp, metallicity, and logg from parameter file.
    Parameters
    ----------
    data_dir: str
        Directory to phoenix models.
    paramfile: str or dict, or list
        Parameter filename if a string is given.
        Dictionary of parameters if dict is provided, or
        list of parameters in the correct order.

    Returns
    -------
    phoenix_model: str
        Filename of phoenix model closest to given parameters.
    """
    logging.debug(__("phoenix_from_params Data dir = {0}", data_dir))
    if isinstance(paramfile, str):
        params = parse_paramfile(paramfile)
    else:
        params = paramfile

    if isinstance(params, dict):
        if "alpha" not in params.keys():
            params["alpha"] = None
        params = [params["temp"], params["logg"], params["fe_h"], params["alpha"]]

    elif isinstance(params, list):
        if len(params) == 3:
            params = params + [None]  # for alpha
        elif len(params) != 4:
            raise ValueError("Length of parameter list given is not valid, {}".format(len(params)))

    return find_closest_phoenix_name(data_dir, params[0], params[1], params[2], alpha=params[3])


# def find_phoenix_model_names(base_dir: str, original_model: str) -> List[str]:  # mypy
def find_phoenix_model_names(base_dir, original_model):
    """Find other phoenix models with similar temp and metallicities.

    Returns list of model name strings.

    """
    # "lte05200-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    z_dir = "Z" in original_model
    try:
        model_name = os.path.split(original_model)[-1]
    except Exception:
        model_name = original_model
    logging.debug(__("Original_name = {0}", original_model))
    logging.debug(__("model_name = {0}", model_name))
    temp = int(model_name[3:8])
    logg = float(model_name[9:13])
    metals = float(model_name[13:17])

    new_temps, new_loggs, new_metals = gen_new_param_values(temp, logg, metals, small=False)

    close_models = []
    for t, l, m in itertools.product(new_temps, new_loggs, new_metals):
        name = phoenix_name(t, l, m, Z=z_dir)
        name = os.path.join(base_dir, name)

        if os.path.isfile(name):
            close_models.append(name)
    return close_models


def phoenix_name(teff, logg, feh, alpha=None, Z=False):
    if alpha is not None:
        raise NotImplementedError("Need to add alpha to phoenix name.")
    name = os.path.join("lte{0:05d}-{1:1.02f}{2:+1.10}."
                        "PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(teff, logg, feh))
    if Z:
        name = os.path.join("Z{0:+1.10}".format(feh), name)

    if "+0.0" in name:  # Positive zero is not allowed in naming
        name = name.replace("+0.0", "-0.0")
    return name


def phoenix_regex(teff, logg, feh, alpha=None, Z=False):
    if alpha is not None:
        raise NotImplementedError("Need to add alpha to phoenix name.")
    regex = ("*{0:05d}-{1:4.2f}{2:+4.1f}.PHOENIX*.fits"
             "").format(teff, logg, feh)
    if Z:
        regex = os.path.join("Z{0:+1.10}".format(feh), regex)
    if "+0.0" in regex:  # Positive zero is not allowed in naming
        regex = regex.replace("+0.0", "-0.0")
    return regex
