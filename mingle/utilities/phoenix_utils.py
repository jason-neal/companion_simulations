"""Phoenix Utilities.

Some functions to deal with phoenix models
i.e. searching for models with certain parameters

Jason Neal, January 2017
"""
import glob
import itertools
import logging
import os

import numpy as np
from Starfish.grid_tools import HDF5Interface
from astropy.io import fits
from spectrum_overload import Spectrum

# from typing import List
import simulators
from mingle.utilities.norm import spec_local_norm
from mingle.utilities.param_file import parse_paramfile
from mingle.utilities.simulation_utilities import check_inputs

debug = logging.debug


def load_phoenix_spectrum(phoenix_name, limits=None, normalize=False):
    wav_dir = simulators.starfish_grid["raw_path"]
    wav_model = fits.getdata(os.path.join(wav_dir, "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"))
    wav_model /= 10  # turn into nanometers
    flux = fits.getdata(phoenix_name)
    spec = Spectrum(flux=flux, xaxis=wav_model)

    # Limit to K band
    spec.wav_select(2070, 2350)
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
    limits= List[float, float] default=None
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
            spec = spec * phoenix_area(spec.header)
        else:
            raise ValueError("No header provided for stellar area scaling")

    if wav_scale:
        # Convert into photon counts, (constants ignored)
        spec = spec * spec.xaxis

    if normalize:
        spec = spec_local_norm(spec, method="exponential")

    if limits is not None:
        if limits[0] > spec.xaxis[-1] or limits[-1] < spec.xaxis[0]:
            print("Warning: The wavelength limits do not overlap the spectrum."
                  "There is no spectrum left... Check your wavelength, or limits.")
        spec.wav_select(*limits)

    return spec


def load_btsettl_spectrum(params, limits=None, hdr=False, normalize=False, area_scale=False, flux_rescale=False):
    """Load spectrum from hdf5 grid file.

    Parameters
    ----------
    params: list
        Model parameters [teff, logg, Z]
    limits= List[float, float] default=None
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
    # Starfish.grid["btsettle_hdf5_path"], instrument, ranges=Starfish.grid["parrange"]
    my_hdf5 = HDF5Interface(filename=simulators.grid["btsettle_hdf5_path"], key_name=simulators.grid["key_name"])
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
            spec = spec * phoenix_area(spec.header)
        else:
            raise ValueError("No header provided for stellar area scaling")
    if normalize:
        spec = spec_local_norm(spec, method="exponential")

    if limits is not None:
        if limits[0] > spec.xaxis[-1] or limits[-1] < spec.xaxis[0]:
            print("Warning: The wavelength limits do not overlap the spectrum."
                  "There is no spectrum left... Check your wavelength, or limits.")
        spec.wav_select(*limits)

    return spec


def phoenix_area(header):
    """In units of Gigameters.
    Returns
    -------
    surface_area: float
        Stellar effective surface area. in Gm**2
    """
    if header is None:
        raise ValueError("Header should not be None.")
    # BUNIT 	'erg/s/cm^2/cm' 	Unit of flux
    # PHXREFF 	67354000000.0	[cm] Effective stellar radius
    radius = header["PHXREFF"] * 1e-11  # cm to Gm
    surface_area = np.pi * radius ** 2  # towards Earth
    return surface_area


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
    alphas = np.arange(-0.2, 0.3, 0.2)  # use only these alpha values if necessary

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
    phoenix_glob = phoenix_glob.replace("+0.0", "-0.0")  # Replace positive 0 metallicity with negative 0
    logging.debug("New Phoenix_glob {}".format(phoenix_glob))
    joint_glob = os.path.join(data_dir, phoenix_glob)
    logging.debug("Data dir = {}".format(data_dir))
    logging.debug("Glob path/file {}".format(os.path.join(data_dir, phoenix_glob)))
    logging.debug(" joint Glob path/file {}".format(joint_glob))

    files = glob.glob(os.path.join(data_dir, phoenix_glob))
    if len(files) > 1:
        logging.warning("More than one file returned")
    return files


def phoenix_name_from_params(data_dir, params):
    """Return closest phoenix model given a stellar parameter file.

    Obtain temp, metallicity, and logg from parameter file.
    Parameters
    ----------
    data_dir: str
        Directory to phoenix models.
    params: str or dict, or list
        Parameter filename if a string is given.
        Dictionary of parameters if dict is provided, or
        list of parameters in the correct order.

    Returns
    -------
    phoenix_model: str
        Filename of phoenix model closest to given parameters.
    """
    logging.debug("phoenix_from_params Data dir = {}".format(data_dir))
    if isinstance(params, str):
        params = parse_paramfile(params)
    else:
        params = params

    if isinstance(params, dict):
        if "alpha" not in params.keys():
            params["alpha"] = None
        logging.debug(params)
        return find_closest_phoenix_name(data_dir, params["temp"], params["logg"], params["fe_h"],
                                         alpha=params["alpha"])
    elif isinstance(params, list):
        if len(params) == 3:
            params = params + [None]  # for alpha
        elif len(params) == 4:  # assumes alpha given
            return find_closest_phoenix_name(data_dir, params[0], params[1], params[2],
                                             alpha=params[4])
        else:
            raise ValueError("Length of parameter list given is not valid, {}".format(len(params)))


def generate_close_params(params, small=True, limits="phoenix"):
    """teff, logg, Z."""
    temp, logg, metals = params[0], params[1], params[2]

    new_temps, new_loggs, new_metals = gen_new_param_values(temp, logg, metals, small=small)

    if limits == "phoenix":
        new_temps = new_temps[(new_temps >= 2300) * (new_temps <= 12000)]
        new_loggs = new_loggs[(new_loggs >= 0) * (new_loggs <= 6)]
        new_metals = new_metals[(new_metals >= -4) * (new_metals <= 1)]

    for t, l, m in itertools.product(new_temps, new_loggs, new_metals):
        yield [t, l, m]


def generate_close_params_with_simulator(params, target, small=True, limits="phoenix"):
    """teff, logg, Z.

    "Target" is required to make sure this is used correctly..."""
    temp, logg, metals = params[0], params[1], params[2]
    new_temps, new_loggs, new_metals = gen_new_param_values(temp, logg, metals, small=small)

    if target == "host":
        if simulators.sim_grid["teff_1"] is None:
            pass
        else:
            new_temps = np.arange(*simulators.sim_grid["teff_1"]) + temp
        if simulators.sim_grid["feh_1"] is None:
            pass
        else:
            new_metals = np.arange(*simulators.sim_grid["feh_1"]) + metals
        if simulators.sim_grid["logg_1"] is None:
            pass
        else:
            new_loggs = np.arange(*simulators.sim_grid["logg_1"]) + logg

    elif target == "companion":
        if simulators.sim_grid["teff_2"] is None:
            pass
        else:
            new_temps = np.arange(*simulators.sim_grid["teff_2"]) + temp
        if simulators.sim_grid["feh_2"] is None:
            pass
        else:
            new_metals = np.arange(*simulators.sim_grid["feh_2"]) + metals
        if simulators.sim_grid["logg_2"] is None:
            pass
        else:
            new_loggs = np.arange(*simulators.sim_grid["logg_2"]) + logg

    else:
        raise ValueError("Target must be 'host' or 'companion', not '{}'".format(target))

    # print("simulator new params")
    # print("new_temps", new_temps, "\nnew_loggs", new_loggs, "\nnew_metals", new_metals)

    if limits == "phoenix":
        new_temps = new_temps[(new_temps >= 2300) * (new_temps <= 12000)]
        new_loggs = new_loggs[(new_loggs >= 0) * (new_loggs <= 6)]
        new_metals = new_metals[(new_metals >= -4) * (new_metals <= 1)]

    check_inputs(new_temps)
    check_inputs(new_loggs)
    check_inputs(new_metals)
    for t, l, m in itertools.product(new_temps, new_loggs, new_metals):
        yield [t, l, m]


def gen_new_param_values(temp, logg, metals, small=True):
    if small == "host":
        # only include error bounds.
        new_temps = np.array([-100, 0, 100]) + temp
        new_metals = np.array([-0.5, 0.0, 0.5]) + metals
        new_loggs = np.array([-0.5, 0.0, 0.5]) + logg
    elif small:
        new_temps = np.arange(-600, 1001, 100) + temp
        new_metals = np.array([-0.5, 0.0, 0.5]) + metals
        new_loggs = np.array([-0.5, 0.0, 0.5]) + logg
    else:
        new_temps = np.arange(-500, 501, 100) + temp
        new_metals = np.arange(-1, 1.1, 0.5) + metals
        new_loggs = np.arange(-1, 1.1, 0.5) + logg
    return new_temps, new_loggs, new_metals


def find_phoenix_model_names(base_dir, ref_model, mode="temp"):
    """Find other phoenix models with similar temp and metallicities.

    Parameters
    ----------
    base_dir: str
        Path to phoenix modes HiResFITS folder.
    ref_model:
       Model to start from and search around.
    mode: str
        Mode to find models, "temp" means all metallicity and logg but
        just limit temperature to +/- 400 K, "small" - smaller range of
        +/- 1 logg and metallicity. "all" search all.
        "closest", find the closest matches the given parameters.

    Returns
    -------
    phoenix_models: list[str]
       List of filenames for phoenix models that match mode criteria.

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
    t_range = 400  # K
    l_range = 1
    f_range = 1
    teffs = np.concatenate((np.arange(2300, 7000, 100),
                            np.arange(7000, 12100, 200)))
    loggs = np.arange(0, 6.1, 0.5)
    fehs = np.concatenate((np.arange(-4, -2, 1), np.arange(-2, 1.1, 0.5)))
    # alphas = np.arange(-0.2, 0.3, 0.2)  # use only these alpha values if necessary

    ref_model = ref_model.split("/")[-1]  # In case has folders in name
    ref_temp = int(ref_model[4:8])
    ref_logg = float(ref_model[9:13])
    ref_feh = float(ref_model[14:17])

    if mode == "all":
        glob_temps = teffs
        glob_loggs = loggs
        glob_fehs = fehs
    elif mode == "temp":
        glob_temps = teffs[((teffs > (ref_temp - t_range)) & (teffs < (ref_temp + t_range)))]
        glob_loggs = loggs
        glob_fehs = fehs
    elif mode == "small":
        glob_temps = teffs[((teffs > (ref_temp - t_range)) & (teffs < (ref_temp + t_range)))]
        glob_loggs = loggs[((loggs > (ref_logg - l_range)) & (loggs < (ref_logg + l_range)))]
        glob_fehs = fehs[((fehs > (ref_feh - f_range)) & (fehs < (ref_feh + f_range)))]

    file_list = []
    for t_, logg_, feh_ in itertools.product(glob_temps, glob_loggs, glob_fehs):
        phoenix_glob = ("/Z{2:+4.1f}/*{0:05d}-{1:4.2f}{2:+4.1f}.PHOENIX*.fits"
                        "").format(t_, logg_, feh_)
        logging.debug("Phoenix glob = {0}".format(phoenix_glob))
        model_to_find = base_dir + phoenix_glob
        files = glob.glob(model_to_find)
        file_list += files
    logging.debug("file list = {0}".format(file_list))
    phoenix_models = file_list
    # folder_file = ["/".join(f.split("/")[-2:]) for f in phoenix_models]

    return phoenix_models


# def find_phoenix_model_names2(base_dir: str, original_model: str) -> List[str]:    # mypy
def find_phoenix_model_names2(base_dir, original_model):
    """Find other phoenix models with similar temp and metallicities.

    Returns list of model name strings.

    """
    # "lte05200-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    try:
        model_name = os.path.split(original_model)[-1]
    except Exception:
        model_name = original_model
    logging.debug("original_name = {}".format(original_model))
    logging.debug("model_name = {}".format(model_name))
    temp = int(model_name[3:8])
    logg = float(model_name[9:13])
    metals = float(model_name[13:17])

    new_temps = np.arange(-500, 501, 100) + temp
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

        if "+0.0" in name:  # Positive zero is not allowed in naming
            name = name.replace("+0.0", "-0.0")

        if os.path.isfile(name):
            close_models.append(name)
    return close_models
