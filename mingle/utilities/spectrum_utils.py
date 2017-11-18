import logging
import os

from mingle.utilities.io import get_filenames
from astropy.io import fits
from matplotlib import pyplot as plt
from spectrum_overload import Spectrum

debug = logging.debug


def load_spectrum(name, corrected=True):
    """Load in fits file and return as a Spectrum object.

    Parameters
    ----------
    name: str
        Filename of spectrum.
    corrected: bool
        Use telluric corrected spectra. (depreciated).

    Returns
    -------
    spectrum: Spectrum
        Spectra loaded into a Spectrum object.

    """
    data = fits.getdata(name)
    hdr = fits.getheader(name)

    # TODO: log lambda sampling.
    #      see starfish

    # Turn into Spectrum
    xaxis = data["wavelength"]
    try:
        flux = data["flux"]
    except KeyError:
        try:
            flux = data["Corrected_DRACS"]
        except KeyError:
            try:
                flux = data["Extracted_DRACS"]
            except KeyError:
                print("The fits columns are {}".format(data.columns))
                raise

    spectrum = Spectrum(xaxis=xaxis, flux=flux,
                        header=hdr)
    return spectrum


def select_observation(star, obs_num, chip):
    """Select the observation to load in.

    Inputs:
    star: name of host star target
    obs_num: observation number
    chip: Crires detector chip number

    Returns:
    crires_name: name of file
    """
    if str(chip) not in "1234":
        print("The Chip is not correct. It needs to be 1,2,3 or 4")
        raise ValueError("Chip not in 1-4.")

    else:
        # New reduction and calibration
        path = ("/home/jneal/Phd/data/Crires/BDs-DRACS/2017/{}-"
                "{}/Combined_Nods".format(star, obs_num))
        filenames = get_filenames(path, "CRIRE.*wavecal.tellcorr.fits",
                                  "*_{}.nod.ms.*".format(chip))
        debug("Filenames from 2017 reductions {}".format(filenames))
        if len(filenames) is not 0:
            crires_name = filenames[0]
        else:
            path = ("/home/jneal/Phd/data/Crires/BDs-DRACS/{}-"
                    "{}/Combined_Nods".format(star, obs_num))
            print("Path =", path)
            filenames = get_filenames(path, "CRIRE.*wavecal.tellcorr.fits",
                                      "*_{}.nod.ms.*".format(chip))

            crires_name = filenames[0]
        return os.path.join(path, crires_name)


def spectrum_plotter(spectra, label=None, show=False):
    """Plot a Spectrum object."""
    plt.figure()
    plt.plot(spectra.xaxis, spectra.flux, label=label)
    plt.ylabel("Flux")
    plt.xlabel("Wavelength")
    if label:
        plt.legend(loc=0, bbox_to_anchor=(1.4, 0.9), ncol=1,
                   fancybox=True, shadow=True)
    if show:
        plt.show()


def plot_spectra(obs, model):
    """Plot two spectra."""
    plt.plot(obs.xaxis, obs.flux, label="obs")
    plt.plot(model.xaxis, model.flux, label="model")
    plt.legend()
    plt.show()