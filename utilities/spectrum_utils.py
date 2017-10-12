import logging
import os

from astropy.io import fits
from Get_filenames import get_filenames
from spectrum_overload.Spectrum import Spectrum

debug = logging.debug


def load_spectrum(name, corrected=True):
    """Load in fits file and return as a Spectrum object.

    Parameters
    ----------
    name: str
        Filename of spectrum.
    corrected: bool
        Use telluric corrected spectra. (deprectiated).

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


# I should already have these sorts of functions
def select_observation(star, obs_num, chip):
    """Select the observation to load in.

    inputs:
    star: name of host star target
    obs_num: observation number
    chip: crires detector chip number

    returns:
    crires_name: name of file
    """
    if str(chip) not in "1234":
        print("The Chip is not correct. It needs to be 1,2,3 or 4")
        raise Exception("Chip Error")
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
