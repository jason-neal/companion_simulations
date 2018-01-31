#!/usr/bin/env python

# barycorrect_crires.py
# A script/module to barycentric correct a crires observation
# Barycentric correction of CRIRES spectrum given the header information


import logging
import warnings

import ephem
from logutils import BraceMessage as __
from PyAstronomy import pyasl
from spectrum_overload import Spectrum


def barycorr_crires_spectrum(spectrum, extra_offset=None):
    """Wrapper to apply barycorr for CRIRES spectra if given a Spectrum object."""
    if spectrum.header.get("BARYDONE", False):
        warnings.warn("Spectrum already berv corrected")
        if (extra_offset is not None) or (extra_offset != 0):
            warnings.warn("Only applying the extra offset.")
            _, nflux = barycorr_crires(spectrum.xaxis, spectrum.flux,
                                       {}, extra_offset=extra_offset)
        else:
            warnings.warn("Not changing spectrum.")
            return spectrum
    else:
        _, nflux = barycorr_crires(spectrum.xaxis, spectrum.flux,
                                   spectrum.header, extra_offset=extra_offset)

    new_spectrum = Spectrum(flux=nflux, xaxis=spectrum.xaxis, header=spectrum.header)
    new_spectrum.header["BARYDONE"] = True
    return new_spectrum


def barycorr_crires(wavelength, flux, header, extra_offset=None):
    """Calculate Heliocentric correction values and apply to spectrum.

    # SHOULD test again with bary and see what the difference is.
    """
    if header is None:
        logging.warning("No header information to calculate heliocentric correction.")
        header = {}
        if (extra_offset is None) or (extra_offset == 0):
            return wavelength, flux

    try:
        longitude = float(header["HIERARCH ESO TEL GEOLON"])
        latitude = float(header["HIERARCH ESO TEL GEOLAT"])
        altitude = float(header["HIERARCH ESO TEL GEOELEV"])

        ra = header["RA"]  # CRIRES RA already in degrees
        dec = header["DEC"]  # CRIRES hdr DEC already in degrees

        time = header["DATE-OBS"]  # Observing date  '2012-08-02T08:47:30.8425'

        # Convert easily to julian date with ephem
        jd = ephem.julian_date(time.replace("T", " ").split(".")[0])

        # Calculate Heliocentric velocity
        helcorr = pyasl.helcorr(longitude, latitude, altitude, ra, dec, jd,
                                debug=False)
        helcorr = helcorr[0]

        if header.get("BARYDONE", False):
            warnings.warn("Applying barycentric correction when 'BARYDONE' already flag set.")

    except KeyError as e:
        logging.warning("Not a valid header so can't do automatic correction.")

        helcorr = 0.0

    if extra_offset is not None:
        logging.warning("Warning!!!! have included a manual offset for testing")
    else:
        extra_offset = 0.0

    helcorr_val = helcorr + extra_offset

    if helcorr_val == 0:
        logging.warning("Helcorr value was zero")
        return wavelength, flux
    else:
        # Apply Doppler shift to the target spectra with helcorr correction velocity
        nflux, wlprime = pyasl.dopplerShift(wavelength, flux, helcorr_val,
                                            edgeHandling=None, fillValue=None)

        logging.info(__("RV Size of Heliocenter correction for spectra {}", helcorr_val))
        return wlprime, nflux


def crires_resolution(header):
    """Set CRIRES resolution based on rule of thumb equation from the manual.

    resolving_power = 100000 * 0.2 / slit_width

    Warning! The use of adaptive optics is not checked for!!
    # This code has been copied from tapas xml request script.
    """
    instrument = header["INSTRUME"]

    slit_width = header["HIERARCH ESO INS SLIT1 WID"]
    if "CRIRES" in instrument:
        # print("Resolving Power\nUsing the rule of thumb equation from the
        # CRIRES manual. \nWarning! The use of adaptive optics is not
        # checked for!!")
        resolving_power = 100000 * 0.2 / slit_width
        resolving_power = int(resolving_power)
    else:
        print("Instrument is not CRIRES")
    return resolving_power
