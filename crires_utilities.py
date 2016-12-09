#!/usr/bin/env python

# barycorrect_crires.py
# A script/module to barycentric correct a crires observation
# Barycentric correction of CRIRES spectrum given the header information

import ephem
from PyAstronomy import pyasl

# TODO: Add a line in the header to check if this script has already been
# applied.
# TODO: wrapper to handle Specturm objects


def barycorr_crires_spectrum(spectrum, extra_offset=None):
    """ Wrapper to apply barycorr for crires spectra if given a Spectrum object
    """
    from spectrum_overload.Spectrum import Spectrum
    nflux, wlprime = barycorr_crires(spectrum.xaxis, spectrum.flux,
                                     spectrum.header, extra_offset=extra_offset)
    new_spectrum = Spectrum(flux=nflux, xaxis=spectrum.xaxis, header=spectrum.header)
    return new_spectrum


def barycorr_crires(wavelength, flux, header, extra_offset=None):
    # ""
    # Calculate Heliocenteric correction values and apply to spectrum.

    # SHOULD test again with bary and see what the  difference is.
    # """

    longitude = float(header["HIERARCH ESO TEL GEOLON"])
    latitude = float(header["HIERARCH ESO TEL GEOLAT"])
    altitude = float(header["HIERARCH ESO TEL GEOELEV"])

    ra = header["RA"]    # CRIRES RA already in degrees
    dec = header["DEC"]  # CRIRES hdr DEC already in degrees

    # Pyastronomy helcorr needs the time of observation in julian Days
    # #############################################
    Time = header["DATE-OBS"]    # Observing date  '2012-08-02T08:47:30.8425'

    # Convert easily to julian date with ephem
    jd_ephem = ephem.julian_date(Time.replace("T"," ").split(".")[0])

    # Calculate helocentric velocity
    helcorr = pyasl.helcorr(longitude, latitude, altitude, ra, dec, jd,
                            debug=False)

    if extra_offset:
        print("Warning!!!! have included a manual offset for testing")
        helcorr_val = helcorr[0] + extra_offset
    else:
        helcorr_val = helcorr[0]
    # Apply dooplershift to the target spectra with helcorr correction velocity
    nflux, wlprime = pyasl.dopplerShift(wavelength, flux, helcorr_val,
                                        edgeHandling=None, fillValue=None)

    print(" RV size of heliocenter correction for spectra", helcorr_val)
    return nflux, wlprime


def crires_resolution(header):
    """ Set CRIRES resolution based on rule of thumb equation from the manual.
    Warning! The use of adpative optics is not checked for!!
    # This code has been copied from tapas xml request script.
    """
    instrument = header["INSTRUME"]

    slit_width = header["HIERARCH ESO INS SLIT1 WID"]
    if "CRIRES" in instrument:
        # print("Resolving Power\nUsing the rule of thumb equation from the
        # CRIRES manual. \nWarning! The use of adpative optics is not
        # checked for!!")
        R = 100000 * 0.2 / slit_width
        resolving_power = int(R)
        # print("Slit width was {0} inches.\n
        # Therefore the resolving_power is set = {1}".format(slit_width,
        # resolving_power))
    else:
        print("Instrument is not CRIRES")
    return resolving_power
