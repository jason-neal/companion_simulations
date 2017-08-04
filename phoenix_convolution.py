import glob
import os

from astropy.io import fits
from spectrum_overload.Spectrum import Spectrum

# Convolve All Phoenix Spectra to 50000.
resolution = 50000   # 20000, 80000, 100000

source_path = ("/home/jneal/Phd/data/PHOENIX-ALL/PHOENIX/")
output_path = "/home/jneal/Phd/data/fullphoenix/convolved_R{0:d}k".format(int(resolution / 1000))

# Limit phoenix spectra to the K Band "K": (2.07, 2.35) to reduce file sizes and convolution time.
band_limits = [2070, 2350]  # In nanometers


def make_dirs(old_path, new_path):
    """Create a copy of folders in a new directory."""
    old_dirs = glob.glob(os.path.join(old_path, "*"))

    for d in old_dirs:
        folder = os.path.split(d)[-1]
        new_dir = os.path.join(new_path, folder)

        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
    return 0


make_dirs(source_path, output_path)  # This is completed now.

folders = glob.glob(os.path.join(source_path, "*"))

phoenix_wave = ("/home/jneal/Phd/data/PHOENIX-ALL/PHOENIX/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
wave = fits.getdata(phoenix_wave) / 10  # Convert to nm

for folder in folders:
    # Now have a specific directory with phoenix fits files.
    phoenix_files = glob.glob(os.path.join(folder, "*.fits"))

    for f in phoenix_files:
        print(f)
        # Load in file
        spectrum = fits.getdata(f)

        # Wavelenght narrow to K band only 2.07, 2.35 micron
        phoenix_spectrum = Spectrum(flux=spectrum, xaxis=wave)
        phoenix_spectrum.wav_select(*band_limits)

        # Convolutions
        # Convolution in spectrum overload?
        # phoenix_spectrum.convolution(R, ...)

        # Save result to fits file in new directory.

        new_f = f.replace(".fits", "_R{0:d}k.fits".format(int(resolution / 1000)))
        new_f = new_f.replace(source_path, output_path)
        print(new_f)
