import os
import glob
from astropy.io import fits

# Convolve All Phoenix Spectra to 50000.
resolution = 50000

source_path = ("/home/jneal/Phd/data/fullphoenix/phoenix.astro.physik.uni-goettingen.de/"
               "HiResFITS/PHOENIX-ACES-AGSS-COND-2011")
output_path = "/home/jneal/Phd/data/fullphoenix/convolved_R{0:d}k".format(int(resolution / 1000))


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

phoenix_wave = ("/home/jneal/Phd/data/fullphoenix/phoenix.astro.physik.uni-goettingen.de/"
                "HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
wave = fits.getdata(phoenix_wave) / 10  # Convert to nm

for folder in folders:
    # Now have a specific directory with phoenix fits files.
    phoenix_files = glob.glob(os.path.join(folder, "*.fits"))

    for f in phoenix_files:
        print(f)

        spectrum = fits.getdata(f)

        # Wavelenght narrow

        # Convolutions

        # Save result to fits file in new directory.

        new_f = f.replace(".fits", "_R{0:d}k.fits".format(int(resolution / 1000)))
        new_f = new_f.replace(source_path, output_path)
        print(new_f)
