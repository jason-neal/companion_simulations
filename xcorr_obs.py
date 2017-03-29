# from utitlies.phoenix_utils import find_closest_phoenix
import os
import glob
import copy
import itertools
import numpy as np
from tqdm import tqdm
from joblib import Memory
from astropy.io import fits
# from PyAstronomy import pyasl
import matplotlib.pyplot as plt
from collections import defaultdict

from IP_multi_Convolution import IPconvolution
from Chisqr_of_observation import load_spectrum
from spectrum_overload.Spectrum import Spectrum
from utilities.crires_utilities import crires_resolution
from utilities.phoenix_utils import spec_local_norm
from alpha_detection_limit_multiprocess import apply_convolution

cachedir = "/home/jneal/.simulation_cache"
if not os.path.exists(cachedir):
    os.makedirs(cachedir)
memory = Memory(cachedir=cachedir, verbose=0)


# find_closest_phoenix(data_dir, teff, logg, feh, alpha=None)
model_base_dir = "/home/jneal/Phd/data/fullphoenix/phoenix.astro.physik.uni-goettingen.de/HiResFITS/"
phoenix_path = ("/home/jneal/Phd/data/fullphoenix/"
                "phoenix.astro.physik.uni-goettingen.de" "/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/")


def generate_phoenix_files(data_dir, logg=4.5, feh=-0):
    # temps = range(2300, 13000, 100)
    temps = range(4000, 9000, 100)
    for temp in temps:
        phoenix_glob = ("/Z{2:+4.1f}/*{0:05d}-{1:4.2f}{2:+4.1f}.PHOENIX*.fits"
                        "").format(temp, logg, feh)
        if feh == 0:
            phoenix_glob = phoenix_glob.replace("+0.0", "-0.0")
        phoenix_name = glob.glob(data_dir + phoenix_glob)
        if len(phoenix_name) != 0:
            yield phoenix_name[0]
        else:
            continue


star = "HD30501"
chips = range(1, 5)
obs_nums = ("1", "2a", "2b", "3")
target_rv = defaultdict(dict)
target_cc = defaultdict(dict)
for chip, obs_num in itertools.product(chips, obs_nums):
    phoenix_names = generate_phoenix_files(phoenix_path)

    # obs_num = 3
    # chip = 1
    # obs_name = select_observation(star, obs_num, chip)
    obs_name = glob.glob("/home/jneal/Phd/data/Crires/BDs-DRACS/{}-"
                         "{}/Combined_Nods/CRIRE.*_{}.nod.ms.*wavecal.tellcorr.fits".format(star, obs_num, chip))[0]

    print("obs_name", obs_name)

    # Load observation
    observed_spectra = load_spectrum(obs_name)
    obs_resolution = crires_resolution(observed_spectra.header)

    wav_model = fits.getdata(model_base_dir + "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
    wav_model /= 10   # turn into nm

    temp_store = []
    rv_store = []
    cc_store = []

    for model in tqdm(phoenix_names):
        temp_store.append(int(model.split("/")[-1][3:8]))
        phoenix_data = fits.getdata(model)
        phoenix_spectrum = Spectrum(flux=phoenix_data, xaxis=wav_model, calibrated=True)

        phoenix_spectrum.wav_select(2080, 2220)
        phoenix_spectrum = spec_local_norm(phoenix_spectrum)
        phoenix_spectrum = apply_convolution(phoenix_spectrum, R=obs_resolution, chip_limits=[2080, 2220])
        rv, cc = observed_spectra.crosscorrRV(phoenix_spectrum, rvmin=-100., rvmax=100.0, drv=0.1,
                                              mode='doppler', skipedge=50)

        maxind = np.argmax(cc)
        # print("Cross-correlation function is maximized at dRV = ", rv[maxind], " km/s")

        rv_store.append(rv[maxind])
        cc_store.append(cc[maxind])

        # ADD CHISQUARED Value


    plt.subplot(211)
    plt.plot(temp_store, rv_store, label="{} {}".format(chip, obs_num))
    plt.title("RV value.")
    plt.xlabel("Temperature")
    plt.subplot(212)
    plt.plot(temp_store, cc_store, label="{} {}".format(chip, obs_num))
    plt.title("Cross-correlation value.")
    plt.xlabel("Temperature")
    plt.ylabel("Cross-Correlation value")
    plt.legend()

    # Store results
    target_rv[obs_num][chip] = [rv_store]
    target_cc[obs_num][chip] = [cc_store]


plt.show()
