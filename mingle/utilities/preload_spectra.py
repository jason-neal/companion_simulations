from itertools import product

from mingle.utilities.phoenix_utils import load_starfish_spectrum


def make_dict():
    spectra_dict = {}
    for i, (t, l) in enumerate(product(range(2300, 6000, 100), [4.5, 5.0])):
        print(t, l / 10)
        spec = load_starfish_spectrum([t, l / 10, 0.0], limits=[2110, 2165], wav_scale=True,
                                      hdr=True, area_scale=True, normalize=False, flux_rescale=True)

        spectra_dict[(t, l / 10)] = spec
    return spectra_dict


spectra_dict = make_dict()

len(spectra_dict)


def read_dict(t, l):
    import time
    time.sleep(0.1)
    print(spectra_dict[(t, l / 10)])


for i, (t, l) in enumerate(product(range(2300, 5100, 100), range(45, 51, 5))):
    read_dict(t, l)

print("hello")
