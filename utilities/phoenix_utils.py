""" Phoenix Utilities.

Some functions to deal with phoenix models
i.e. searching for models with certian parameters

Jason Neal, Janurary 2017
"""
import glob
import numpy as np
import itertools


def find_closest_phoenix(data_dir, teff, logg, feh, alpha=None):
    """ Find the closest PHOENIX-ACES model to the stellar parameters given.

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
        raise NotImplemented("Alpha not implemented")

    teffs = np.concatenate((np.arange(2300, 7000, 100),
                            np.arange(7000, 12100, 200)))
    loggs = np.arange(0, 6.1, 0.5)
    fehs = np.concatenate((np.arange(-4, -2, 1), np.arange(-2, 1.1, 0.5)))
    alphas = np.arange(-0.2, 0.3, 0.2)  # use only these alpha values if nesessary

    closest_teff = teffs[np.abs(teffs - teff).argmin()]
    closest_logg = loggs[np.abs(loggs - logg).argmin()]
    closest_feh = fehs[np.abs(fehs - feh).argmin()]

    if alpha is not None:
        if abs(alpha) > 0.2:
            print("Warning! Alpha is outside acceptable range -0.2->0.2")
        closest_alpha = alphas[np.abs(alphas - alpha).argmin()]
        phoenix_glob = ("/Z{2:+4.1f}.Alpha={3:+5.2f}/*{0:05d}{1:4.2f}"
                        "{2:+4.1f}.Alpha={3:+5.2f}.PHOENIX*.fits"
                        "").format(closest_teff, closest_logg, closest_feh,
                                   closest_alpha)
    else:
        phoenix_glob = ("/Z{2:+4.1f}/*{0:05d}{1:4.2f}{2:+4.1f}.PHOENIX*.fits"
                        "").format(closest_teff, closest_logg, closest_feh)
    files = glob.glob(data_dir + phoenix_glob)
    if len(files) > 1:
        print("More than one file returned")
    return files


def find_phoenix_models(base_dir, ref_model, mode="temp"):
    """ Find other phoenix models with similar temp and metalicities.

    Parameters
    ----------
    base_dir: str
        Path to phoenix modes HiResFITS folder.
    ref_model:
       Model to start from and search around.
    mode: str
        Mode to find models, "temp" means all metalicity and logg but
        just limit temperature to +/- 400 K, "small" - smaller range of
        +/- 1 logg and metalicity. "all" search all.
        "closest", find the closest matches the given parameters.

    Returns
    -------
    phoenix_models: list[str]
       List of filenames of phoenix models that match mode criteria.
    """
    # "lte05200-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"

    # Phoenix parameters
    # Parameter   	Range	 Step size
    # Teff [K]	 2300 - 7000	100
    # 	        7000 - 12000	200
    # log(g)	   0.0 - 6.0	0.5
    # [Fe/H]	 -4.0 - -2.0	1.0
    # 	         -2.0 - +1.0	0.5
    # [α/M]	     -0.2 - +1.2	0.2
    teffs = np.concatenate((np.arange(2300, 7000, 100),
                            np.arange(7000, 12100, 200)))
    loggs = np.arange(0, 6.1, 0.5)
    fehs = np.concatenate((np.arange(-4, -2, 1), np.arange(-2, 1.1, 0.5)))
    # alphas = np.arange(-0.2, 0.3, 0.2)  # use only these alpha values if nesessary

    ref_model = ref_model.split("/")[-1]  # Incase has folders in name
    ref_temp = int(ref_model[4:8])
    ref_logg = float(ref_model[9:13])
    ref_feh = float(ref_model[14:17])

    if mode == "all":
        glob_temps = teffs
        glob_loggs = loggs
        glob_fehs = fehs
    elif mode == "temp":
        glob_temps = teffs[((teffs > (ref_temp - 400)) & (teffs < (ref_temp + 400)))]
        glob_loggs = loggs
        glob_fehs = fehs
    elif mode == "small":
        glob_temps = teffs[((teffs > (ref_temp - 400)) & (teffs < (ref_temp + 400)))]
        glob_loggs = loggs[((loggs > (ref_logg - 1)) & (loggs < (ref_logg + 1)))]
        glob_fehs = fehs[((fehs > (ref_feh - 1)) & (fehs < (ref_feh + 1)))]

    file_list = []
    for t_, logg_, feh_ in itertools.product(glob_temps, glob_loggs, glob_fehs):
        phoenix_glob = ("/Z{2:+4.1f}/*{0:05d}-{1:4.2f}{2:+4.1f}.PHOENIX*.fits"
                        "").format(t_, logg_, feh_)
        print(phoenix_glob)
        model_to_find = base_dir + phoenix_glob
        files = glob.glob(model_to_find)
        file_list += files
    print("file list", file_list)
    phoenix_models = file_list
    # folder_file = ["/".join(f.split("/")[-2:]) for f in phoenix_models]

    return phoenix_models
