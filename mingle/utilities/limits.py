import logging


def get_phoenix_limits(limits="phoenix"):
    """Return synthetic library limits for each parameter"""
    if limits == "phoenix":
        phoenix_limits = [[2300, 12000], [0, 6], [-4, 1]]
    elif limits == "cifist":
        phoenix_limits = [[1200, 7000], [2.5, 5], [0, 0]]
    else:
        raise ValueError("Error with phoenix limits. Invalid limits name '{0}'".format(limits))
    return phoenix_limits


def set_model_limits(temps, loggs, metals, limits):
    """Apply limits to list of models

    limits format = [[temp1, temp2][log-1, logg2][feh_1, feh_2]
    """
    new_temps = temps[(temps >= limits[0][0]) * (temps <= limits[0][1])]
    new_loggs = loggs[(loggs >= limits[1][0]) * (loggs <= limits[1][1])]
    new_metals = metals[(metals >= limits[2][0]) * (metals <= limits[2][1])]

    if len(temps) > len(new_temps) | len(loggs) > len(new_loggs) | len(metals) > len(new_metals):
        logging.warning("Some models were removed using the 'parrange' limits.")
    return new_temps, new_loggs, new_metals
