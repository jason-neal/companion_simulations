import numpy as np
from joblib import Parallel, delayed
from mingle.utilities import spectrum_chisqr
from spectrum_overload import Spectrum


def model_chisqr_wrapper(spectrum_1: Spectrum, model, params, error=None):
    """Evaluate model and call chisquare."""
    evaluated_model = model(*params)  # unpack parameters

    if np.all(np.isnan(evaluated_model.flux)):
        raise Exception("Evaluated model is all Nans")

    return spectrum_chisqr(spectrum_1, evaluated_model, error=error)


def parallel_chisqr(iter1, iter2, observation, model_func, model_params, n_jobs=1):
    """Parallel chisquared calculation with two iterators."""
    grid = Parallel(n_jobs=n_jobs)(delayed(model_chisqr_wrapper)(observation,
                                                                 model_func, (a, b, *model_params))
                                   for a in iter1 for b in iter2)
    return np.asarray(grid)