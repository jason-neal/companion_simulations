
# New Version of alpha_detection using Parallel and methodolgy from grid_chisquare.

 

def spectrum_chisqr(spectrum_1, spectrum_2, error=None):
    """ Chi squared for specturm objects. """
    # Spectrum wrapper for chissquare
    # make sure xaxis is the Same
    # if len(spectrum_1) == len(spectrum_2):
    if np.all(spectrum_1.xaxis == spectrum_2.xaxis):
        # print("xaxis are equal")
        c2 = chi_squared(spectrum_1.flux, spectrum_2.flux, error=error)
        # return chi_squared(spectrum_1.flux, spectrum_2.flux, error=None)
        # print("chisqrayured value", c2)
        # plot_spectrum(spectrum_1, label="obs", show=False)
        # plot_spectrum(spectrum_2, label="evauated")
        if np.isnan(c2):
            print(" Nan chisqr")
            # print(spectrum_1.xaxis, spectrum_1.flux, spectrum_2.xaxis, spectrum_2.flux)
        return c2
    else:

        #print(len(spectrum_1), len(spectrum_2))
        raise Exception("TODO: make xaxis equal in chisquare of spectrum")

def model_chisqr_wrapper(spectrum_1, model, params, error=None):
    """ Evaluate model and call chisquare """
    # print("params for model", params)
    # params = copy.copy(params)
    evaluated_model = model(*params)  # # unpack parameters

    return spectrum_chisqr(spectrum_1, evaluated_model, error=error)

# @memory.cache
def parallel_chisqr(iter1, iter2, observation, model_func, model_params, numProcs=1):

    grid = Parallel(n_jobs=numProcs)(delayed(model_chisqr_wrapper)(observation,
                                     model_func, (a, b, *model_params))
                                     for a in iter1 for b in iter2)
    return np.asarray(grid)


def alpha_model(alpha, rv, host, companion, limits, new_x=None):
    """ Entangled spectrum model.
    inputs:
    spectrum_1
    spectrum_2
    alpha
    rv - rv offset of spec2
    xrange = location of points to return for spectrum. e.g. observation.xaxis.
    should find better name.

    returns:
    Spectrum object
    """
    # this copy solved my nan issue.
    companion = copy.copy(companion)
    host = copy.copy(host)

    companion.doppler_shift(rv)
    combined = combine_spectra(host, companion, alpha)

    if new_x:
        combined.spline_interpolate_to(new_x)
    combined.wav_select(limits[0], limits[1])
    # observation.wav_select(2100, 2200)

    return combined

def plot_spectrum(spectrum, label=False, show=True):
    """Plot a spectrum object"""
    if label:
        plt.plot(spectrum.xaxis, spectrum.flux, label=label)
        plt.legend()
    else:
        plt.plot(spectrum.xaxis, spectrum.flux)

    plt.ylabel("Flux")
    plt.xlabel("xaxis")

    if show:
        plt.show()

