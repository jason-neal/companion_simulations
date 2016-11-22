
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

