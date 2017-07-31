import sys
import argparse


def _parser():
    """Take care of all the argparse stuff.

    :returns: the args
    """
    # parser = GooeyParser(description='Wavelength Calibrate CRIRES Spectra')
    parser = argparse.ArgumentParser(description='Wavelength Calibrate CRIRES \
                                    Spectra')
    parser.add_argument('star', help='Star name file')
    parser.add_argument('-n', '--obs_num', help='Obervation number')
    parser.add_argument('-d', '--detector', default=None, help='detector number, All if not provided.')  # if False/nune the [1,2,3,4]
    parser.add_argument('-o', '--output', default=False, help='Ouput Filename')
    parser.add_argument('-m', '--model', choices=["tcm", "bhm"],
                        help='Choose model to evaulate, ["tcm"(default), "bhm"]')
    parser.add_argument('-m', '--mode', choices=["chi2", "plot"],
                        help='Calcualte chi2 or plot results.')

    args = parser.parse_args()
    return args


def main(star, obs_num, detector, output=None, model="tcm", mode="plot"):
    if output is None:
        output = "Analysis-{0}-{1}_{2}-{}_chisqr_results.dat".format(star, obs_num, detector, model)

    if mode == "plot":
        # Load chi2 and dot he plotting
        pass
    elif mode == "chi2":
        # Do the chi2 calcualtions and save to a file.
        pass


if __name__ == '__main__':
    args = vars(_parser())
    opts = {k: args[k] for k in args}

    sys.exit(main(**opts))
