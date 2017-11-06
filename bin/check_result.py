# Make min chi_2 spectral model

def _parser():
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Produce spectrum of results.')
    parser.add_argument('star', help='Star Name')
    parser.add_argument('obsnum', help="Observation label")
    parser.add_argument('teff_1', type=int, default=None,
                        help='Host Temperature')
    parser.add_argument('logg_1', type=float, default=None,
                        help='Host Temperature')
    parser.add_argument('feh_1', type=float, default=None,
                        help='Host Temperature')
    parser.add_argument('teff_2', type=int, help='Host Temperature')
    parser.add_argument('logg_2', type=float, help='Host Temperature')
    parser.add_argument('feh_2', type=float, help='Host Temperature')
    parser.add_argument('gamma', type=float, help='Host Temperature')
    parser.add_argument("rv",, type = float, help = 'Host Temperature')

    parser.add_argument('-v', '--verbose', action="store_true",
                        help='Turn on Verbose.')
    return parser.parse_args()


def main(star, obsnum, teff_1, logg_1, feh_1, teff_2, logg_2, feh_2, gamma, rv):
    # Get observation data

    # Create model with given parameters

    # plot

    return 0


if __name__ == "__main__":
    args = vars(_parser)
    opts = {k: args[k] for k in args}
    sys.exit(main(**opts))
