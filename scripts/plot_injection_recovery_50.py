import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from styler import styler

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

error = None
home = "/home/jneal/Phd/Analysis/injection/all_rv100_dontnorm_repeated/"
fname_template = "HD211847_synth_injector_results_logg={}_error=None_chip_[1, 2, 3]_rv2_100_{}.txt"

colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
markers = [".", "^", "o", "*", ">", "<", "h"]


@styler
def f(fig, *args, **kwargs):
    # plot the injection-recovery
    star = kwargs.get("star")
    obs = kwargs.get("obsnum")
    error = kwargs.get("error")
    suffix = kwargs.get("suffix", "")
    comp_logg = kwargs.get("comp_logg", 4.5)
    ms = kwargs.get("ms", 2)
    lw = kwargs.get("lw", 1)
    ax1 = plt.subplot(111)

    for ii in range(1, 51):
        print(ii)
        fname = fname_template.format(comp_logg, ii)
        input_, output, rv1, rv2 = np.loadtxt(fname, skiprows=6, unpack=True)
        temp_err = 100 * np.ones_like(input_)
        output[output >= 5000] = np.nan
        assert len(input_) == len(temp_err)
        ax1.errorbar(input_, output, yerr=temp_err, marker=markers[ii % 7], color=colors[ii % 8], ms=ms, lw=lw, ls="",
                     label="num {0}".format(ii))

    ax1.plot(input_, input_, "k--", alpha=0.7, lw=lw)
    plt.xlabel("Injected Temperature (K)")
    ax1.set_ylabel("Recovered Temperature (km/s)")
    ax1.set_xlim(2450, 5050)
    # ax1.legend(title="Number")

    plt.suptitle("Synethic recovery repetion")
    plt.tight_layout()

    # plt.show()


def parse_args(args):
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Chisquare analysis.')
    parser.add_argument('star', help='Star Name')
    parser.add_argument('obsnum', help="Observation label")
    parser.add_argument('-s', '--suffix', default="",
                        help='Suffix to add to database name.')
    parser.add_argument('--error', default=None,
                        help='Error value added.')
    return parser.parse_args(args)


if __name__ == "__main__":
    args = vars(parse_args(sys.argv[1:]))
    opts = {k: args[k] for k in args}
    starname = opts.get("star")
    obsnum = opts.get("obsnum")
    suffix = opts.get("suffix", "")
    error = opts.get("error", None)
    for comp_logg in (4.5, 5.0):
        f(type="one", tight=True, dpi=400, figsize=(None, .70),
          axislw=0.5, ms=1.8, lw=0.7,
          save="inject_recovery{0}-{1}_error={2}_{3}_multirun.pdf".format(starname, obsnum, suffix, error),
          formatcbar=False, formatx=False, formaty=False, **opts, comp_logg=comp_logg)
    print("Done")
