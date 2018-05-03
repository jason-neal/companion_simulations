import matplotlib.pyplot as plt
import numpy as np
# Load in the injection recovery files and plot
from matplotlib import rc

from styler import styler

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

colors = ["C0", "C1", "C2", "C3", "C4"]
markers = [".", "^", "o", "*", ">"]


@styler
def f(fig, *args, **kwargs):
    # plot the injection-recovery
    show_mass = kwargs.get("show_mass", False)
    fname = kwargs.get("fname")
    type_name = kwargs.get("type_name")
    ms = kwargs.get("ms", 2)
    if "real" in type_name:
        xi = 0
    else:
        xi = 1
    lw = kwargs.get("lw", 1)
    # ax1 = plt.subplot(211)
    ax1 = plt.subplot(111)

    input_, output, rv1, rv2 = np.loadtxt(fname, skiprows=6, unpack=True)
    temp_err = 100 * np.ones_like(input_)
    output[output >= 5000] = np.nan
    assert len(input_) == len(temp_err)
    ax1.errorbar(input_, output, yerr=temp_err, marker=markers[xi], color=colors[xi], ms=ms, lw=lw, ls="",
                 label=type_name)
    ax1.plot(input_, input_, "k--", alpha=0.7, lw=lw)
    plt.xlabel("Injected Temperature (K)")
    ax1.set_ylabel("Recovered Temperature (km/s)")
    ax1.set_xlim(2450, 5050)
    prefix = fname[:13]
    plt.title(r"Inector results: {}".format(prefix))

    plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    print("Starting")
    show_mass = False
    import glob

    files = glob.glob("*injector_results_logg=*.txt")
    print("files", files)

    for fname in files:
        try:
            print("Doing", fname)
            plotname = fname.replace(".txt", ".pdf")
            if "real" in fname:
                type_name = "real"
            else:
                type_name = "synthetic"

            f(type="one", fname=fname, type_name=type_name, tight=True, dpi=400, figsize=(None, .70),
              axislw=0.5, ms=1.8, lw=0.7, save=plotname,
              formatcbar=False, formatx=False, formaty=False, show_mass=show_mass)
        except Exception as e:
            print(e)

    print("Done")
