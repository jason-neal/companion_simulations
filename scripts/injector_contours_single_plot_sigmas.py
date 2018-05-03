"""HD211847 Example companion of sun like star"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from mingle.utilities.db_utils import SingleSimReader, DBExtractor
from styler import styler

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

chi2_val = "chi2_123"

dir_bases = ["/home/jneal/Phd/Analysis/injection/INJECTORSIMS_analysis",
             "/home/jneal/Phd/Analysis/injection/injection_shape_167665/analysis",
             "/home/jneal/Phd/Analysis/injection/injection_shape_211847/analysis",
             "/home/jneal/Phd/Analysis/injection/injection_shape_30501/analysis"]

ms = 10


@styler
def g(fig, *args, **kwargs):
    # height_ratios = kwargs.get("height_ratios", (1))
    dir_base = kwargs.get("dir_base")
    lw = kwargs.get("lw", 0.8)

    ax1 = fig.subplots(1, 1)

    # for comp_temp in [2500, 3000, 3500, 3800, 4000, 4500]:
    temps = np.arange(2300, 4501, 100)
    for comp_temp in temps:
        try:
            name = f"INJECTORSIMS{comp_temp}"
            print(name)
            sim_example = SingleSimReader(base=dir_base, name=name, mode="iam", suffix="*", chi2_val=chi2_val)
            extractor = DBExtractor(sim_example.get_table())

            df = extractor.simple_extraction(columns=["teff_2", chi2_val])
            chi2s = []
            teffs = []
            for teff_2 in df["teff_2"]:
                teffs.append(teff_2)
                chi2s.append(np.min(df[df["teff_2"] == teff_2][chi2_val]))

            ax1.errorbar(comp_temp, teffs[np.argmin(chi2s)], 100, fmt=".", label=comp_temp, lw=lw)
        except:
            print("Failed with temp-", comp_temp)
            pass
    ax1.set_ylabel("Recovered Temp (K)")
    # ax1.set_ylim(bottom=-50, top=5000)
    # ax1.set_xlim(right=5200)
    ax1.set_xlabel("Injected Temp (K)")
    plt.plot(temps, temps, "k--", lw=lw)
    plt.tight_layout()


@styler
def f(fig, *args, **kwargs):
    height_ratios = kwargs.get("height_ratios", (3, 1))
    dir_base = kwargs.get("dir_base")
    lw = kwargs.get("lw", 0.8)

    ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": height_ratios})

    for comp_temp in [2500, 3000, 3500, 3800, 4000, 4500, 5000]:
        name = f"INJECTORSIMS{comp_temp}"
        print(name)
        sim_example = SingleSimReader(base=dir_base, name=name, mode="iam", suffix="*", chi2_val=chi2_val)
        extractor = DBExtractor(sim_example.get_table())

        df = extractor.simple_extraction(columns=["teff_2", chi2_val])
        chi2s = []
        teffs = []
        for teff_2 in df["teff_2"]:
            teffs.append(teff_2)
            chi2s.append(np.min(df[df["teff_2"] == teff_2][chi2_val]))

        ax1.plot(teffs, chi2s - min(chi2s), "-.", label=comp_temp, lw=lw)

        ax2.axvline(teffs[np.argmin(chi2s)], lw=lw, color="grey", alpha=0.9, linestyle="--")
        ax2.plot(teffs, chi2s - min(chi2s), "-.", label=comp_temp, lw=lw)

    ax1.set_ylabel("$\Delta \chi^2$")
    ax1.set_ylim(bottom=-50, top=5000)
    ax1.set_xlim(right=5200)
    ax1.legend(title="Injected", fontsize="small")

    from mingle.utilities.chisqr import chi2_at_sigma
    sigma1 = chi2_at_sigma(1, dof=2)
    sigma2 = chi2_at_sigma(2, dof=2)
    sigma3 = chi2_at_sigma(3, dof=2)
    ax2.axhline(sigma2, linestyle="-", alpha=0.5, color="k", label="$2-\sigma$", lw=lw - 0.2)
    ax2.axhline(sigma1, linestyle="-", alpha=0.5, color="k", label="$1-\sigma$", lw=lw - 0.2)
    ax2.axhline(sigma3, linestyle="-", alpha=0.5, color="k", label="$3-\sigma$", lw=lw - 0.2)
    ax2.set_ylim(bottom=0, top=15)
    ax2.set_xlabel("Model Companion Temperature (K)")
    ax2.set_ylabel("$\Delta \chi^2$")

    ax1.axvline(2300, color="k", lw=lw)
    ax2.axvline(2300, color="k", lw=lw)
    plt.tight_layout()


if __name__ == "__main__":
    for dir in dir_bases:
        try:
            f(type="one", tight=True, dpi=500, save=os.path.join(dir, "../chi2_shape_sigmas.pdf"), figsize=(None, .80),
              axislw=0.5, formatcbar=False, formatx=False, formaty=False, lw=0.8, dir_base=dir)
        except:
            pass
        try:

            g(type="one", tight=True, dpi=500, save=os.path.join(dir, "../temperature_cutoff.pdf"), figsize=(None, .80),
              axislw=0.5, formatcbar=False, formatx=False, formaty=False, lw=0.8, dir_base=dir)
        except:
            pass
        print("passing in g")
print("Done")
