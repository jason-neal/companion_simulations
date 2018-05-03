"""HD211847 Example companion of sun like star"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from mingle.utilities.db_utils import SingleSimReader, DBExtractor
from styler import styler

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

chi2_val = "chi2_123"

dir_base = "/home/jneal/Phd/Analysis/injection/INJECTORSIMS_analysis"
dir_base = "/home/jneal/Phd/Analysis/injection/INJECTORSIMS_analysis"
dir_base = "/home/jneal/Desktop/Inbox/b/testing_injector_results_50kms_newspectra/analysis"
dir_base = "/home/jneal/Desktop/Inbox/b/testing_injector_results_20kms/analysis"
#dir_base = "/home/jneal/Desktop/Inbox/b/testing_injector_results_newspectra/analysis"

correct9 = None

ms = 10


# grid = True


@styler
def f(fig, *args, **kwargs):
    name = kwargs.get("name")
    comp_temp = kwargs.get("comp_temp", False)
    grid = kwargs.get("grid", False)
    print("comp_temp", comp_temp)
    injected_point = {"teff_1": 5800, "logg_2": 4.5, "feh_1": 0.0, "gamma": 0,
                      "teff_2": comp_temp, "logg_2": 5.0, "feh_2": 0.0, "rv": 100, "obsnum": 1}

    kwargs = {"correct": injected_point, "dof": 4, "grid": grid, "ms": ms, "sigma": [3]}

    ##################
    # ### IAM RESULTS
    for comp_temp in [3000, 3500, 3800, 4000, 4500]:
        name = f"INJECTORSIMS{comp_temp}"
        sim_example = SingleSimReader(base=dir_base, name=name, mode="iam", suffix="*", chi2_val=chi2_val)

        extractor = DBExtractor(sim_example.get_table())

        df_min = extractor.minimum_value_of(chi2_val)

        df = extractor.simple_extraction(columns=["teff_2", chi2_val])

        chi2s = []
        teffs = []
        for teff_2 in df["teff_2"]:
            teffs.append(teff_2)
            chi2s.append(np.min(df[df["teff_2"] == teff_2][chi2_val]))

        # plt.subplot(211)
        # plt.plot(df["teff_2"], df["chi2_123"], ".", label=comp_temp)
        plt.plot(teffs, chi2s, "-.", label=comp_temp, lw=1)
        plt.xlabel("Recovered Temp (K)")
        plt.ylabel("$\chi^2$")
        plt.legend(title="Injected Temp", loc="outside_right", fontsize="small")


if __name__ == "__main__":
    f(type="one", tight=True, dpi=500, save="injector_temp_investigation_new_20.pdf", figsize=(None, .70),
      axislw=0.5, formatcbar=False, formatx=False, formaty=False, grid=True)
print("Done")
