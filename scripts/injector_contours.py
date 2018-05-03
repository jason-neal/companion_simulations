"""HD211847 Example companion of sun like star"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from mingle.utilities.db_utils import SingleSimReader, DBExtractor
from styler import styler

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

chi2_val = "chi2_123"

dir_base = ["/home/jneal/Phd/Analysis/injection/INJECTORSIMS_analysis",
            "/home/jneal/Phd/Analysis/injection/injection_shape_30501/analysis",
            "/home/jneal/Phd/Analysis/injection/injection_shape_211847/analysis",
            "/home/jneal/Phd/Analysis/injection/injection_shape_167665/analysis"]

ms = 10


@styler
def f(fig, *args, **kwargs):
    name = kwargs.get("name")
    comp_temp = kwargs.get("comp_temp", False)
    grid = kwargs.get("grid", False)
    print("comp_temp", comp_temp)
    injected_point = {"teff_1": 5800, "logg_2": 4.5, "feh_1": 0.0, "gamma": 0,
                      "teff_2": comp_temp, "logg_2": 5.0, "feh_2": 0.0, "rv": 100, "obsnum": 1}

    kwargs = {"correct": injected_point, "dof": 4, "grid": grid, "ms": ms, "sigma": [3]}
    plt.subplot(221)

    ##################
    # IAM RESULTS

    sim_example = SingleSimReader(base=dir_base, name=name, mode="iam", suffix="*", chi2_val=chi2_val)

    extractor = DBExtractor(sim_example.get_table())

    df_min = extractor.minimum_value_of(chi2_val)
    print(df_min)

    cols = ['teff_2', 'logg_2', 'feh_2', 'rv', 'gamma',
            chi2_val, 'teff_1', 'logg_1', 'feh_1']

    fixed = {key: df_min[key].values[0] for key in ["logg_1", "logg_2", "feh_1", "feh_2"]}

    df = extractor.fixed_extraction(cols, fixed, limit=-1)

    kwargs.update({"dof": 4})

    plt.subplot(222)
    plt.plot(df["teff_1"], df["chi2_123"], ".")
    plt.xlabel("teff_1")
    plt.ylabel("$\chi^2$")  # df_contour(df, "teff_2", "teff_1", chi2_val, df_min, ["gamma", "rv"], **kwargs)
    plt.annotate("$C^2$", (.01, 0.95), xycoords='axes fraction')

    plt.subplot(223)
    plt.plot(df["teff_2"], df["chi2_123"], ".")
    plt.xlabel("teff_2")
    plt.ylabel("$\chi^2$")  # df_contour(df, "teff_1", "gamma", chi2_val, df_min, ["teff_2", "rv"], **kwargs)
    plt.annotate("$C^2$", (.01, 0.95), xycoords='axes fraction')

    plt.subplot(224)
    plt.plot(df["rv"], df["chi2_123"], ".")
    plt.xlabel("$\chi^2$")
    plt.ylabel("rv")
    # df_contour(df, "teff_2", "rv", chi2_val, df_min, ["gamma", "teff_1"], **kwargs)
    plt.annotate("$C^2$", (.01, 0.95), xycoords='axes fraction')
    print("HD211847 example min values.")
    print(df_min.head())


if __name__ == "__main__":
    # for comp_temp in [2500, 3000, 3500, 3600, 3700, 3800, 3900, 4000, 4500]:
    for dir in dir_base:
        print(dir)
        for comp_temp in np.arange(2300, 4501, 100):
            try:
                name = f"INJECTORSIMS{comp_temp}"

                f(type="two", tight=True, dpi=500, figsize=(None, .70),
                  save= dir + "/../injector_shapes_{}.pdf".format(comp_temp),
                  axislw=0.5, formatx=False, formaty=False, grid=True,
                  formatcbar=False, comp_temp=comp_temp, name=name)
            except:
                print("passing", comp_temp)
                pass
    print("Done")
