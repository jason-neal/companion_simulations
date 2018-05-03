import matplotlib.pyplot as plt
import numpy as np
import simulators
from simulators.iam_module import prepare_iam_model_spectra, inherent_alpha_model
from spectrum_overload.spectrum import Spectrum

area_scale = False
wav_scale = False
# area_scale = True
wav_scale = True

if "CIFIST" in simulators.starfish_grid["hdf5_path"]:
    phoenix = "BT-SETTL"
else:
    phoenix = "ACES"
rv_limits = [(2110, 2125), (2126.5, 2139), (2140, 2154)]
host_temp = 5500
comp_temps = np.arange(3500, 4401, 100)
gammas = [0]
rvs = [50]
obs_spec = np.linspace(2111, 2163, 6072)
last_binary = 0

comp_temps = np.arange(4000, 5301, 100)
for ii, ctemp in enumerate(comp_temps):
    print("ctemp", ctemp)
    mod1_spec, mod2_spec = prepare_iam_model_spectra([host_temp, 4.5, 0], [ctemp, 5.0, 0.0], limits=(2110, 2164),
                                                     area_scale=True, wav_scale=area_scale)
    print("model temp ", mod2_spec.header["PHXTEFF"])
    iam_grid_func = inherent_alpha_model(mod1_spec.xaxis, mod1_spec.flux, mod2_spec.flux,
                                         rvs=rvs, gammas=gammas)
    iam_grid_models = iam_grid_func(obs_spec)
    this_binary = Spectrum(xaxis=obs_spec, flux=iam_grid_models.squeeze())
    if ii > 1:
        diff = this_binary - last_binary
        # diff += (ii * 5e7)
        diff.plot(label="comp {} - {}".format(ctemp, ctemp - 100))

    last_binary = this_binary
for limits in rv_limits:
    plt.axvline(limits[0], color="black", alpha=0.8, ls="--")
    plt.axvline(limits[-1], color="black", alpha=0.8, ls="--")
plt.title(f"1# {phoenix} Binary differences with 4000-4400")
plt.legend()
plt.show()

comp_temps = np.arange(3400, 4401, 100)
for ii, ctemp in enumerate(comp_temps):
    print("ctemp", ctemp)
    mod1_spec, mod2_spec = prepare_iam_model_spectra([host_temp, 4.5, 0], [ctemp, 5.0, 0.0], limits=(2110, 2164),
                                                     area_scale=True, wav_scale=area_scale)
    print("model temp ", mod2_spec.header["PHXTEFF"])
    iam_grid_func = inherent_alpha_model(mod1_spec.xaxis, mod1_spec.flux, mod2_spec.flux,
                                         rvs=rvs, gammas=gammas)
    iam_grid_models = iam_grid_func(obs_spec)
    this_binary = Spectrum(xaxis=obs_spec, flux=iam_grid_models.squeeze())
    if ii > 0:
        diff = this_binary - last_binary
        # diff += (ii * 5e7)

        diff.plot(label="comp {} - {}".format(ctemp, ctemp - 100))

    last_binary = this_binary
for limits in rv_limits:
    plt.axvline(limits[0], color="black", alpha=0.8, ls="--")
    plt.axvline(limits[-1], color="black", alpha=0.8, ls="--")
plt.title(f"#2: {phoenix} Binary differences with Host temp = {host_temp}")
plt.legend()
plt.show()

comp_temps = np.arange(3400, 4401, 100)
for ii, ctemp in enumerate(comp_temps):
    print("ctemp", ctemp)
    _, mod2_spec = prepare_iam_model_spectra([host_temp, 4.5, 0], [ctemp, 5.0, 0.0], limits=(2110, 2164),
                                             area_scale=True, wav_scale=area_scale)
    # mod2_spec += 1e9
    mod2_spec.plot(label=f"{phoenix}-{ctemp}K".format(ctemp))

for limits in rv_limits:
    plt.axvline(limits[0], color="black", alpha=0.8, ls="--")
    plt.axvline(limits[-1], color="black", alpha=0.8, ls="--")
plt.title(f"#3:{phoenix} Binary companion")
plt.legend()
plt.show()

for ii, ctemp in enumerate(np.arange(2300, 3301, 100)):
    print("ctemp", ctemp)
    mod1_spec, mod2_spec = prepare_iam_model_spectra([host_temp, 4.5, 0], [ctemp, 5.0, 0.0], limits=(2110, 2164),
                                                     area_scale=True, wav_scale=area_scale)

    iam_grid_func = inherent_alpha_model(mod1_spec.xaxis, mod1_spec.flux, mod2_spec.flux,
                                         rvs=rvs, gammas=gammas)
    iam_grid_models = iam_grid_func(obs_spec)
    this_binary = Spectrum(xaxis=obs_spec, flux=iam_grid_models.squeeze())
    if ii > 0:
        diff = this_binary - last_binary
        # diff += (ii * 5e6)

        diff.plot(label="comp {} - {}".format(ctemp, ctemp - 100))

    last_binary = this_binary
for limits in rv_limits:
    plt.axvline(limits[0], color="black", alpha=0.8, ls="--")
    plt.axvline(limits[-1], color="black", alpha=0.8, ls="--")
plt.title(f"#4:{phoenix} Binary differences with Host temp = {host_temp}")
plt.legend()
plt.show()

comp_temps = np.arange(3400, 4401, 100)
for ii, ctemp in enumerate(comp_temps):
    print("ctemp", ctemp)
    mod1_spec, mod2_spec = prepare_iam_model_spectra([host_temp, 4.5, 0], [ctemp, 5.0, 0.0], limits=(2110, 2164),
                                                     area_scale=True,
                                                     wav_scale=area_scale)

    iam_grid_func = inherent_alpha_model(mod1_spec.xaxis, mod1_spec.flux, mod2_spec.flux,
                                         rvs=rvs, gammas=gammas)
    iam_grid_models = iam_grid_func(obs_spec)
    this_binary = Spectrum(xaxis=obs_spec, flux=iam_grid_models.squeeze())  # + (ii * 1e9)
    this_binary.plot(label="comp {}".format(ctemp))
for limits in rv_limits:
    plt.axvline(limits[0], color="black", alpha=0.8, ls="--")
    plt.axvline(limits[-1], color="black", alpha=0.8, ls="--")
plt.title(f"#5: {phoenix} Binary with Host temp = {host_temp}")
plt.legend()
plt.show()

comp_temps = np.arange(2300, 7001, 100)
for ii, ctemp in enumerate(comp_temps):
    print("ctemp", ctemp)
    mod1_spec, mod2_spec = prepare_iam_model_spectra([host_temp, 4.5, 0], [ctemp, 5.0, 0.0], limits=(2110, 2164),
                                                     area_scale=True,
                                                     wav_scale=area_scale)

    iam_grid_func = inherent_alpha_model(mod1_spec.xaxis, mod1_spec.flux, mod2_spec.flux,
                                         rvs=rvs, gammas=gammas)
    iam_grid_models = iam_grid_func(obs_spec)
    this_binary = Spectrum(xaxis=obs_spec, flux=iam_grid_models.squeeze())  # + (ii * 1e9)
    this_binary.plot(label="comp {}".format(ctemp))
for limits in rv_limits:
    plt.axvline(limits[0], color="black", alpha=0.8, ls="--")
    plt.axvline(limits[-1], color="black", alpha=0.8, ls="--")
plt.title(f"#6: {phoenix} Binary with Host temp = {host_temp}")
plt.legend()
plt.show()

#######################################################
#  BT-SETTL plots
#######################################################
if phoenix == "BT-SETTL":
    host_temp = 5500
    comp_temps = np.arange(1200, 2301, 100)
    for ii, ctemp in enumerate(comp_temps):
        print("ctemp", ctemp)
        mod1_spec, mod2_spec = prepare_iam_model_spectra([host_temp, 4.5, 0], [ctemp, 5.0, 0.0], limits=(2110, 2164),
                                                         area_scale=True,
                                                         wav_scale=area_scale)
        print("PHXTEFF", mod2_spec.header["PHXTEFF"])
        iam_grid_func = inherent_alpha_model(mod1_spec.xaxis, mod1_spec.flux, mod2_spec.flux,
                                             rvs=rvs, gammas=gammas)
        iam_grid_models = iam_grid_func(obs_spec)
        this_binary = Spectrum(xaxis=obs_spec, flux=iam_grid_models.squeeze())  # + (ii * 1e8)
        this_binary.plot(label="comp {}".format(ctemp))
    for limits in rv_limits:
        plt.axvline(limits[0], color="black", alpha=0.8, ls="--")
        plt.axvline(limits[-1], color="black", alpha=0.8, ls="--")
    plt.title(f"{phoenix} Binary with Host temp = {host_temp}")
    plt.legend()
    plt.show()

    host_temp = 5500
    comp_temps = np.arange(1200, 2301, 100)
    for ii, ctemp in enumerate(comp_temps):
        print("ctemp", ctemp)
        mod1_spec, mod2_spec = prepare_iam_model_spectra([host_temp, 4.5, 0], [ctemp, 5.0, 0.0], limits=(2110, 2164),
                                                         area_scale=area_scale,
                                                         wav_scale=area_scale)
        print("PHXTEFF", mod2_spec.header["PHXTEFF"])
        iam_grid_func = inherent_alpha_model(mod1_spec.xaxis, mod1_spec.flux, mod2_spec.flux,
                                             rvs=rvs, gammas=gammas)
        iam_grid_models = iam_grid_func(obs_spec)
        this_binary = Spectrum(xaxis=obs_spec, flux=iam_grid_models.squeeze())  # + (ii * 1e8)
        this_binary.plot(label="comp {}".format(ctemp))
    for limits in rv_limits:
        plt.axvline(limits[0], color="black", alpha=0.8, ls="--")
        plt.axvline(limits[-1], color="black", alpha=0.8, ls="--")
    plt.title(f"{phoenix} Binary with Host temp = {host_temp}")
    plt.legend()
    plt.show()

    # if phoenix == "BT-SETTL":
    host_temp = 5500
    comp_temps = np.arange(1200, 2301, 100)
    for ii, ctemp in enumerate(comp_temps):
        print("ctemp", ctemp)
        mod1_spec, mod2_spec = prepare_iam_model_spectra([host_temp, 4.5, 0], [ctemp, 5.0, 0.0], limits=(2110, 2164),
                                                         area_scale=area_scale, wav_scale=area_scale)

        iam_grid_func = inherent_alpha_model(mod1_spec.xaxis, mod1_spec.flux, mod2_spec.flux,
                                             rvs=rvs, gammas=gammas)
        iam_grid_models = iam_grid_func(obs_spec)
        this_binary = Spectrum(xaxis=obs_spec, flux=iam_grid_models.squeeze())
        if ii > 0:
            diff = this_binary - last_binary
            # diff += (ii * 0.1e7)
            vert_shift = ii * 0.00001
            # diff += vert_shift

            diff.plot(label="comp {} - {} (+ {:1.0e})".format(ctemp, ctemp - 100, vert_shift))

        last_binary = this_binary
    for limits in rv_limits:
        plt.axvline(limits[0], color="black", alpha=0.8, ls="--")
        plt.axvline(limits[-1], color="black", alpha=0.8, ls="--")
    plt.title(f"{phoenix} Binary differences with Host temp = {host_temp}")
    plt.legend()
    plt.show()

    # if phoenix == "BT-SETTL":
    # host_temp = 5500
    comp_temps = np.arange(1200, 2301, 100)
    plt.subplot(311)
    for ii, ctemp in enumerate(comp_temps):
        print("ctemp", ctemp)
        mod1_spec, mod2_spec = prepare_iam_model_spectra([host_temp, 4.5, 0], [ctemp, 5.0, 0.0], limits=(2110, 2164),
                                                         area_scale=area_scale, wav_scale=area_scale)

        if ii > 0:
            mod2_spec.plot(label=f"comp {ctemp}")

    plt.title(f"{phoenix} companion spectra = {host_temp}")
    plt.legend()

    plt.subplot(312)
    comp_temps = np.arange(3400, 3501, 100)
    for ii, ctemp in enumerate(comp_temps):
        print("ctemp", ctemp)
        mod1_spec, mod2_spec = prepare_iam_model_spectra([host_temp, 4.5, 0], [ctemp, 5.0, 0.0], limits=(2110, 2164),
                                                         area_scale=area_scale, wav_scale=area_scale)

        if ii > 0:
            mod2_spec.plot(label=f"comp {ctemp}")
    plt.title(f"{phoenix} companion spectra = {host_temp}")
    plt.legend()

    plt.subplot(313)
    comp_temps = np.arange(3600, 5001, 100)
    for ii, ctemp in enumerate(comp_temps):
        print("ctemp", ctemp)
        mod1_spec, mod2_spec = prepare_iam_model_spectra([host_temp, 4.5, 0], [ctemp, 5.0, 0.0], limits=(2110, 2164),
                                                         area_scale=area_scale, wav_scale=area_scale)

        if ii > 0:
            mod2_spec.plot(label=f"comp {ctemp}")
    for limits in rv_limits:
        plt.axvline(limits[0], color="black", alpha=0.8, ls="--")
        plt.axvline(limits[-1], color="black", alpha=0.8, ls="--")
        plt.title(f"{phoenix} companion spectra = {host_temp}")
    plt.legend()
    plt.show()

    # if phoenix == "BT-SETTL":
    # host_temp = 5500
    comp_temps = np.arange(4000, 5001, 100)
    for ii, ctemp in enumerate(comp_temps):
        print("ctemp", ctemp)
        mod1_spec, mod2_spec = prepare_iam_model_spectra([host_temp, 4.5, 0], [ctemp, 5.0, 0.0], limits=(2110, 2164),
                                                         area_scale=area_scale, wav_scale=area_scale)

        if ii > 0:
            mod2_spec.plot(label=f"comp {ctemp}")
    plt.title(f"{phoenix} large companions")
    plt.legend()
    plt.show()

    # if phoenix == "BT-SETTL":
    # host_temp = 5500
    comp_temps = np.arange(3000, 4001, 100)
    for ii, ctemp in enumerate(comp_temps):
        print("ctemp", ctemp)
        mod1_spec, mod2_spec = prepare_iam_model_spectra([host_temp, 4.5, 0], [ctemp, 5.0, 0.0], limits=(2110, 2164),
                                                         area_scale=area_scale, wav_scale=area_scale)

        if ii > 0:
            mod2_spec.plot(label=f"comp {ctemp}")
    for limits in rv_limits:
        plt.axvline(limits[0], color="black", alpha=0.8, ls="--")
        plt.axvline(limits[-1], color="black", alpha=0.8, ls="--")
    plt.title(f"{phoenix} large companions 4000-5000K")
    plt.legend()
    plt.show()

    # if phoenix == "BT-SETTL":
    # host_temp = 5500
    comp_temps = np.arange(3800, 4201, 100)
    for ii, ctemp in enumerate(comp_temps):
        print("ctemp", ctemp)
        mod1_spec, mod2_spec = prepare_iam_model_spectra([host_temp, 4.5, 0], [ctemp, 5.0, 0.0],
                                                         limits=(2110, 2164), area_scale=area_scale,
                                                         wav_scale=area_scale)
        if ii > 0:
            mod2_spec.plot(label=f"comp {ctemp}")
    for limits in rv_limits:
        plt.axvline(limits[0], color="black", alpha=0.8, ls="--")
        plt.axvline(limits[-1], color="black", alpha=0.8, ls="--")
        plt.title(f"{phoenix} large companions 4000-5000K")
    plt.legend()
    plt.show()

print("Starfish grid", simulators.starfish_grid["hdf5_path"])
import os

print(os.getcwd())
