import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import PyAstronomy.pyasl as pyasl

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "spectra"))

from dech_processing import make_txt_from_spectra
from general_processing import (
    median_normalization,
    full_pipeline,
    normalize_orders_median,
)


def depricated():
    dir_path = os.path.dirname(os.path.realpath(__file__)).replace("RV", "")
    folder_to_spectra = dir_path + "/spectra/R_Cam/"
    spectra_content = os.listdir(folder_to_spectra)

    spectra_content = [
        "20120802",
        "20121126",
        "20130202",
        "20130529",
        "20131008",
        "20140417",
        "20140811",
    ]

    bcvr_arr = [
        4450.769,
        6698.387,
        -5093.433,
        -6032.999,
        10471.263,
        -10221.369,
        5778.002,
    ]

    i = 0
    save = True
    reduction_arr = []

    for file in spectra_content:
        spectra_path = folder_to_spectra + file + "/"
        data = make_txt_from_spectra(spectra_path, True, True)
        # median_norm = normalize_orders_median(data)
        # final = full_pipeline(data, degree=2, do_median=True, do_continuum=True)
        _, data[:, 0] = pyasl.dopplerShift(
            data[:, 0], data[:, 1], bcvr_arr[i] / 1000, edgeHandling="firstlast"
        )
        i += 1
        reduction_arr.append(data)
        if save:
            np.savetxt(f"/home/delta/miras_1/RV/{file}.txt", data)

    import matplotlib.pyplot as plt
    import scienceplots

    star_spectrum_path = "/home/delta/miras_1/mols/synth_all.spec"
    nu_star, F_star_norm, F_star = np.loadtxt(star_spectrum_path, unpack=True)

    # fig, ax = plt.subplots()
    # for j in range(len(reduction_arr)):
    #     ax.plot(reduction_arr[j][:, 0], reduction_arr[j][:, 1])

    # plt.show()

    for date in range(len(reduction_arr)):
        fig, ax = plt.subplots()
        ax.plot(
            nu_star,
            F_star_norm * np.median(reduction_arr[date][:, 1]),
            label="Synth data",
            color="crimson",
            alpha=0.8,
        )
        ax.plot(reduction_arr[date][:, 0], reduction_arr[date][:, 1], label="obs data")
        ax.set_xlim((4220, 4235))
        plt.show()

    # with plt.style.context(["science", "ieee"]):
    #     fig, ax = plt.subplots(ncols=2, figsize=(4, 2))
    #     # ax[0].plot(nu_star, F_star*np.median(reduction_arr[3][:, 1]), label="Synth data", color="crimson", alpha=0.8)
    #     # ax[0].plot([4226.728, 4226.728], [0, max(reduction_arr[3][:, 1])], label="Ca I 4226.728", color="green", linestyle="-.")

    #     # ax[1].plot([5889.95, 5889.95], [0, max(reduction_arr[3][:, 1])], label="Na I 5889.95", color="green", linestyle="-.")
    #     # ax[1].plot([5895.92, 5895.92], [0, max(reduction_arr[3][:, 1])], label="Na I 5895.92", color="green", linestyle="-.")
    #     # ax[1].plot(nu_star, F_star*np.median(reduction_arr[3][:, 1]), label="Synth data", color="crimson", alpha=0.8)

    #     for j in range(len(reduction_arr)):
    #         ax[0].set_xlim((4220, 4235))
    #         ax[0].plot(reduction_arr[j][:, 0], reduction_arr[j][:, 1], label=f"{j}", alpha=0.7)
    #         ax[0].set_title("Ca I line")
    #         # ax[0].set_ylim((0, 1.1e2))

    #         ax[1].set_title("Na I line")
    #         ax[1].set_xlim((5886, 5900))
    #         ax[1].plot(reduction_arr[j][:, 0], reduction_arr[j][:, 1], label=f"{j}", alpha=0.7)

    #         # ax[1].set_ylim((0, 1.1e4))
    #         ax[0].legend()
    #         ax[1].legend()
    #         plt.tight_layout()

    #     plt.show()

    pass


data_arr = []
star_spectrum_path = "/home/delta/miras_1/mols/synth_all.spec"
nu_star, F_star_norm, F_star = np.loadtxt(star_spectrum_path, unpack=True)

for i in range(0, 7):
    data_arr.append(np.genfromtxt(f"norm_spectra_{i}.txt"))

# fig, ax = plt.subplots()
# ax.set_ylim(0, 2)
# ax.plot(nu_star, F_star_norm, label="Synth", color="black", alpha=0.7)

# for i in range(len(data_arr)):
#     ax.plot(data_arr[i][:, 0], data_arr[i][:, 1], label=i)
# #
# ax.legend()
# plt.show()


for i in range(len(data_arr)):
    fig, ax = plt.subplots()
    ax.plot(data_arr[i][:, 0], data_arr[i][:, 1], label=i + 1)
    ax.plot(nu_star, F_star_norm, label="Synth")
    ax.set_ylim(0, 2)
    ax.set_xlim((4220, 4235))
    ax.legend()
    # plt.show()


def doppler_shift(observed_wavelength, rest_wavelength):
    if observed_wavelength is None:
        return None
    c = 299792.458  # km / s
    z = (observed_wavelength - rest_wavelength) / rest_wavelength
    return z * c


date = [20120802, 20121126, 20130202, 20130529, 20131008, 20131009, 20140417, 20140811]
phase = [
    0.5456244988037824,
    0.1070403905168877,
    0.8550671815539488,
    0.4143372847806423,
    0.9190653269519604,
    0.9190653269519604,
    0.2049986121020386,
    0.7687455734933750,
]
lambda_nad1_a = [5889.243, None, 5889.319, 5889.248, 5889.318, None, 5889.233, 5889.257]
lambda_nad2_a = [5895.222, None, 5895.27, 5895.218, 5895.278, None, 5895.224, 5895.233]
lambda_ca = [4226.46, None, 4226.31, 4226.30, None, None, None, 4226.318]

rv_d1 = [doppler_shift(lambda_nad1_a[x], 5889.95) for x in range(len(lambda_nad1_a))]
rv_d2 = [doppler_shift(lambda_nad2_a[x], 5895.92) for x in range(len(lambda_nad2_a))]
rv_ca = [doppler_shift(lambda_ca[x], 4226.72) for x in range(len(lambda_ca))]

fig, ax = plt.subplots()
x = [x for x in range(len(rv_d1))]
ax.scatter(phase, rv_d1, label="Na D1")
ax.scatter(phase, rv_d2, label="Na D2")
ax.scatter(phase, rv_ca, label="Ca")
plt.legend()

fig, ax = plt.subplots()
x = [x for x in range(len(rv_d1))]
ax.scatter(x, rv_d1, label="Na D1")
ax.scatter(x, rv_d2, label="Na D2")
ax.scatter(x, rv_ca, label="Ca")
plt.legend()
print(f"RV 0.41 spectra: {rv_d1[6]} km/s")
plt.show()
