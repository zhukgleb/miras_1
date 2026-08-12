import numpy as np
import PyAstronomy.pyasl as pyasl
import matplotlib.pyplot as plt
import scienceplots

# for 6 spectra:
# rv = -79.5 - 1.06
# rv_tio = -79 - 1.06
# rv_combo = -80 - 1.06


rv = -84
rv_tio = -84
rv_combo = -81.06


rv_zro_arr = [-81.06, -84, -84, 0, 0, 0, 0]
rv_tio_arr = [-81.06, -84, -84, 0, 0, 0, 0]
# 0 -- ?
# 1 -- Вроде норм, но есть переменность спектра. Кажется, что там только титан
# 2 -- прямо хорошо
# 3 -- норм
# 4 -- норм
# 5 -- норм
# 6 -- норм

rv_combo_arr = [-90, -90, -70, -84, -90, -74, -82]
spectra_date = [
    2456141.49514,
    2456258.17083,
    2456325.20278,
    2456442.44931,
    2456574.20556,
    2456764.16736,
    2456880.22292,
]

zro_spectrum_arr = []
tio_spectrum_arr = []
combo_spectrum_arr = []
obs_norm_arr = []
num_start = 3
num_end = 7

for spec_num in range(7):
    obs_norm = np.genfromtxt(
        f"/home/delta/looks_the_same/{spec_num}/obs_norm_molecular_corrected.txt",
        skip_header=1,
    )
    zro_spectrum = np.genfromtxt(
        f"/home/delta/looks_the_same/{spec_num}/zro_normalized.txt", skip_header=1
    )
    tio_spectrum = np.genfromtxt(
        f"/home/delta/looks_the_same/{spec_num}/tio_normalized.txt", skip_header=1
    )
    combo_spectrum = np.genfromtxt(
        f"/home/delta/looks_the_same/{spec_num}/molecular_combined.txt", skip_header=1
    )
    synth_spectrum = np.genfromtxt(
        "/home/delta/looks_the_same/2026-07-20-13-28-24_0.7248514425106289_LTE_synthetic_spectra_parameters/0.spec"
    )
    _, zro_spectrum[:, 0] = pyasl.dopplerShift(
        zro_spectrum[:, 0],
        zro_spectrum[:, 1],
        rv_zro_arr[spec_num],
        edgeHandling="firstlast",
    )
    _, tio_spectrum[:, 0] = pyasl.dopplerShift(
        tio_spectrum[:, 0],
        tio_spectrum[:, 1],
        rv_tio_arr[spec_num],
        edgeHandling="firstlast",
    )
    _, combo_spectrum[:, 0] = pyasl.dopplerShift(
        combo_spectrum[:, 0],
        combo_spectrum[:, 1],
        rv_combo_arr[spec_num],
        edgeHandling="firstlast",
    )

    obs_norm_arr.append(obs_norm)
    zro_spectrum_arr.append(zro_spectrum)
    tio_spectrum_arr.append(tio_spectrum)
    combo_spectrum_arr.append(combo_spectrum)


# fig, ax = plt.subplots()
# ax.plot(obs_norm[:, 0], obs_norm[:, 1], label="obs")
# ax.plot(zro_spectrum[:, 0], zro_spectrum[:, 1], label=f"zro shifted on {rv} km/s")
# ax.plot(tio_spectrum[:, 0], tio_spectrum[:, 1], label=f"tio shifted on {rv_tio} km/s")
# ax.legend()


with plt.style.context(["science"]):
    fig, ax = plt.subplots()
    ax.set_title(r"$H_{\alpha}$ region")
    ax.set_xlim((6550, 6570))
    ax.set_ylim((0, 3))
    for spectra in range(num_start, num_end):
        ax.plot([6562.8, 6562.8], [0, 5], label=r"$H_{\alpha}$ rest")
        ax.plot(
            obs_norm_arr[spectra][:, 0],
            obs_norm_arr[spectra][:, 1],
            label=f"obs {spectra}",
        )
        #        ax.plot(
        #    zro_spectrum_arr[spectra][:, 0],
        #    zro_spectrum_arr[spectra][:, 1],
        #    label=f"ZrO shifted on {rv_combo_arr[spectra]} km/s",
        #    ls="-",
        #    alpha=0.6,
        #    lw=1,
        # )
        # ax.plot(
        #    tio_spectrum_arr[spectra][:, 0],
        #    tio_spectrum_arr[spectra][:, 1],
        #    label=f"TiO shifted on {rv_combo_arr[spectra]} km/s",
        #    ls="-",
        #    alpha=0.6,
        #    lw=1,
        # )
        ax.plot(
            combo_spectrum_arr[spectra][:, 0],
            combo_spectrum_arr[spectra][:, 1],
            label=f"Combined spectra {rv_combo_arr[spectra]}, num {spectra}",
            ls="-",
            alpha=0.6,
            color="darkred",
        )

        ax.legend()
        plt.show()

    fig, ax = plt.subplots()
    ax.scatter(spectra_date, rv_combo_arr, label="molecular band speed")
    plt.show()
