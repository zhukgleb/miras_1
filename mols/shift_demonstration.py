import numpy as np
import PyAstronomy.pyasl as pyasl
import matplotlib.pyplot as plt
import scienceplots


obs_norm = np.genfromtxt(
    "/home/delta/miras_1/continuum/obs_norm_zro_corrected.txt", skip_header=1
)
zro_spectrum = np.genfromtxt(
    "/home/delta/miras_1/continuum/zro_normalized.txt", skip_header=1
)
tio_spectrum = np.genfromtxt(
    "/home/delta/miras_1/continuum/tio_normalized.txt", skip_header=1
)

combo_spectrum = np.genfromtxt(
    "/home/delta/miras_1/continuum/molecular_combined.txt", skip_header=1
)


synth_spectrum = np.genfromtxt(
    "/home/delta/miras_1/continuum/2026-07-20-13-28-24_0.7248514425106289_LTE_synthetic_spectra_parameters/0.spec"
)
# rv = -103.5
rv = -80.5
rv_tio = -79
rv_combo = -80

_, zro_spectrum[:, 0] = pyasl.dopplerShift(
    zro_spectrum[:, 0], zro_spectrum[:, 1], rv, edgeHandling="firstlast"
)

_, tio_spectrum[:, 0] = pyasl.dopplerShift(
    tio_spectrum[:, 0], tio_spectrum[:, 1], rv_tio, edgeHandling="firstlast"
)

_, combo_spectrum[:, 0] = pyasl.dopplerShift(
    combo_spectrum[:, 0], combo_spectrum[:, 1], rv_combo, edgeHandling="firstlast"
)


# fig, ax = plt.subplots()
# ax.plot(obs_norm[:, 0], obs_norm[:, 1], label="obs")
# ax.plot(zro_spectrum[:, 0], zro_spectrum[:, 1], label=f"zro shifted on {rv} km/s")
# ax.plot(tio_spectrum[:, 0], tio_spectrum[:, 1], label=f"tio shifted on {rv_tio} km/s")
# ax.legend()


with plt.style.context(["science", "ieee"]):
    fig, ax = plt.subplots(nrows=3, figsize=(4, 6))
    ax[0].plot(obs_norm[:, 0], obs_norm[:, 1], label="Obs data")
    ax[0].plot(
        zro_spectrum[:, 0],
        zro_spectrum[:, 1],
        label=f"ZrO shifted on {rv} km/s",
        ls="-",
        alpha=0.6,
        color="crimson",
        lw=1,
    )
    ax[0].set_xlim((4612, 4679))

    ax[1].set_title("Na I line")
    ax[1].set_xlim((5886, 5900))
    ax[1].plot(
        synth_spectrum[:, 0],
        synth_spectrum[:, 1],
        label="Synth data",
        alpha=0.6,
        ls="-.",
    )
    ax[1].plot(obs_norm[:, 0], obs_norm[:, 1], label="Obs data", color="black", ls="-")
    ax[1].plot(
        [5889.95, 5889.95], [0, 3], label="Na I 5889.95", color="green", linestyle="-."
    )
    ax[1].plot(
        [5895.92, 5895.92], [0, 3], label="Na I 5895.92", color="green", linestyle="-."
    )

    ax[1].set_xlim((5886, 5900))

    ax[2].plot(obs_norm[:, 0], obs_norm[:, 1], label="obs")
    ax[2].plot(
        zro_spectrum[:, 0],
        zro_spectrum[:, 1],
        label=f"ZrO shifted on {rv} km/s",
        ls="-",
        alpha=0.6,
        lw=1,
    )
    ax[2].plot(
        tio_spectrum[:, 0],
        tio_spectrum[:, 1],
        label=f"TiO shifted on {rv_tio} km/s",
        ls="-",
        alpha=0.6,
        lw=1,
    )
    ax[2].plot(
        combo_spectrum[:, 0],
        combo_spectrum[:, 1],
        label=f"Combined spectra TiO and ZrO shifted on {rv_combo} km/s",
        ls="-",
        alpha=0.6,
        lw=1,
    )

    ax[2].set_xlim((6440, 6680))

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[2].set_xlabel(r"Wavelength, \AA")
    ax[1].set_ylabel(r"Relative intensity, \AA")

    plt.tight_layout()
    #    plt.savefig("shifts.pdf")
    #    plt.savefig("shift.png", dpi=300)

    fig, ax = plt.subplots()
    ax.set_title(r"H_{\alpha} region")
    ax.set_xlim((6550, 6570))
    ax.set_ylim((0, 1.2))

    ax.plot([6562.8, 6562.8], [0, 1], label=r"H_{\alpha} rest")
    ax.plot(obs_norm[:, 0], obs_norm[:, 1], label="obs")
    ax.plot(
        zro_spectrum[:, 0],
        zro_spectrum[:, 1],
        label=f"ZrO shifted on {rv} km/s",
        ls="-",
        alpha=0.6,
        lw=1,
    )
    ax.plot(
        tio_spectrum[:, 0],
        tio_spectrum[:, 1],
        label=f"TiO shifted on {rv_tio} km/s",
        ls="-",
        alpha=0.6,
        lw=1,
    )
    ax.plot(
        combo_spectrum[:, 0],
        combo_spectrum[:, 1],
        label=f"Combined spectra TiO and ZrO shifted on {rv_combo} km/s",
        ls="-",
        alpha=0.6,
        lw=1,
    )

    ax.legend()
    plt.show()
