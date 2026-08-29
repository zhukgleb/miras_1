import pandas as pd
import matplotlib.pyplot as plt
import PyAstronomy.pyasl as pyasl
import numpy as np
import scienceplots


mol_data = np.genfromtxt("/home/delta/exocross/input/H2O_all.xsec")
o_data = np.genfromtxt("/home/delta/exocross/input/O2.xsec")

nu_s, sigma_s = mol_data[:, 0], mol_data[:, 1]
nu_o, sigma_o = o_data[:, 0], o_data[:, 1]

nu_s = 1e8 / nu_s
nu_o = 1e8 / nu_o

nu_s = nu_s[::-1]
nu_o = nu_o[::-1]

nu_s = pyasl.vactoair2(nu_s)
nu_o = pyasl.vactoair2(nu_o)
sigma_s = sigma_s[::-1]
sigma_o = sigma_o[::-1]


obs_norm = np.genfromtxt(
    "/home/delta/looks_the_same/1/obs_norm_molecular_corrected.txt", skip_header=1
)


with plt.style.context(["science", "ieee"]):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_title("Atmospheric lines")
    ax.set_ylabel("Relative intensity")
    ax.set_xlabel(r"Wavelength, $\AA$")
    ax.plot(
        obs_norm[:, 0],
        obs_norm[:, 1],
        label="observed spectra",
        color="black",
        alpha=0.9,
    )
    n_s = 2.1e22
    n_o = 1.5e24
    ax.plot(
        nu_s,
        1 * np.exp(-sigma_s * n_s),
        label=f"H2O, n={n_s},  n / cm-2",
        ls="-",
        alpha=0.8,
    )
    ax.plot(
        nu_o,
        1 * np.exp(-sigma_o * n_o),
        label=f"O2, n={n_o}, n/cm-2",
        ls="-",
        alpha=0.8,
    )

    # ax.set_xlim((min(obs_norm[:, 0]), max(obs_norm[:, 0])))
    # ax.set_xlim((6910, 6970))
    # ax.set_ylim(0, 1.3)
    #
    ax.set_xlim((6865, 6885))
    ax.set_ylim(0, 1.3)

    ax.legend()
    plt.savefig("atmospheric lines_1.pdf")
    plt.savefig("atmospheric lines_1.png", dpi=300)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_title("Atmospheric lines")
    ax.set_ylabel("Relative intensity")
    ax.set_xlabel(r"Wavelength, $\AA$")

    ax.plot(
        obs_norm[:, 0],
        obs_norm[:, 1],
        label="observed spectra",
        color="black",
        alpha=0.9,
    )
    ax.plot(
        nu_s,
        1 * np.exp(-sigma_s * n_s),
        label=f"H2O, n={n_s},  n / cm-2",
        ls="-",
        alpha=0.8,
    )
    ax.plot(
        nu_o,
        1 * np.exp(-sigma_o * n_o),
        label=f"O2, n={n_o}, n/cm-2",
        ls="-",
        alpha=0.8,
    )

    ax.set_xlim((6910, 6970))
    # ax.set_ylim(0, 1.3)
    #
    # ax.set_xlim((6865, 6885))
    ax.set_ylim(0, 1.3)

    ax.legend()
    plt.savefig("atmospheric lines_2.pdf")
    plt.savefig("atmospheric lines_2.png", dpi=300)
