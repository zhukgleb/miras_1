import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import PyAstronomy.pyasl as pyasl


def doppler_shift(observed_wavelength, rest_wavelength):
    if observed_wavelength is None:
        return None
    c = 299792.458  # km / s
    z = (observed_wavelength - rest_wavelength) / rest_wavelength
    return z * c


obs_norm = np.genfromtxt(
    "/home/delta/miras_1/continuum/obs_norm_molecular_corrected.txt", skip_header=1
)

synth_spectrum = np.genfromtxt(
    "/home/delta/looks_the_same/2026-07-20-13-28-24_0.7248514425106289_LTE_synthetic_spectra_parameters/0.spec"
)

obs_norm_arr = []
obs_norm_arr_A = []
obs_norm_arr_B = []

rv_d1_arr = []
rv_d1_arr_A = []
rv_d1_arr_B = []


for spec_num in range(7):
    obs_norm = np.genfromtxt(
        f"/home/delta/looks_the_same/{spec_num}/obs_norm_molecular_corrected.txt",
        skip_header=1,
    )
    obs_norm_arr.append(obs_norm)
    obs_norm_arr_A.append(obs_norm)
    obs_norm_arr_B.append(obs_norm)

    rv_d1_arr.append(doppler_shift(obs_norm[:, 0], 5889.95))

synth_rv = doppler_shift(synth_spectrum[:, 0], 5889.95)

spectra_date = [
    2456141.49514,
    2456258.17083,
    2456325.20278,
    2456442.44931,
    2456574.20556,
    2456764.16736,
    2456880.22292,
]

first_graph = False

with plt.style.context(["ieee", "bright"]):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_title("Na D velocity profile")
    ax.plot(
        synth_rv,
        synth_spectrum[:, 1],
        color="black",
        label="synthetic",
    )
    for spectra in range(len(obs_norm_arr)):
        ax.plot(
            rv_d1_arr[spectra],
            obs_norm_arr[spectra][:, 1],
            label=f"{float(spectra_date[spectra]) - 2456000:.2f}",
        )

    ax.plot([-10, -10], [0, 3])
    ax.set_xlim(-50, 50)
    ax.set_xlabel("Velocity, km/s")
    ax.set_ylabel("Relative intensity")
    ax.grid()
    ax.set_ylim((0, 1))
    ax.legend()

    plt.savefig("D_line.pdf")
    plt.savefig("D_line.png", dpi=300)

    if other:
        fig, ax = plt.subplots(figsize=(8, 4))
        na_A_velocity = [-0.5, 0, 3.8, -0.8, 2.8, 0, 0]
        na_B_velocity = [-12, -14.4, -11.3, -10, -10.4, -11.1, -10]
        rv_combo_arr = [-95, -90, -82, -92, -82, -86, -80]
        rv_combo_arr = [x + 81 for x in rv_combo_arr]

        ax.plot(spectra_date, na_A_velocity, label="Velocity Na component A")
        ax.plot(spectra_date, na_B_velocity, label="Velocity Na component B")
        ax.plot(spectra_date, rv_combo_arr, label="ZrO velocity")
        ax.legend()

        for spec_num in range(len(na_A_velocity)):
            _, obs_norm_arr_A[spec_num][:, 0] = pyasl.dopplerShift(
                obs_norm_arr_A[spec_num][:, 0],
                obs_norm_arr_A[spec_num][:, 1],
                -na_A_velocity[spec_num],
                edgeHandling="firstlast",
            )
            rv_d1_arr_A.append(doppler_shift(obs_norm_arr_A[spec_num][:, 0], 5889.95))
            _, obs_norm_arr_B[spec_num][:, 0] = pyasl.dopplerShift(
                obs_norm_arr_B[spec_num][:, 0],
                obs_norm_arr_B[spec_num][:, 1],
                -na_B_velocity[spec_num],
                edgeHandling="firstlast",
            )
            rv_d1_arr_B.append(doppler_shift(obs_norm_arr_B[spec_num][:, 0], 5889.95))

        fig, ax = plt.subplots(ncols=2)
        ax[0].set_title("After correction A component")
        ax[1].set_title("After correction B component")
        for spectra in range(len(obs_norm_arr)):
            ax[0].plot(
                rv_d1_arr_A[spectra],
                obs_norm_arr_A[spectra][:, 1],
                label=f"profile spectra {spectra}",
            )
            ax[1].plot(
                rv_d1_arr_B[spectra],
                obs_norm_arr_B[spectra][:, 1],
                label=f"profile spectra {spectra}",
            )

        ax[0].set_xlim(-50, 50)
        ax[0].set_xlabel("Velocity")
        ax[0].set_ylabel("Relative intensity")
        ax[0].grid()
        ax[0].set_ylim((0, 1))
        ax[0].legend()

        ax[1].set_xlim(-50, 50)
        ax[1].set_xlabel("Velocity")
        ax[1].set_ylabel("Relative intensity")
        ax[1].grid()
        ax[1].set_ylim((0, 1))
        ax[1].legend()

plt.show()
