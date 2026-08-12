import numpy as np
import matplotlib.pyplot as plt
import scienceplots


def doppler_shift(observed_wavelength, rest_wavelength):
    if observed_wavelength is None:
        return None
    c = 299792.458  # km / s
    z = (observed_wavelength - rest_wavelength) / rest_wavelength
    return z * c


obs_norm = np.genfromtxt(
    "/home/delta/miras_1/continuum/obs_norm_molecular_corrected.txt", skip_header=1
)

obs_norm_arr = []
rv_d1_arr = []

for spec_num in range(7):
    obs_norm = np.genfromtxt(
        f"/home/delta/looks_the_same/{spec_num}/obs_norm_molecular_corrected.txt",
        skip_header=1,
    )
    obs_norm_arr.append(obs_norm)
    rv_d1_arr.append(doppler_shift(obs_norm[:, 0], 5889.95))


with plt.style.context(["science"]):
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--"]

    fig, ax = plt.subplots()
    ax.set_title("Na D velocity profile")
    for spectra in range(len(obs_norm_arr)):
        ax.plot(
            rv_d1_arr[spectra],
            obs_norm_arr[spectra][:, 1],
            label=f" profile spectra {spectra}",
            color=colors[spectra],
            ls=linestyles[spectra],
        )
    ax.set_xlim(-50, 50)
    ax.set_xlabel("Velocity")
    ax.set_ylabel("Relative intensity")
    ax.grid()
    ax.set_ylim((0, 1))
    ax.legend()
    plt.show()
