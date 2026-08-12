import numpy as np
import PyAstronomy.pyasl as pyasl
import matplotlib.pyplot as plt
import scienceplots

# for 6 spectra:
# rv = -79.5 - 1.06
# rv_tio = -79 - 1.06
# rv_combo = -80 - 1.06


rv = -79.5 - 1.06
rv_tio = -79 - 1.06
rv_combo = -80 - 1.06

obs_norm_arr = []

for spec_num in range(6):
    obs_norm = np.genfromtxt(
        f"/home/delta/looks_the_same/{spec_num}/obs_norm_molecular_corrected.txt",
        skip_header=1,
    )
    obs_norm_arr.append(obs_norm)

# fig, ax = plt.subplots()
# ax.plot(obs_norm[:, 0], obs_norm[:, 1], label="obs")
# ax.plot(zro_spectrum[:, 0], zro_spectrum[:, 1], label=f"zro shifted on {rv} km/s")
# ax.plot(tio_spectrum[:, 0], tio_spectrum[:, 1], label=f"tio shifted on {rv_tio} km/s")
# ax.legend()


with plt.style.context(["science"]):
    fig, ax = plt.subplots()
    ax.set_title(r"Na region")
    ax.set_xlim((5876, 5931))
    ax.set_ylim((0, 1.4))
    for spectra in range(6):
        ax.plot(
            obs_norm_arr[spectra][:, 0],
            obs_norm_arr[spectra][:, 1],
            label=f"obs {spectra}",
        )
        ax.legend()

    fig, ax = plt.subplots()
    ax.set_title("Ca I region")
    ax.set_xlim((4225, 4228))
    ax.set_ylim((0, 1.4))
    for spectra in range(6):
        ax.plot(
            obs_norm_arr[spectra][:, 0],
            obs_norm_arr[spectra][:, 1],
            label=f"obs {spectra}",
        )
        ax.legend()

    ax.legend()

    fig, ax = plt.subplots()
    ax.set_title("CH region")
    ax.set_xlim((4299, 4301))
    ax.set_ylim((0, 1.4))
    for spectra in range(6):
        ax.plot(
            obs_norm_arr[spectra][:, 0],
            obs_norm_arr[spectra][:, 1],
            label=f"obs {spectra}",
        )
    ax.legend()

    fig, ax = plt.subplots()
    ax.set_title("DIB region")
    ax.set_xlim((6180, 6220))
    ax.set_ylim((0, 1.4))
    for spectra in range(6):
        ax.plot(
            obs_norm_arr[spectra][:, 0],
            obs_norm_arr[spectra][:, 1],
            label=f"obs {spectra}",
        )
    ax.legend()


plt.show()
