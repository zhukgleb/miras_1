import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import PyAstronomy.pyasl as pyasl
from scipy.interpolate import interp1d
import scienceplots

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "mols"))


from mols_work import Molecule


cp1 = "/home/delta/exocross/input/TiO_all.xsec"
cp2 = "/home/delta/exocross/input/ZrO_all.xsec"
star_spectrum_path = "/home/delta/miras_1/mols/synth_all.spec"
obs_spectrum_path = "/home/delta/miras_1/RV/norm_spectra_3.txt"
cross_section_data_1 = np.genfromtxt(cp1)
cross_section_data_2 = np.genfromtxt(cp2)
nu_star, F_star_norm, F_star = np.loadtxt(star_spectrum_path, unpack=True)
nu_obs, F_obs = np.loadtxt(obs_spectrum_path, unpack=True)

tio = Molecule(
    name="TiO",
    velocity=-0,
    column_density=2e1,
    cross_section=cross_section_data_1,
    stick_spectrum=None,
)

zro = Molecule(
    name="ZrO", velocity=-0, column_density=1e16, cross_section=cross_section_data_2
)
optical_depth_tio = tio.get_optical_depth()
optical_depth_zro = zro.get_optical_depth()
total_optical_depth = tio + zro

# Basic model -- 1 * e^{-tau}

F_interp_func = interp1d(nu_star, [1 for x in range(len(nu_star))], kind="linear", fill_value=0.0, bounds_error=False)
F_star_interp = F_interp_func(total_optical_depth[:, 0])
F_transmitted = F_star_interp * np.exp(-total_optical_depth[:, 1])


# rv spectra is -35.73108524114256
_, nu_obs = pyasl.dopplerShift(nu_obs, F_obs, 35.73, edgeHandling="firstlast")

fig, ax = plt.subplots()
ax.plot(total_optical_depth[:, 0], total_optical_depth[:, 1], label="tau zro + tio")


fig, ax = plt.subplots()
ax.set_xlim((5886, 5900))
ax.set_ylim((0, 2))
ax.plot(nu_star, F_star_norm, label="synth")
ax.plot(nu_obs, F_obs, label="obs")
plt.legend()


fig, ax = plt.subplots()
# total_optical_depth[:, 1]  = total_optical_depth[:, 1]  * np.median(total_optical_depth[:, 1]) / 2
# ax.plot(total_optical_depth[:, 0], total_optical_depth[:, 1])
ax.plot(total_optical_depth[:, 0], F_transmitted, label="1 * e^-tau")
ax.plot(nu_obs, F_obs, label="obs")
plt.show()