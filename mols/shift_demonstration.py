import numpy as np
import PyAstronomy.pyasl as pyasl
import matplotlib.pyplot as plt


obs_norm = np.genfromtxt(
    "/home/delta/miras_1/continuum/obs_norm_zro_corrected.txt", skip_header=1
)
zro_spectrum = np.genfromtxt(
    "/home/delta/miras_1/continuum/zro_normalized.txt", skip_header=1
)


fig, ax = plt.subplots()
ax[1].plot(nu_obs, F_obs_recalibrated, label="obs data recalibrated", color="black")
ax[1].plot(total_optical_depth[:, 0], F_transmitted, label="Transmited")

ax[0].plot([4637.909, 4637.909], [0, max(optical_depth2[:, 1])], label="element 1")
ax[0].plot([4638.95, 4638.95], [0, max(optical_depth2[:, 1])], label="element 2")
ax[1].plot([4637.909, 4637.909], [0, max(F_obs_recalibrated)], label="element 1")
ax[1].plot([4638.95, 4638.95], [0, max(F_obs_recalibrated)], label="element 2")
