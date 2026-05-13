import matplotlib.pyplot as plt
import scienceplots
import numpy as np
from atlas import *
import PyAstronomy.pyasl as pyasl

from atlas import *

data = np.genfromtxt("rcam_medium.txt")
wave = data[:, 0]
flux = data[:, 1]
_, wave = pyasl.dopplerShift(wave, flux, -47, edgeHandling="firstlast")


with plt.style.context("science"):
    fig, ax = plt.subplots()
    ax.plot(wave, flux, color="black")
    ymin, ymax = ax.get_ylim()
    xmin, xmax = min(wave), max(wave)
    ax.set_xlim((xmin, xmax))
    for wl, label in atlas_all.items():
        ax.axvline(x=wl, color="red", linestyle="--", linewidth=0.8, alpha=0.7)

        # Текстовая метка (можно настроить угол поворота)
        ax.text(
            wl,
            ymax * 0.95,
            label,
            rotation=90,
            verticalalignment="top",
            fontsize=8,
            color="darkred",
        )

    plt.show()


#                    if atlas_lines[x] < min(h_data[i][j][:, 0]):
#                        pass
#                    elif atlas_lines[x] > max(h_data[i][j][:, 0]):
#                        pass
#                    else:
#                        ax[i][j].plot(
#                            [atlas_lines[x], atlas_lines[x]],
#                            [0, max(h_data[i][j][:, 1])],
#                        )
#                except ValueError:
#                   ax[i][j].plot([atlas_lines[x], atlas_lines[x]], [0, 0])
