import matplotlib.pyplot as plt
import scienceplots
import numpy as np
from atlas import *
import PyAstronomy.pyasl as pyasl
import random

from atlas import *
from atlas_gen import get_atlas, find_duplicate_terms

all = False
test = True

data = np.genfromtxt("h_gamma.txt")
wave = data[:, 0]
flux = data[:, 1]
_, wave = pyasl.dopplerShift(wave, flux, -53.6131 + 0.3625, edgeHandling="firstlast")


# with plt.style.context("science"):
fig, ax = plt.subplots()
ax.plot(wave, flux, color="black")
ymin, ymax = ax.get_ylim()
xmin, xmax = min(wave), max(wave)
ax.set_xlim((xmin, xmax))

if all:
    for wl, label in term_atlas.items():
        ax.axvline(x=float(wl), color="red", linestyle="--", linewidth=1.2, alpha=0.7)

        # Текстовая метка (можно настроить угол поворота)
        ax.text(
            float(wl),
            ymax * 0.95,
            label,
            rotation=90,
            verticalalignment="top",
            fontsize=12,
            color="crimson",
        )

    plt.show()


if test:
    wd = get_atlas()
    terms = find_duplicate_terms(wd)
    print(terms)

    terms = {
        "Ce II 4D*": [4335.479, 4339.377, 4340.553],
        "Zr II c2D (blend)": [4336.338, 4337.619],
        "Ru I a3F(shift?)": [4337.264, 4342.075],
        "Mn II a5F(shift?)": [4338.367, 4343.98],
        "Ti I c3P": [4340.473, 4343.369],
        "Gd II 8D*": [4341.287, 4342.181],
        "Ti I z3D*": [4342.426, 4342.971],
    }

    for label, wls in terms.items():
        color = np.random.rand(3)
        line_styles = ["-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted"]
        selected_style = random.choice(line_styles)
        for wl in wls:
            ax.axvline(
                x=float(wl),
                color=color,
                linestyle=selected_style,
                linewidth=1.2,
                alpha=0.7,
            )

            ax.text(
                float(wl),
                ymax * 0.95,
                label,
                rotation=90,
                verticalalignment="top",
                fontsize=12,
                color="crimson",
            )

    plt.tight_layout()
    plt.show()
