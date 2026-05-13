import numpy as np
import PyAstronomy.pyasl as pyasl
import scienceplots
import matplotlib.pyplot as plt
from atlas import *
from scipy.signal import find_peaks
import math

data = np.genfromtxt("h_gamma.txt")
wave, flux = data[:, 0], data[:, 1]
# shift = -49.0 + (2500 / 1000)  # in km


# shift = -44 + (2500 / 1000)  # in km

_, wave = pyasl.dopplerShift(wave, flux, 0, edgeHandling="firstlast")

plot = False
if plot:
    with plt.style.context("science"):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(wave, flux, color="black")

        ymin, ymax = ax.get_ylim()
        y_text_position = ymax  # подписи сверху (можно ymax * 0.95)
        offset = 0  # для смещения подписей, если мешают

        for wl, label in gamma.items():
            # Вертикальная линия (пунктир)
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

        # Оформление
        ax.set_xlabel("Wavelength (Å)", fontsize=12)
        ax.set_ylabel("Intensity (arb. units)", fontsize=12)
        ax.set_title("Spectrum with identified lines", fontsize=14)
        #  ax.grid(True, alpha=0.3)
        ax.set_xlim(wave[0], wave[-1])

        plt.tight_layout()
        plt.show()


def atlas_test(wave, flux, atlas):
    peaks, _ = find_peaks(1 / flux, prominence=0.1)
    # plt.plot(wave, flux)
    # plt.plot(wave[peaks], flux[peaks], "x")
    # plt.show()

    detected_lines = wave[peaks]
    nearest_delta = []
    for line in atlas:
        nearest_delta_temp = []
        for j in range(len(detected_lines)):
            nearest_delta_temp.append(abs(line - detected_lines[j]))
            if line < detected_lines[0] or line > detected_lines[-1]:
                nearest_delta_temp.append(math.nan)
        nearest_delta.append(min(nearest_delta_temp))

    # plt.plot([x for x in range(len(nearest_delta))], nearest_delta)
    # plt.show()
    #
    return nearest_delta


if __name__ == "__main__":
    delta_arr = []
    shift_indexes = []
    for shift in range(-100 * 1000, -20 * 1000, 10):
        # shift = -49.0 + (2500 / 1000)  # in km
        _, wave = pyasl.dopplerShift(wave, flux, shift / 1000, edgeHandling="firstlast")
        delta_arr.append(atlas_test(wave, flux, gamma))
        shift_indexes.append(shift)

    fig, ax = plt.subplots(ncols=2)
    for i in range(len(delta_arr)):
        ax[0].plot(
            [x for x in range(len(delta_arr[i]))],
            delta_arr[i],
        )
    mustly_good = min(delta_arr, default=math.nan)
    mg_index = delta_arr.index(mustly_good)
    mg_shift = shift_indexes[mg_index]
    ax[1].plot(
        [x for x in range(len(mustly_good))],
        mustly_good,
        label=f"best shift: {mg_shift}",
    )
    ax[0].plot(
        [x for x in range(len(mustly_good))],
        mustly_good,
        label=f"best shift: {mg_shift}",
        ls="--",
        color="crimson",
    )

    plt.legend()
    plt.show()
