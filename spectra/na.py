import numpy as np
import matplotlib.pyplot as plt
import scienceplots


data = []


for i in range(6):
    data.append(np.genfromtxt(f"na_{i}.txt"))


na_data = []
delta = data[0][:, 0][1] - data[0][:, 0][0]
delta_idx_na = 1.5 // delta


for i in range(len(data)):
    center_idx = np.argmin(np.abs(data[i][:, 0] - 5890.00))
    na_data.append(
        data[i][int(center_idx - delta_idx_na) : int(center_idx + delta_idx_na)]
    )


with plt.style.context("science"):
    fig, ax = plt.subplots()
    for i in range(len(na_data)):
        na_data[i][:, 1] = na_data[i][:, 1] / np.median(na_data[i][:, 1])
        if i != 1 and i != 5:
            plt.plot(na_data[i][:, 0], na_data[i][:, 1], label=i)
        plt.legend()

    ax.set_xlabel(r"Wavelength, $\AA$")
    ax.set_ylabel("y / median(y)")
    plt.show()
