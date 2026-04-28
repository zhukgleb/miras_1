import numpy as np
import PyAstronomy.pyasl as pyasl
from dech_processing import make_txt_from_spectra
import matplotlib.pyplot as plt
import scienceplots
import os


dir_path = os.path.dirname(os.path.realpath(__file__))
folder_to_spectra = dir_path + "/R_Cam/"
spectra_content = os.listdir(folder_to_spectra)
spectra_content.remove("20120413")
spectra_content.remove("20131008")
spectra_path = [
    folder_to_spectra + spectra_content[i] + "/" for i in range(len(spectra_content))
]

print(spectra_path)

rd = []
for i in range(len(spectra_path)):
    rd.append(make_txt_from_spectra(spectra_path[i], True, True))
# rd = make_txt_from_spectra(wf, True, True)


bcvr_arr = [
    8200.185,
    4450.928,
    6698.503,
    -4907.927,
    -6186.043,
    10429.766,
    -10185.650,
    5704.073,
]

h_lines = [6562.8, 4861.3, 4340.5, 4101.7, 3970.1]
delta = rd[0][:, 0][1] - rd[0][:, 0][0]
delta_idx = 5 // delta

h_data = []
for i in range(len(rd)):
    h_data_spec = []
    for j in range(len(h_lines)):
        center_idx = np.argmin(np.abs(rd[i][:, 0] - h_lines[j]))
        h_data_spec.append(
            rd[i][int(center_idx - delta_idx) : int(center_idx + delta_idx)]
        )
    h_data.append(h_data_spec)

print(len(rd))
print(len(h_data))
for i in range(len(rd)):
    _, rd[i][:, 0] = pyasl.dopplerShift(
        rd[i][:, 0], rd[i][:, 1], bcvr_arr[i] / 1000, edgeHandling="firstlast"
    )


date = [
    "15.11.2011",
    "02.08.2012",
    "26.11.2012",
    "02.02.2013",
    "29.05.2013",
    "09.10.2013",
    "17.04.2014",
    "11.08.2014",
]

lines = [
    r"$H_{\alpha}$",
    r"$H_{\beta}$",
    r"$H_{\gamma}$",
    r"$H_{\delta}$",
    r"$H_{\epsilon}$",
]

with plt.style.context("science"):
    fig, ax = plt.subplots(nrows=len(h_data), ncols=len(h_lines))
    for i in range(len(ax)):
        for j in range(len(ax[i])):
            ax[i][j].plot(
                h_data[i][j][:, 0], h_data[i][j][:, 1], color="black", alpha=1
            )
            if j == 0:
                ax[i, j].set_ylabel(date[i], fontsize=12, rotation=90, labelpad=10)
            if i == 0:
                ax[i, j].set_title(lines[j], fontsize=12, pad=10)

    plt.show()

#    for i in range(len(rd)):
#        ax.plot(rd[i][:, 0], rd[i][:, 1])
#    plt.show()
