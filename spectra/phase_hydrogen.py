import numpy as np
import PyAstronomy.pyasl as pyasl
from dech_processing import make_txt_from_spectra
import matplotlib.pyplot as plt
import scienceplots
import os


dir_path = os.path.dirname(os.path.realpath(__file__))
folder_to_spectra = dir_path + "/R_Cam/"
spectra_content = os.listdir(folder_to_spectra)

spectra_content = [
    "20121126",
    "20140417",
    "20130529",
    "20111115",
    "20120802",
    "20140811",
    "20131009",
]
# spectra_content = os.listdir(folder_to_spectra)
# spectra_content.remove("20120413")
# spectra_content.remove("20131008")
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

for i in range(len(rd)):
    _, rd[i][:, 0] = pyasl.dopplerShift(
        rd[i][:, 0], rd[i][:, 1], (bcvr_arr[i] - 2500) / 1000, edgeHandling="firstlast"
    )


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


h_alpha = h_data[6][1]
x = h_alpha[:, 0]
y = h_alpha[:, 1]
# y = y - 7500
y = y - 500
y = y / 1500

np.savetxt("h_beta.txt", np.column_stack((x, y)))

# date = [
#    "15.11.2011 \n 0.52",
#    "02.08.2012 \n 0.54",
#    "26.11.2012 \n 0.1",
#    "02.02.2013 \n 0.85",
#    "29.05.2013 \n 0.41",
#    "09.10.2013 \n 0.91",
#   "17.04.2014 \n 0.20",
#    "11.08.2014 \n 0.76",
# ]

date = [
    "0.10",
    "0.20",
    "0.41",
    "0.52",
    "0.54",
    "0.76",
    "0.91",
]


lines = [
    r"$H_{\alpha}$",
    r"$H_{\beta}$",
    r"$H_{\gamma}$",
    r"$H_{\delta}$",
    r"$H_{\epsilon}$",
]
plot = True


identification = {
    3966.10: "Fe 45",
    3972.18: "Ni 29",
    3973.62: "Ni 31",
    4098.40: "Pr II",
    4099.97: "Fe",
    4100.20: "SiH (0, 0)?",
    4105.18: "V 27",
    4341.36: "Ti II Gd II",
    4337.05: "Fe I",
}

from atlas import *

atlas_lines = list(atlas_lines.keys())

if plot:
    with plt.style.context("science"):
        fig, ax = plt.subplots(nrows=len(h_data), ncols=len(h_lines))
        for i in range(len(ax)):
            for j in range(len(ax[i])):
                ax[i][j].plot(
                    h_data[i][j][:, 0], h_data[i][j][:, 1], color="black", alpha=1
                )
                for x in range(len(atlas_lines)):
                    try:
                        if atlas_lines[x] < min(h_data[i][j][:, 0]):
                            pass
                        elif atlas_lines[x] > max(h_data[i][j][:, 0]):
                            pass
                        else:
                            ax[i][j].plot(
                                [atlas_lines[x], atlas_lines[x]],
                                [0, max(h_data[i][j][:, 1])],
                            )
                    except ValueError:
                        ax[i][j].plot([atlas_lines[x], atlas_lines[x]], [0, 0])

                if j == 0:
                    ax[i, j].set_ylabel(date[i], fontsize=12, rotation=90, labelpad=10)
                if i == 0:
                    ax[i, j].set_title(lines[j], fontsize=12, pad=10)

        plt.show()

        # for i in range(len(h_data)):
        #    fig, ax = plt.subplots(nrows=1, ncols=len(h_lines), figsize=(15, 4))
        #    for j in range(len(ax)):
        #        ax[j].plot(
        #            h_data[i][j][:, 0], h_data[i][j][:, 1], color="black", alpha=1
        #        )
        # plt.tight_layout()
        #    fig.suptitle(f"phase: {date[i]}")
        # plt.savefig(f"{i}_.pdf")
        # plt.show()
        #

        for i in range(len(lines)):
            fig, ax = plt.subplots(nrows=1, ncols=len(h_data), figsize=(15, 4))
            for j in range(len(ax)):
                ax[j].plot(
                    h_data[j][i][:, 0], h_data[j][i][:, 1], color="black", alpha=1
                )
                ax[j].set_title(date[j])
            # plt.tight_layout()
            fig.suptitle(f"{lines[i]}")
            # plt.show()
            # plt.savefig(f"{lines[i]}.pdf")

#    for i in range(len(rd)):
#        ax.plot(rd[i][:, 0], rd[i][:, 1])
#    plt.show()
