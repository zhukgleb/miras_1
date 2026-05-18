import numpy as np
import PyAstronomy.pyasl as pyasl
from dech_processing import make_txt_from_spectra
import matplotlib.pyplot as plt
import scienceplots
import os


dir_path = os.path.dirname(os.path.realpath(__file__))
folder_to_spectra = dir_path + "/R_Cam/"
spectra_content = os.listdir(folder_to_spectra)

spectra_content_old = [
    "20121126",
    "20140417",
    "20130529",
    "20111115",
    "20120802",
    "20140811",
    "20131009",
]

spectra_content = [
"20120802",
"20121126",
"20130202",
"20130529",
"20131008",
"20140417",
"20140811"
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


bcvr_arr_old = [
    4450.928,
    6698.503,
    -4907.927,
    -6186.043,
    10429.766,
    -10185.650,
    5704.073,
]

bcvr_arr = [
    9193.632,
    6700.746,
    -4907.927,
    -4838.691,
    2649.248,
    8686.852,
    5704.073
]

for i in range(len(rd)):
    _, rd[i][:, 0] = pyasl.dopplerShift(
        rd[i][:, 0], rd[i][:, 1], bcvr_arr[i] / 1000, edgeHandling="firstlast"
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


h_alpha = h_data[0][0]
x = h_alpha[:, 0]
y = h_alpha[:, 1]
# y = y - 7500
y = y - np.median(y)


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
    "2012 11 26",
    "2014 04 17",
    "2013 05 29",
    "2011 11 15",
    "2012 08 02",
    "2014 08 11",
    "2013 10 09",
]


lines = [
    r"$H_{\alpha}$",
    r"$H_{\beta}$",
    r"$H_{\gamma}$",
    r"$H_{\delta}$",
    r"$H_{\epsilon}$",
]
plot = True


if plot:
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

        # plt.show()
        h_gamma = [row[1] for row in h_data]

        #        fig, ax = plt.subplots(nrows=len(h_gamma), ncols=2)
        #        for i in range(len(ax)):
        #            if i == 0:
        #                for j in range(len(ax[0])):
        #                    ax[i][j].plot(
        #                        h_gamma[i][:, 0], h_gamma[i][:, 1], color="black", alpha=1
        #                    )
        #            else:
        #                pass

        #        plt.show()
        #
        #
        fig, ax = plt.subplots()
        for i in range(len(h_gamma)):
            plt.plot(h_gamma[i][:, 0], h_gamma[i][:, 1], label=i)
        plt.legend()
        plt.show()
#


#    for i in range(len(rd)):
#        ax.plot(rd[i][:, 0], rd[i][:, 1])
#    plt.show()
