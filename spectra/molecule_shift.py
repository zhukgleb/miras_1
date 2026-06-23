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
    "20140811",
]

# spectra_content = os.listdir(folder_to_spectra)
# spectra_content.remove("20120413")
# spectra_content.remove("20131008")
spectra_path = [
    folder_to_spectra + spectra_content[i] + "/" for i in range(len(spectra_content))
]

print("spectra path: ", spectra_path)

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

bcvr_arr = [9193.632, 6700.746, -4927.691, -4838.691, 2649.248, 8686.852, 5704.073]
addition = [2000, -2000, -10000, 8000, 0, -17000, -7000]  # addition by eye
na_addition = [-13 * 1000, 0.0, 1 * 1000, -17 * 1000, 0, -9 * 1000, 0]
# rv = -31 * 1000
rv = +45 * 1000
# rv = -45 ? -19 TiO
bcvr_arr = [
    bcvr_arr[x] + addition[x] + rv + na_addition[x] for x in range(len(bcvr_arr))
]
# bcvr_arr = bcvr_arr + addition

print("corrected speed are: ", bcvr_arr)

for i in range(len(rd)):
    _, rd[i][:, 0] = pyasl.dopplerShift(
        rd[i][:, 0], rd[i][:, 1], bcvr_arr[i] / 1000, edgeHandling="firstlast"
    )
    if i != 1:
        plt.plot(rd[i][:, 0], rd[i][:, 1] / np.median(rd[i][:, 1]), label=i)
plt.legend()
plt.show()
