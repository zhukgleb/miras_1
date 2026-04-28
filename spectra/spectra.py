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

for i in range(len(rd)):
    _, rd[i][:, 0] = pyasl.dopplerShift(
        rd[i][:, 0], rd[i][:, 1], bcvr_arr[i] / 1000, edgeHandling="firstlast"
    )

with plt.style.context("science"):
    fig, ax = plt.subplots()
    for i in range(len(rd)):
        ax.plot(rd[i][:, 0], rd[i][:, 1])
    plt.show()
