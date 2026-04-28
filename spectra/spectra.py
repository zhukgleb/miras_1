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
spectra_path = [
    folder_to_spectra + spectra_content[i] for i in range(len(spectra_content))
]

print(spectra_path)
# rd = make_txt_from_spectra(wf, True, True)


def test():
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

    rd[:, 0] = pyasl.dopplerShift(
        rd[:, 0], rd[:, 1], bcvr_arr[0] / 1000, edgeHandling="firstlast"
    )

    with plt.style.context("science"):
        fig, ax = plt.subplots()
        ax.plot(rd[:, 0], rd[:, 1])
        plt.show()
