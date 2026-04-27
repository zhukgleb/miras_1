import numpy as np
import PyAstronomy.pyasl as pyasl
from dech_processing import make_txt_from_spectra
import matplotlib.pyplot as plt
import scienceplots


bcvr_arr = [8200.202]  # in ms

wf = "/home/alpha/miras_1/spectra/R_Cam/20111115/"
rd = make_txt_from_spectra(wf, True, True)


_, rd[:, 0] = pyasl.dopplerShift(
    rd[:, 0], rd[:, 1], bcvr_arr[0] / 1000, edgeHandling="firstlast"
)


with plt.style.context("science"):
    fig, ax = plt.subplots()
    ax.plot(rd[:, 0], rd[:, 1])
    plt.show()
