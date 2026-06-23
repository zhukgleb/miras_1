import numpy as np


def get_spectra_cut(start_wl, end_wl, data):
    delta = data[:, 0][1] - data[:, 0][0]

    cut_data = []
    start_idx = np.argmin(np.abs(data[:, 0] - start_wl))
    end_idx = np.argmin(np.abs(data[:, 0] - end_wl))

    cut_data.append(data[int(start_idx) : int(end_idx)])
    return cut_data
