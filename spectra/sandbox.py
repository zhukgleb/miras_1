import os
from dech_processing import make_txt_from_spectra
from general_processing import median_normalization, full_pipeline, normalize_orders_median
import matplotlib.pyplot as plt
import numpy as np
import PyAstronomy.pyasl as pyasl


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

bcvr_arr = [4450.769, 6698.387, -5093.433, -6032.999, 10471.263, -10221.369, 5778.002]

num = 6
spectra_path = folder_to_spectra + spectra_content[num] + "/"

data = make_txt_from_spectra(spectra_path, True, True)
median_norm = normalize_orders_median(data)
final = full_pipeline(data, degree=2, do_median=True, do_continuum=True)

# fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# for i, (ax, order_data) in enumerate(zip(axes, [data, median_norm, final])):
#     # Разбиваем по нулям
#     zero_idx = np.where(order_data[:, 1] == 0)[0]
#     start = 0
#     for zi in zero_idx:
#         ax.plot(order_data[start:zi, 0], order_data[start:zi, 1], label=f'Order {i+1}')
#         start = zi + 1
#     ax.plot(order_data[start:, 0], order_data[start:, 1])
#     ax.set_title(['Original', 'Median normalized', 'Median + Parabolic continuum'][i])
#     ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()
_, final[:, 0] = pyasl.dopplerShift(final[:, 0], final[:, 1], bcvr_arr[num] / 1000, edgeHandling="firstlast")

np.savetxt(f"/home/delta/miras_1/mols/norm_spectra_{num}.txt", final)