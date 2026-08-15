import numpy as np
import matplotlib.pyplot as plt
import PyAstronomy.pyasl as pyasl

tio_data = np.genfromtxt("/home/delta/exocross/input/TiO_all.xsec")
tio_wave = 1e8 / tio_data[:, 0][::-1]
tio_cross_section = tio_data[:, 1][::-1]

tio_data_kur = np.genfromtxt("TiO_Kurucz_2000.xsec")
tio_wave_kur = 1e8 / tio_data_kur[:, 0][::-1]
tio_cross_section_kur = tio_data_kur[:, 1][::-1]

v = 0

tio_cross_section, tio_wave = pyasl.dopplerShift(
    tio_wave, tio_cross_section, v, edgeHandling="firstlast"
)

tio_cross_section_kur, tio_wave_kur = pyasl.dopplerShift(
    tio_wave_kur,
    tio_cross_section_kur,
    v,
    edgeHandling="firstlast",
)

data_per = [
    {"wavelength": 4626.983, "num_lines": 12, "strength_2000K": 2.66},
    {"wavelength": 4760.885, "num_lines": 12, "strength_2000K": 3.14},
    {"wavelength": 4763.935, "num_lines": 12, "strength_2000K": 3.08},
    {"wavelength": 4804.558, "num_lines": 12, "strength_2000K": 2.96},
    {"wavelength": 4807.490, "num_lines": 13, "strength_2000K": 2.91},
    {"wavelength": 4849.334, "num_lines": 12, "strength_2000K": 2.51},
    {"wavelength": 4851.949, "num_lines": 12, "strength_2000K": 2.50},
    {"wavelength": 4954.454, "num_lines": 13, "strength_2000K": 3.47},
    {"wavelength": 4956.791, "num_lines": 13, "strength_2000K": 3.46},
    {"wavelength": 4998.882, "num_lines": 13, "strength_2000K": 2.91},
    {"wavelength": 5002.232, "num_lines": 13, "strength_2000K": 2.90},
    {"wavelength": 5166.664, "num_lines": 13, "strength_2000K": 3.58},
    {"wavelength": 5169.066, "num_lines": 13, "strength_2000K": 3.56},
    {"wavelength": 5328.310, "num_lines": 21, "strength_2000K": 2.16},
    {"wavelength": 5359.408, "num_lines": 22, "strength_2000K": 2.11},
    {"wavelength": 5391.153, "num_lines": 21, "strength_2000K": 1.95},
    {"wavelength": 5423.564, "num_lines": 21, "strength_2000K": 1.73},
    {"wavelength": 5448.132, "num_lines": 14, "strength_2000K": 3.26},
    {"wavelength": 5450.789, "num_lines": 19, "strength_2000K": 3.21},
    {"wavelength": 5496.605, "num_lines": 14, "strength_2000K": 2.75},
    {"wavelength": 5499.455, "num_lines": 14, "strength_2000K": 2.70},
    {"wavelength": 5598.390, "num_lines": 24, "strength_2000K": 2.04},
    {"wavelength": 5598.400, "num_lines": 24, "strength_2000K": 2.05},
    {"wavelength": 5598.410, "num_lines": 24, "strength_2000K": 3.17},
    {"wavelength": 5598.421, "num_lines": 24, "strength_2000K": 2.17},
    {"wavelength": 5598.432, "num_lines": 24, "strength_2000K": 2.21},
    {"wavelength": 5629.981, "num_lines": 24, "strength_2000K": 2.77},
    {"wavelength": 5662.221, "num_lines": 24, "strength_2000K": 2.37},
    {"wavelength": 5695.146, "num_lines": 23, "strength_2000K": 1.95},
    {"wavelength": 5758.992, "num_lines": 14, "strength_2000K": 2.61},
    {"wavelength": 5761.939, "num_lines": 14, "strength_2000K": 2.57},
    {"wavelength": 5810.096, "num_lines": 14, "strength_2000K": 2.50},
    {"wavelength": 5813.257, "num_lines": 14, "strength_2000K": 2.45},
    {"wavelength": 5847.666, "num_lines": 13, "strength_2000K": 2.65},
    {"wavelength": 5873.424, "num_lines": 29, "strength_2000K": 2.65},
    {"wavelength": 5899.678, "num_lines": 32, "strength_2000K": 2.78},
    {"wavelength": 5950.565, "num_lines": 32, "strength_2000K": 2.62},
    {"wavelength": 6149.359, "num_lines": 32, "strength_2000K": 2.79},
    {"wavelength": 6158.676, "num_lines": 14, "strength_2000K": 3.18},
    {"wavelength": 6186.998, "num_lines": 27, "strength_2000K": 3.17},
    {"wavelength": 6215.558, "num_lines": 33, "strength_2000K": 3.31},
    {"wavelength": 6239.693, "num_lines": 30, "strength_2000K": 2.51},
    {"wavelength": 6268.860, "num_lines": 32, "strength_2000K": 2.65},
    {"wavelength": 6357.332, "num_lines": 15, "strength_2000K": 2.68},
    {"wavelength": 6384.175, "num_lines": 15, "strength_2000K": 2.68},
    {"wavelength": 6420.615, "num_lines": 15, "strength_2000K": 2.57},
    {"wavelength": 6447.901, "num_lines": 16, "strength_2000K": 2.56},
    {"wavelength": 6626.460, "num_lines": 34, "strength_2000K": 2.51},
    {"wavelength": 6651.178, "num_lines": 10, "strength_2000K": 3.26},
]


fig, ax = plt.subplots()
ax.plot(tio_wave, tio_cross_section, color="black", label="stick spectra ToTo")
ax.set_ylim((0 * 10**-17, 4.3 * 10**-16))
ax.plot(tio_wave_kur, tio_cross_section_kur, color="red", label="stick spectra Kurucz")
ax.legend()
for i in range(len(data_per)):
    x = float(data_per[i].get("wavelength"))
    y = float(data_per[i].get("strength_2000K"))
    ax.plot([x, x], [0, y])
plt.show()
