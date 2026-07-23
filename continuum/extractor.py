from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "spectra"))
from dech_processing import make_txt_from_spectra
from scipy.signal import medfilt
from Model_extractor import ModelGridExtractor
from process import (
    fit_parabola,
    split_spectral_orders,
    move_parabola,
    normalize_with_poly,
)

dir_path = os.path.dirname(os.path.realpath(__file__)).replace("continuum", "")
folder_to_spectra = dir_path + "/spectra/R_Cam/"
spectra_content = os.listdir(folder_to_spectra)




# CONFIG
plot=True
save=False
folder_path = "2026-07-20-13-28-24_0.7248514425106289_LTE_synthetic_spectra_parameters"

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
orders = split_spectral_orders(data)
synth_data = np.loadtxt("/home/delta/miras_1/mols/synth_all.spec")



hdu_list = fits.open("e619020c.fits")
hdu_list.info()

image_data = hdu_list[0].data
fit_arr = []

for i in range(len(image_data)):
    x_pixels = np.array([x for x in range(len(image_data[i]))])
    y_pixels = image_data[i]
    y_pixels = y_pixels.astype("float64")
    y_pixels = medfilt(y_pixels, kernel_size=15)
    y_pixels = medfilt(y_pixels, kernel_size=31)
    y_pixels = medfilt(y_pixels, kernel_size=31)
    y_pixels = medfilt(y_pixels, kernel_size=51)

    result = fit_parabola(x_pixels, y_pixels)
    fit_arr.append(result)
    y_smooth = result["func"](x_pixels)


obs_norm = []
obs_norm_p = []
obs_norm_s = []

for order in range(len(orders) - 1):
    x1 = np.mean(orders[order][:, 0])
    print(f"Center of order {order} is {x1} AA")
    a, b, c, yn, yn_p = (
        fit_arr[order]["a"],
        fit_arr[order]["b"],
        fit_arr[order]["c"],
        fit_arr[order]["yn"],
        fit_arr[order]["yn_p"],
    )
    an, bn, cn = move_parabola(a, b, c, 1024, x1)

    # numpy var
    n_target = len(orders[order][:, 0])
    x_original = np.linspace(0, 1, len(yn))
    x_target = np.linspace(0, 1, n_target)
    n_target = len(orders[order][:, 0])
    yn_interpolated = np.interp(x_target, x_original, yn)
    yn_p_interpolated = np.interp(x_target, x_original, yn_p)
    # normy
    mean_in_order = np.mean(
        synth_data[:, 1][
            np.where(
                (synth_data[:, 0] >= min(orders[order][:, 0]))
                & (synth_data[:, 0] <= max(orders[order][:, 0]))
            )
        ]
    )
    y_new = (
        orders[order][:, 1]
        / (yn_interpolated)
        * (mean_in_order)
        / np.mean(orders[order][:, 1])
    )
    y_new_p = (
        orders[order][:, 1]
        / (yn_p_interpolated)
        * (mean_in_order)
        / np.mean(orders[order][:, 1])
    )
    y_new_s = orders[order][:, 1] * mean_in_order / np.mean(orders[order][:, 1])
    obs_norm.append(np.column_stack((orders[order][:, 0], y_new)))
    obs_norm_p.append(np.column_stack((orders[order][:, 0], y_new_p)))
    obs_norm_s.append(np.column_stack((orders[order][:, 0], y_new_s)))


obs_norm = np.concatenate(obs_norm)
obs_norm_p = np.concatenate(obs_norm_p)
obs_norm_s = np.concatenate(obs_norm_s)



extractor = ModelGridExtractor(folder_path)
params_df = extractor.load_parameters()
print("\nПараметры моделей:")
print(params_df.head())
spectra = extractor.load_spectra()
grid = extractor.build_grid()



for key in grid.keys():
    if key == "param_grid":
        pass
    else:
        wl = grid[key]["spectrum"]["wavelength"]
        flux = grid[key]["spectrum"]["flux_norm"]
        params = grid[key]["parameters"]

rep_wave = grid["0.spec"]["spectrum"]["wavelength"]
rep_flux = grid["0.spec"]["spectrum"]["flux_norm"]
flux_arr = []


for key in grid.keys():
    if key == "param_grid":
        pass
    else:
        flux_arr.append(grid[key]["spectrum"]["flux_norm"])

delta_arr = np.array([abs(rep_flux - flux_arr[i]) for i in range(len(flux_arr))])
delta_arr_smart = []


for i in range(len(flux_arr)):
    d_map = []
    for j in range(len(flux_arr[i])):
        if flux_arr[i][j] < 0.5:
            d_map.append(1)
        else:
            d_map.append(abs(rep_flux[j] - flux_arr[i][j]))
    delta_arr_smart.append(d_map)


delta_arr_mean = np.mean(delta_arr, axis=0)
delta_arr_mean_smart = np.mean(delta_arr_smart, axis=0)




p_obs, p_flux, p_flux_new = normalize_with_poly(
    rep_wave,
    rep_flux,
    delta_arr_mean_smart,
    obs_norm_p[:, 0],
    obs_norm_p[:, 1],
    poly_degree=7,
    plot=plot
)

if plot:
    with plt.style.context(["science", "ieee"]):


        fig, ax = plt.subplots()
        ax.plot(obs_norm_p[:, 0], p_obs)
        ax.set_title("Polynorm approximation")

        fig, ax = plt.subplots(figsize=(3, 5))
        ax.plot(x_pixels, y_pixels, ls="--", color="black", alpha=0.9)
        ax.plot(x_pixels, y_smooth, ls="-", color="red")
        ax.set_ylim((6000, 60000))
        ax.set_xlabel("Order pixels")
        ax.set_ylabel("Counts")
        ax.set_yscale("log")

        fig, ax = plt.subplots()
        scatter = ax.scatter(grid["0.spec"]['spectrum']['wavelength'], grid["0.spec"]['spectrum']['flux_norm'], c=delta_arr_mean, cmap='plasma', s=1, alpha=0.5)
        ax.set_title("Delta graph")
        ax.set_xlabel(r"Wavelength, \AA")
        ax.set_ylabel(r"mean delta flux")
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Delta', fontsize=12)

        fig, ax = plt.subplots()
        good_delta_data = np.where(delta_arr_mean < 0.1)
        sc = ax.scatter(grid["0.spec"]['spectrum']['wavelength'][good_delta_data], grid["0.spec"]['spectrum']['flux_norm'][good_delta_data], c=delta_arr_mean[good_delta_data], cmap='plasma', s=1)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label('Delta', fontsize=12)


        fig, ax = plt.subplots()
        for key in grid.keys():
            if key == "param_grid":
                pass
            else:
                wl = grid[key]['spectrum']['wavelength']
                flux = grid[key]['spectrum']['flux_norm']
                params = grid[key]['parameters']
                label = f"{params['specname']}, teff: {params['teff']}, log g: {params['logg']}, [Fe/H] = {params["feh"]}"
                ax.plot(wl, flux, label=label)
                ax.legend()


        fig, ax = plt.subplots(figsize=(4, 2))
        ax.plot(obs_norm[:, 0], obs_norm[:, 1], label="flat-corrected", color="navy")
        ax.plot(obs_norm[:, 0], obs_norm_p[:, 1], label="parabola-corrected", color='crimson')
        ax.plot(obs_norm_s[:, 0], obs_norm_s[:, 1], label="uncorrected", color="black")
        ax.set_xlim((6613, 6685))
        ax.set_ylim((0, 2))
        ax.legend()

        fig, ax = plt.subplots()
        ax.plot(orders[order][:, 0], orders[order][:, 1], label="obs")
        ax.plot(orders[order][:, 0], orders[order][:, 1] * yn_interpolated, label="compensated")
        ax.legend()

        plt.tight_layout()


    plt.show()
