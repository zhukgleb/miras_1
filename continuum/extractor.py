from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from scipy.optimize import curve_fit
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "spectra"))
from dech_processing import make_txt_from_spectra
from general_processing import median_normalization, full_pipeline, normalize_orders_median
from typing import List
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from Model_extractor import ModelGridExtractor


dir_path = os.path.dirname(os.path.realpath(__file__)).replace("continuum", "")
folder_to_spectra = dir_path + "/spectra/R_Cam/"
spectra_content = os.listdir(folder_to_spectra)



def fit_parabola(x, y):
    def parabola(x, a, b, c):
        return a * x**2 + b * x + c
    
    popt, pcov = curve_fit(parabola, x, y)
    a, b, c = popt
    
    y_fit = parabola(x, a, b, c)
    
    residuals = y - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    xb = -b / (2*a)
    yb = a * xb**2 + b * xb + c
    yn = y / yb
    yn_p = y_fit / yb
    
    return {
        'a': a,
        'b': b,
        'c': c,
        'r2': r2,
        'y_fit': y_fit,
        'func': lambda x_val: parabola(x_val, a, b, c),
        'yn': yn,
        'yn_p': yn_p
    }


def split_spectral_orders(data: np.ndarray, zero_threshold: float = 1e-6) -> List[np.ndarray]:

    if data.shape[1] != 2:
        raise ValueError("Shit")
    
    wavelengths = data[:, 0]
    intensities = data[:, 1]
    
    non_zero_mask = np.abs(intensities) > zero_threshold
    
    if not np.any(non_zero_mask):
        return []
    
    padded_mask = np.concatenate([[False], non_zero_mask])
    starts = np.where(padded_mask[1:] & ~padded_mask[:-1])[0]
    
    ends = np.where(~padded_mask[1:] & padded_mask[:-1])[0]
    
    if len(starts) > len(ends):
        ends = np.append(ends, len(data))
    elif len(ends) > len(starts):
        starts = np.insert(starts, 0, 0)
    
    orders = []
    for start, end in zip(starts, ends):
        if start < end: 
            order = data[start:end, :]
            orders.append(order)
    
    return orders


def parabola(x, a, b, c):
    return a * x**2 + b * x + c

def move_parabola(a, b, c, x0, x1):

    a_new = a
    
    x_vertex_old = -b / (2 * a)
    y_vertex_old = c - b**2 / (4 * a)
    
    x_vertex_new = x_vertex_old + (x1 - x0)
    
    b_new = -2 * a * x_vertex_new
    c_new = y_vertex_old - a * x_vertex_new**2
    
    return a_new, b_new, c_new

def normalize_with_poly(model_wave, model_flux, model_diff, 
                        obs_wave, obs_flux, 
                        threshold=0.1, poly_degree=3, 
                        sigma_clip=3.0, plot=True):
    
    # 1. Реперные точки
    mask = model_diff < threshold
    if np.sum(mask) < 5:
        mask = model_diff < threshold * 1.5
        print(f"Used relaxed threshold: {threshold * 1.5:.2f}")
    
    x_fit = model_wave[mask]
    y_fit = obs_flux[mask] / model_flux[mask]
    
    # 2. Отбрасываем выбросы по Y
    median_y = np.median(y_fit)
    std_y = np.std(y_fit)
    clip_mask = np.abs(y_fit - median_y) < sigma_clip * std_y
    x_fit = x_fit[clip_mask]
    y_fit = y_fit[clip_mask]
    
    print(f"Using {len(x_fit)} reference points for polynomial fit")
    
    # 3. Подгонка полинома
    coeffs = np.polyfit(x_fit, y_fit, poly_degree)
    poly_func = np.poly1d(coeffs)
    
    # 4. Визуализация
    if plot:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(x_fit, y_fit, 'bo', markersize=4, label='Reference ratios')
        x_smooth = np.linspace(min(x_fit), max(x_fit), 200)
        plt.plot(x_smooth, poly_func(x_smooth), 'r-', linewidth=2, 
                 label=f'Polynomial (deg={poly_degree})')
        plt.xlabel('Wavelength')
        plt.ylabel('Observed / Model')
        plt.legend()
        plt.grid(True)
        plt.title('Polynomial fit to reference points')
        
        plt.subplot(1, 2, 2)
        residuals = y_fit - poly_func(x_fit)
        plt.plot(x_fit, residuals, 'go', markersize=4)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Wavelength')
        plt.ylabel('Residuals')
        plt.grid(True)
        plt.title(f'Residuals (RMS = {np.std(residuals):.4f})')
        
        plt.tight_layout()
        plt.show()
    
    # 5. Применяем ко всему спектру
    poly_full = poly_func(obs_wave)
    normalized_obs = obs_flux / poly_full
    
    return normalized_obs, poly_func

hdu_list = fits.open("e619020c.fits")
hdu_list.info()

image_data = hdu_list[0].data
fit_arr = []

with plt.style.context(["science", "ieee"]):
    fig, ax = plt.subplots(figsize=(3, 5))
    for i in range(len(image_data)):
        x_pixels = np.array([x for x in range(len(image_data[i]))])
        y_pixels = image_data[i]
        y_pixels = y_pixels.astype('float64')
        y_pixels = medfilt(y_pixels, kernel_size=15)
        y_pixels = medfilt(y_pixels, kernel_size=31)
        y_pixels = medfilt(y_pixels, kernel_size=31)
        y_pixels = medfilt(y_pixels, kernel_size=51)


        result = fit_parabola(x_pixels, y_pixels)
        fit_arr.append(result)
        y_smooth = result['func'](x_pixels)
        ax.plot(x_pixels, y_pixels, ls="--", color="black", alpha=0.9)
        ax.plot(x_pixels, y_smooth, ls="-", color="red")
        ax.set_ylim((6000, 60000))
        ax.set_xlabel("Order pixels")
        ax.set_ylabel("Counts")
        ax.set_yscale("log")


    plt.tight_layout()
    # plt.savefig("orders_normalization.pdf")
    # plt.savefig("orders_normalization.png", dpi=300)
    # plt.show()


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
orders = split_spectral_orders(data)
synth_data = np.loadtxt("/home/delta/miras_1/mols/synth_all.spec")

obs_norm = []
obs_norm_p = []
obs_norm_s = []

for order in range(len(orders)-1):
    x1 = np.mean(orders[order][:, 0])
    print(f"Center of order {order} is {x1} AA")
    a, b, c, yn, yn_p = fit_arr[order]['a'], fit_arr[order]['b'], fit_arr[order]['c'], fit_arr[order]['yn'], fit_arr[order]['yn_p']
    an, bn, cn = move_parabola(a, b, c, 1024, x1)

    # numpy var
    n_target = len(orders[order][:, 0])
    x_original = np.linspace(0, 1, len(yn))
    x_target = np.linspace(0, 1, n_target)
    n_target = len(orders[order][:, 0])
    yn_interpolated = np.interp(x_target, x_original, yn)
    yn_p_interpolated = np.interp(x_target, x_original, yn_p)
    # normy
    mean_in_order = np.mean(synth_data[:, 1][np.where((synth_data[:, 0] >= min(orders[order][:, 0])) & (synth_data[:, 0] <= max(orders[order][:, 0])))])
    y_new = orders[order][:, 1] / (yn_interpolated) * (mean_in_order) / np.mean(orders[order][:, 1])
    y_new_p = orders[order][:, 1] / (yn_p_interpolated) * (mean_in_order) / np.mean(orders[order][:, 1])
    y_new_s = orders[order][:, 1] * mean_in_order / np.mean(orders[order][:, 1])
    obs_norm.append(np.column_stack((orders[order][:, 0], y_new)))
    obs_norm_p.append(np.column_stack((orders[order][:, 0], y_new_p)))
    obs_norm_s.append(np.column_stack((orders[order][:, 0], y_new_s)))

    # fig, ax = plt.subplots()
    # ax.plot(orders[order][:, 0], orders[order][:, 1], label="obs")
    # ax.plot(orders[order][:, 0], orders[order][:, 1] * yn_interpolated, label="compensated")
    # plt.legend()
    # plt.show()



# fig, ax = plt.subplots()
# for i in range(len(obs_norm)):
#     ax.plot(obs_norm[i][:, 0], obs_norm[i][:, 1])
#     ax.plot(obs_norm_s[i][:, 0], obs_norm_s[i][:, 1])


# with plt.style.context(["science", "ieee"]):
#     fig, ax = plt.subplots()
#     obs_norm = np.concatenate(obs_norm)
#     obs_norm_p = np.concatenate(obs_norm_p)
#     obs_norm_s = np.concatenate(obs_norm_s)
#     ax.plot(obs_norm[:, 0], obs_norm[:, 1], label="flat-corrected", color="navy")
#     ax.plot(obs_norm[:, 0], obs_norm_p[:, 1], label="parabola-corrected", color='crimson')
#     ax.plot(obs_norm_s[:, 0], obs_norm_s[:, 1], label="uncorrected", color="black")
#     ax.legend()
#     plt.show()


with plt.style.context(["science", "ieee"]):
    fig, ax = plt.subplots(figsize=(4, 2))
    obs_norm = np.concatenate(obs_norm)
    obs_norm_p = np.concatenate(obs_norm_p)
    obs_norm_s = np.concatenate(obs_norm_s)
    ax.plot(obs_norm[:, 0], obs_norm[:, 1], label="flat-corrected", color="navy")
    ax.plot(obs_norm[:, 0], obs_norm_p[:, 1], label="parabola-corrected", color='crimson')
    ax.plot(obs_norm_s[:, 0], obs_norm_s[:, 1], label="uncorrected", color="black")
    ax.set_xlim((6613, 6685))
    ax.set_ylim((0, 2))
    ax.legend()
    plt.savefig("cut_of_spectra.pdf")
    plt.savefig("cut_of_spectra.png", dpi=300)
    # plt.show()


folder_path = "2026-07-20-13-28-24_0.7248514425106289_LTE_synthetic_spectra_parameters"
    
extractor = ModelGridExtractor(folder_path)    
params_df = extractor.load_parameters()
print("\nПараметры моделей:")
print(params_df.head())
spectra = extractor.load_spectra()
grid = extractor.build_grid()


fig, ax = plt.subplots()
for key in grid.keys():
    if key == "param_grid":
        pass
    else:
        wl = grid[key]['spectrum']['wavelength']
        flux = grid[key]['spectrum']['flux_norm']
        params = grid[key]['parameters']
        label = f"{params['specname']}, teff: {params['teff']}, log g: {params['logg']}, [Fe/H] = {params["feh"]}"

        plt.plot(wl, flux, label=label)
plt.legend()
# plt.show()

