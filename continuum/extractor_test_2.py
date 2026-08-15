from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import os
import sys
from scipy.constants import h, c, k
from scipy.signal import savgol_filter

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "spectra"))
from dech_processing import make_txt_from_spectra
from scipy.signal import medfilt
from Model_extractor import ModelGridExtractor
from process import (
    fit_parabola,
    split_spectral_orders,
    move_parabola,
)

dir_path = os.path.dirname(os.path.realpath(__file__)).replace("continuum", "")
folder_to_spectra = dir_path + "/spectra/R_Cam/"
spectra_content = os.listdir(folder_to_spectra)


# CONFIG
plot = True
save = True
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
order_bound = []

for order in range(len(orders) - 1):
    x1 = np.mean(orders[order][:, 0])
    order_bound.append((min(orders[order][:, 0]), max(orders[order][:, 0])))
    a, b, c, yn, yn_p = (
        fit_arr[order]["a"],
        fit_arr[order]["b"],
        fit_arr[order]["c"],
        fit_arr[order]["yn"],
        fit_arr[order]["yn_p"],
    )
    an, bn, cn = move_parabola(a, b, c, 1024, x1)

    n_target = len(orders[order][:, 0])
    x_original = np.linspace(0, 1, len(yn))
    x_target = np.linspace(0, 1, n_target)
    n_target = len(orders[order][:, 0])
    yn_interpolated = np.interp(x_target, x_original, yn)
    yn_p_interpolated = np.interp(x_target, x_original, yn_p)
    
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

# Интерполируем delta_arr_mean на длины волн obs_norm
delta_interp = np.interp(obs_norm[:, 0], rep_wave, delta_arr_mean, left=1.0, right=1.0)


# ==================== НОВАЯ ЧАСТЬ: РАБОТА С ZrO ====================

def planck_function(wavelength, temperature):
    """
    Функция Планка для черного тела
    """
    wl_m = wavelength * 1e-10
    with np.errstate(divide='ignore', invalid='ignore'):
        intensity = (2 * h * c**2) / (wl_m**5) / (np.exp(h * c / (wl_m * k * temperature)) - 1)
        intensity = np.nan_to_num(intensity, nan=0.0, posinf=0.0)
    return intensity


def normalize_molecular_spectrum(mol_wave, mol_cross_section, temp=1500, column_density=1e16):
    """
    Нормировка молекулярного спектра на черное тело
    """
    optical_depth = column_density * mol_cross_section
    transmission = np.exp(-optical_depth)
    bb_intensity = planck_function(mol_wave, temp)
    normalized_spectrum = transmission  # bb_intensity * transmission / bb_intensity
    normalized_spectrum = np.nan_to_num(normalized_spectrum, nan=1.0, posinf=1.0)
    return normalized_spectrum, transmission, bb_intensity


# Загрузка кросс-секции ZrO
mol_data = np.genfromtxt("/home/delta/exocross/input/ZrO_all.xsec")
molecular_wave = 1e8 / mol_data[:, 0][::-1]
molecular_cross_section = mol_data[:, 1][::-1]

# Параметры для ZrO
T_zro = 1500
column_density_zro = 1e16

# Получение нормированного спектра ZrO
zro_norm_spectrum, zro_transmission, zro_bb = normalize_molecular_spectrum(
    molecular_wave, 
    molecular_cross_section, 
    T_zro, 
    column_density_zro
)

# Интерполяция на длины волн obs_norm
zro_norm_obs_interp = np.interp(obs_norm[:, 0], molecular_wave, zro_norm_spectrum, left=1.0, right=1.0)

# ==================== КОРРЕКЦИЯ С ИСПОЛЬЗОВАНИЕМ DELTA ====================

def find_continuum_with_delta_weights(obs_wave, obs_flux, zro_spectrum, delta_array,
                                       window_size=100, delta_threshold=0.1):
    """
    Находит континуум с использованием весов на основе delta_arr_mean
    
    Веса:
    - Чем меньше delta, тем больше вес точки (она более надежна)
    - Точки с delta > delta_threshold получают малый вес
    - Точки с сильным молекулярным поглощением (zro_spectrum < 0.8) также имеют малый вес
    """
    n_points = len(obs_wave)
    continuum = np.zeros(n_points)
    weights = np.ones(n_points)
    
    # Вычисляем веса для каждой точки
    # Вес = exp(-delta / delta_threshold) * (zro_spectrum > 0.8)
    weight_delta = np.exp(-delta_array / delta_threshold)
    weight_zro = (zro_spectrum > 0.8).astype(float)
    
    # Комбинированный вес
    weights = weight_delta * weight_zro
    
    # Дополнительно: точки с очень маленьким delta получают больший вес
    # а точки с большим delta - меньший
    weights = np.clip(weights, 0.01, 1.0)
    
    # Нормализуем веса для устойчивости
    weights = weights / np.max(weights)
    
    # Находим континуум с использованием взвешенного скользящего окна
    half_window = window_size // 2
    
    for i in range(n_points):
        start = max(0, i - half_window)
        end = min(n_points, i + half_window)
        
        # Взвешенные значения в окне
        window_flux = obs_flux[start:end]
        window_weights = weights[start:end]
        
        if np.sum(window_weights) > 0:
            # Взвешенный перцентиль (берем 90-й перцентиль с весами)
            # Сортируем по значению flux
            sorted_indices = np.argsort(window_flux)
            sorted_flux = window_flux[sorted_indices]
            sorted_weights = window_weights[sorted_indices]
            
            # Кумулятивная сумма весов
            cumsum_weights = np.cumsum(sorted_weights)
            total_weight = cumsum_weights[-1]
            
            # Находим 90-й перцентиль
            target_weight = 0.9 * total_weight
            idx = np.searchsorted(cumsum_weights, target_weight)
            if idx < len(sorted_flux):
                continuum[i] = sorted_flux[idx]
            else:
                continuum[i] = sorted_flux[-1]
        else:
            # Если нет весов, берем обычный максимум
            continuum[i] = np.percentile(window_flux, 90)
    
    return continuum, weights


def fit_continuum_with_weights(obs_wave, obs_flux, continuum_initial, weights,
                                poly_order=3, n_iter=3):
    """
    Итеративная подгонка континуума с весами
    """
    n_points = len(obs_wave)
    continuum = continuum_initial.copy()
    
    for iteration in range(n_iter):
        # Нормируем спектр на текущий континуум
        normalized = obs_flux / continuum
        
        # Находим точки, где нормированный спектр близок к 1
        # Используем веса: точки с большим весом и близкие к 1
        good_mask = (np.abs(normalized - 1.0) < 0.15) & (weights > 0.5)
        
        if np.sum(good_mask) < 10:
            # Если слишком мало точек, ослабляем критерии
            good_mask = (np.abs(normalized - 1.0) < 0.25) & (weights > 0.3)
        
        if np.sum(good_mask) > 5:
            # Взвешенная полиномиальная подгонка
            # Используем только точки с хорошими весами
            x_fit = obs_wave[good_mask]
            y_fit = obs_flux[good_mask] / (zro_norm_obs_interp[good_mask] + 1e-10)
            w_fit = weights[good_mask]
            
            # Взвешенный полиномиальный фит
            # Для простоты используем обычный polyfit с весами
            coeffs = np.polyfit(x_fit, y_fit, poly_order, w=w_fit)
            continuum_new = np.polyval(coeffs, obs_wave)
            
            # Проверяем, что континуум не уходит ниже спектра
            continuum_new = np.maximum(continuum_new, obs_flux * 1.01)
            
            # Сглаживаем резкие скачки
            if iteration > 0:
                continuum_new = savgol_filter(continuum_new, 51, 2)
            
            # Обновляем континуум
            continuum = continuum_new
            
            print(f"Iteration {iteration}: {np.sum(good_mask)} points used, "
                  f"median continuum = {np.median(continuum):.3f}")
        else:
            print(f"Iteration {iteration}: not enough good points, stopping")
            break
    
    return continuum


# Применяем коррекцию с учетом delta и ZrO
print("\n=== Поиск континуума с весами ===")
print(f"Диапазон delta: {np.min(delta_interp):.3f} - {np.max(delta_interp):.3f}")
print(f"Медиана delta: {np.median(delta_interp):.3f}")

# Шаг 1: Находим начальный континуум с весами
continuum_initial, weights = find_continuum_with_delta_weights(
    obs_norm[:, 0], 
    obs_norm[:, 1], 
    zro_norm_obs_interp,
    delta_interp,
    window_size=100,
    delta_threshold=0.1
)

# Шаг 2: Итеративная подгонка континуума
continuum_final = fit_continuum_with_weights(
    obs_norm[:, 0],
    obs_norm[:, 1],
    continuum_initial,
    weights,
    poly_order=3,
    n_iter=3
)

# Шаг 3: Финальная нормировка
obs_norm_corrected = obs_norm[:, 1] / continuum_final

# Дополнительная коррекция: нормируем на медиану в областях с хорошим весом
good_weight_mask = weights > 0.7
if np.sum(good_weight_mask) > 10:
    median_flux = np.median(obs_norm_corrected[good_weight_mask])
    if abs(median_flux - 1.0) > 0.05:
        print(f"Коррекция медианы: {median_flux:.3f} -> 1.0")
        obs_norm_corrected = obs_norm_corrected / median_flux
        continuum_final = continuum_final / median_flux

# Создаем исправленный спектр
obs_norm_zro = np.column_stack((obs_norm[:, 0], obs_norm_corrected))

# ==================== ВИЗУАЛИЗАЦИЯ ====================

if plot:
    with plt.style.context(["science"]):
        
        # 1. Показываем веса и delta
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        ax = axes[0, 0]
        ax.plot(obs_norm[:, 0], delta_interp, color='purple', alpha=0.7)
        ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='threshold')
        ax.set_xlabel(r'Wavelength, $\AA$')
        ax.set_ylabel('Delta')
        ax.set_title('Delta array (model difference)')
        ax.legend()
        ax.set_yscale('log')
        
        ax = axes[0, 1]
        ax.plot(obs_norm[:, 0], weights, color='green', alpha=0.7)
        ax.set_xlabel(r'Wavelength, $\AA$')
        ax.set_ylabel('Weight')
        ax.set_title('Weights from delta and ZrO')
        ax.set_ylim(0, 1.1)
        
        ax = axes[1, 0]
        ax.plot(obs_norm[:, 0], obs_norm[:, 1], label='Original', alpha=0.5, color='navy')
        ax.plot(obs_norm[:, 0], continuum_final, label='Continuum (weighted)', color='red', linewidth=2)
        ax.plot(obs_norm[:, 0], continuum_initial, label='Initial continuum', color='orange', alpha=0.5, linestyle='--')
        ax.set_xlabel(r'Wavelength, $\AA$')
        ax.set_ylabel('Flux')
        ax.set_title('Continuum fitting with weights')
        ax.legend()
        ax.set_xlim(6000, 7000)
        
        ax = axes[1, 1]
        ax.scatter(obs_norm[:, 0][weights > 0.7], obs_norm[:, 1][weights > 0.7], 
                   c='green', s=1, alpha=0.5, label='High weight')
        ax.scatter(obs_norm[:, 0][weights <= 0.7], obs_norm[:, 1][weights <= 0.7], 
                   c='gray', s=1, alpha=0.3, label='Low weight')
        ax.plot(obs_norm[:, 0], continuum_final, color='red', linewidth=2, label='Continuum')
        ax.set_xlabel(r'Wavelength, $\AA$')
        ax.set_ylabel('Flux')
        ax.set_title('Points colored by weight')
        ax.legend()
        ax.set_xlim(6000, 7000)
        
        plt.tight_layout()
        
        # 2. Результат нормировки
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        ax = axes[0]
        ax.plot(obs_norm[:, 0], obs_norm[:, 1], label='Before correction', alpha=0.5, color='navy')
        ax.plot(obs_norm_zro[:, 0], obs_norm_zro[:, 1], label='After correction', color='crimson', alpha=0.7)
        ax.plot(obs_norm[:, 0], continuum_final, label='Continuum', color='green', linestyle='--', alpha=0.5)
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlim(6000, 7000)
        ax.set_ylim(0, 1.5)
        ax.set_xlabel(r'Wavelength, $\AA$')
        ax.set_ylabel('Normalized flux')
        ax.legend()
        ax.set_title('Full spectrum correction with ZrO + delta weights')
        
        ax = axes[1]
        ax.plot(obs_norm[:, 0], obs_norm[:, 1], label='Original', alpha=0.5, color='navy')
        ax.plot(obs_norm_zro[:, 0], obs_norm_zro[:, 1], label='Corrected', color='crimson', alpha=0.7)
        ax.plot(obs_norm[:, 0], zro_norm_obs_interp, label='ZrO template', color='orange', alpha=0.5)
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlim(6450, 6550)
        ax.set_ylim(0.2, 1.3)
        ax.set_xlabel(r'Wavelength, $\AA$')
        ax.set_ylabel('Normalized flux')
        ax.legend()
        ax.set_title('Zoom in ZrO band region')
        
        plt.tight_layout()
        plt.show()

# Сохранение результата
if save:
    # Сохраняем все важные данные
    np.savetxt("obs_norm_zro_corrected.txt", obs_norm_zro, 
               header="wavelength flux_norm_zro_corrected")
    np.savetxt("continuum_fit.txt", np.column_stack((obs_norm[:, 0], continuum_final)), 
               header="wavelength continuum")
    np.savetxt("weights.txt", np.column_stack((obs_norm[:, 0], weights, delta_interp)), 
               header="wavelength weight delta")
    np.savetxt("zro_normalized.txt", np.column_stack((molecular_wave, zro_norm_spectrum)), 
               header="wavelength zro_norm")
    np.savetxt("zro_transmission.txt", np.column_stack((molecular_wave, zro_transmission)), 
               header="wavelength zro_transmission")
    
    print("\nФайлы сохранены:")
    print("  - obs_norm_zro_corrected.txt (нормированный спектр)")
    print("  - continuum_fit.txt (найденный континуум)")
    print("  - weights.txt (веса и delta для каждой точки)")
    print("  - zro_normalized.txt (нормированный спектр ZrO)")
    print("  - zro_transmission.txt (пропускание ZrO)")
