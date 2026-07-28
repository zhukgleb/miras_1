from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import os
import sys
from scipy.constants import h, c, k

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


# ==================== НОВАЯ ЧАСТЬ: РАБОТА С ZrO ====================

def planck_function(wavelength, temperature):
    """
    Функция Планка для черного тела
    """
    wl_m = wavelength * 1e-10  # перевод в метры
    intensity = (2 * h * c**2) / (wl_m**5) / (np.exp(h * c / (wl_m * k * temperature)) - 1)
    return intensity


def normalize_molecular_spectrum(mol_wave, mol_cross_section, temp=1500, column_density=1e16):
    """
    Нормировка молекулярного спектра на черное тело
    """
    # Оптическая толщина: tau = N * sigma
    optical_depth = column_density * mol_cross_section
    
    # Интенсивность черного тела
    bb_intensity = planck_function(mol_wave, temp)
    
    # Поток с учетом поглощения: I = I0 * exp(-tau)
    transmitted_intensity = bb_intensity * np.exp(-optical_depth)
    
    # Нормировка на континуум
    normalized_spectrum = transmitted_intensity / bb_intensity
    
    return normalized_spectrum, optical_depth, bb_intensity


# Загрузка кросс-секции ZrO
mol_data = np.genfromtxt("/home/delta/exocross/input/ZrO_all.xsec")
molecular_wave = 1e8 / mol_data[:, 0][::-1]
molecular_cross_section = mol_data[:, 1][::-1]

# Параметры для ZrO
T_zro = 1500
column_density_zro = 1e16

# Получение нормированного спектра ZrO
zro_norm_spectrum, zro_optical_depth, zro_bb = normalize_molecular_spectrum(
    molecular_wave, 
    molecular_cross_section, 
    T_zro, 
    column_density_zro
)

# Интерполяция молекулярного спектра на длины волн obs_norm
zro_norm_obs_interp = np.interp(obs_norm[:, 0], molecular_wave, zro_norm_spectrum, left=1.0, right=1.0)

# ==================== ИСПРАВЛЕННАЯ НОРМИРОВКА С ИСПОЛЬЗОВАНИЕМ DELTA ====================

def normalize_with_delta_and_zro(obs_wave, obs_flux, zro_spectrum, 
                                  rep_wave, rep_flux, delta_arr_mean,
                                  delta_threshold=0.1, 
                                  zro_threshold=0.02,
                                  poly_order=3,
                                  n_iter=3):
    """
    Нормировка спектра с использованием:
    1. Массива delta для определения доверенных точек (где модели близки к реперу)
    2. Молекулярного спектра ZrO для учета поглощения
    3. Итеративной подгонки континуума с принудительным ограничением flux <= 1
    """
    
    # Интерполяция delta на длины волн obs
    delta_interp = np.interp(obs_wave, rep_wave, delta_arr_mean, left=1.0, right=1.0)
    
    # Интерполяция реперного спектра
    rep_interp = np.interp(obs_wave, rep_wave, rep_flux, left=1.0, right=1.0)
    
    # СОЗДАНИЕ МАСКИ ДОВЕРЕННЫХ ТОЧЕК
    # 1. Точки, где delta мала (модели близки к реперу)
    good_delta_mask = delta_interp < delta_threshold
    
    # 2. Точки, где молекулярное поглощение слабое
    good_zro_mask = np.abs(zro_spectrum - 1.0) < zro_threshold
    
    # 3. Точки, где реперный спектр близок к континууму (нет сильных линий)
    good_rep_mask = rep_interp > 0.95
    
    # 4. Объединенная маска - точки, которым можно доверять
    trust_mask = good_delta_mask & good_zro_mask & good_rep_mask
    
    print(f"Доверенных точек: {np.sum(trust_mask)} из {len(obs_wave)}")
    
    if np.sum(trust_mask) < 20:
        print("Предупреждение: слишком мало доверенных точек, расширяем критерии")
        trust_mask = good_delta_mask | (good_zro_mask & good_rep_mask)
        print(f"Доверенных точек после расширения: {np.sum(trust_mask)}")
    
    # ПЕРВАЯ ИТЕРАЦИЯ: подгонка континуума по доверенным точкам
    x_fit = obs_wave[trust_mask]
    y_fit = obs_flux[trust_mask]
    
    # Подгонка полинома
    if len(x_fit) > poly_order:
        coeffs = np.polyfit(x_fit, y_fit, poly_order)
        continuum = np.polyval(coeffs, obs_wave)
    else:
        # Если слишком мало точек, используем медиану
        continuum = np.median(obs_flux[trust_mask]) * np.ones_like(obs_wave)
    
    # Нормировка
    normalized = obs_flux / continuum
    
    # Принудительное ограничение: убираем значения > 1
    # Находим точки, где normalized > 1, и корректируем континуум
    mask_above_one = normalized > 1.0
    
    if np.sum(mask_above_one) > 0:
        # Для точек выше 1, поднимаем континуум
        # Используем итеративный подход
        for iter_num in range(n_iter):
            # Находим точки, где normalized > 1.01 (с небольшим запасом)
            mask_high = normalized > 1.01
            
            if np.sum(mask_high) == 0:
                break
            
            # Добавляем эти точки к доверенным с весом, обратным отклонению
            high_wave = obs_wave[mask_high]
            high_flux = obs_flux[mask_high]
            
            # Объединяем с доверенными точками
            all_wave = np.concatenate([x_fit, high_wave])
            all_flux = np.concatenate([y_fit, high_flux])
            
            # Переподгонка
            if len(all_wave) > poly_order:
                coeffs = np.polyfit(all_wave, all_flux, poly_order)
                continuum_new = np.polyval(coeffs, obs_wave)
            else:
                continuum_new = continuum
            
            # Обновляем нормировку
            normalized_new = obs_flux / continuum_new
            
            # Проверяем, уменьшилось ли количество точек > 1
            if np.sum(normalized_new > 1.01) < np.sum(mask_high):
                continuum = continuum_new
                normalized = normalized_new
            else:
                break
    
    # Финальная коррекция: гарантируем, что ни одна точка не превышает 1
    # Для точек, где normalized > 1, используем интерполяцию между соседними точками < 1
    final_normalized = normalized.copy()
    mask_above = final_normalized > 1.0
    
    if np.sum(mask_above) > 0:
        # Находим индексы точек < 1
        indices_below = np.where(~mask_above)[0]
        
        if len(indices_below) > 1:
            # Для каждой точки > 1, интерполируем по соседним
            for i in np.where(mask_above)[0]:
                # Находим ближайшие точки слева и справа с flux < 1
                left_idx = indices_below[indices_below < i]
                right_idx = indices_below[indices_below > i]
                
                if len(left_idx) > 0 and len(right_idx) > 0:
                    left = left_idx[-1]
                    right = right_idx[0]
                    # Линейная интерполяция
                    weight = (i - left) / (right - left)
                    final_normalized[i] = (1 - weight) * final_normalized[left] + weight * final_normalized[right]
                elif len(left_idx) > 0:
                    final_normalized[i] = final_normalized[left_idx[-1]]
                elif len(right_idx) > 0:
                    final_normalized[i] = final_normalized[right_idx[0]]
                else:
                    final_normalized[i] = 0.99
    
    return final_normalized, continuum, trust_mask, delta_interp


# Применяем исправленную нормировку
obs_norm_corrected, continuum_obs, trust_mask, delta_interp = normalize_with_delta_and_zro(
    obs_norm[:, 0],
    obs_norm[:, 1],
    zro_norm_obs_interp,
    rep_wave,
    rep_flux,
    delta_arr_mean,
    delta_threshold=0.1,
    zro_threshold=0.02,
    poly_order=3,
    n_iter=5
)

# Проверяем, что все значения <= 1
print(f"Максимальное значение после нормировки: {np.max(obs_norm_corrected)}")
print(f"Минимальное значение после нормировки: {np.min(obs_norm_corrected)}")
print(f"Доля точек > 1: {np.sum(obs_norm_corrected > 1.0) / len(obs_norm_corrected) * 100:.2f}%")

# Если все еще есть точки > 1, принудительно ограничиваем
obs_norm_corrected = np.clip(obs_norm_corrected, 0.0, 1.0)

# Создаем исправленный спектр
obs_norm_zro = np.column_stack((obs_norm[:, 0], obs_norm_corrected))

# ==================== ВИЗУАЛИЗАЦИЯ ====================

if plot:
    with plt.style.context(["science"]):
        
        # 1. Маска доверенных точек
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        ax = axes[0]
        ax.plot(obs_norm[:, 0], delta_interp, color='blue', alpha=0.7)
        ax.axhline(y=0.1, color='red', linestyle='--', label='delta threshold')
        ax.set_xlabel(r'Wavelength, $\AA$')
        ax.set_ylabel('Delta')
        ax.set_title('Delta array (model differences from reference)')
        ax.legend()
        
        ax = axes[1]
        ax.plot(obs_norm[:, 0], zro_norm_obs_interp, color='darkred', alpha=0.7)
        ax.axhline(y=0.98, color='green', linestyle='--', label='zro threshold')
        ax.set_xlabel(r'Wavelength, $\AA$')
        ax.set_ylabel('ZrO normalized flux')
        ax.set_title('ZrO molecular spectrum')
        ax.legend()
        
        ax = axes[2]
        ax.scatter(obs_norm[:, 0][trust_mask], obs_norm[:, 1][trust_mask], 
                  s=1, color='green', alpha=0.5, label='Trusted points')
        ax.plot(obs_norm[:, 0], obs_norm[:, 1], color='navy', alpha=0.3, label='Original')
        ax.plot(obs_norm[:, 0], continuum_obs, color='red', linestyle='--', label='Continuum fit')
        ax.set_xlabel(r'Wavelength, $\AA$')
        ax.set_ylabel('Flux')
        ax.set_title('Trusted points for continuum fitting')
        ax.legend()
        
        plt.tight_layout()
        
        # 2. Сравнение спектров
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        
        # Полный спектр
        ax = axes[0]
        ax.plot(obs_norm[:, 0], obs_norm[:, 1], label='Before correction', color='navy', alpha=0.5)
        ax.plot(obs_norm_zro[:, 0], obs_norm_zro[:, 1], label='After correction', color='crimson', alpha=0.7)
        ax.plot(obs_norm[:, 0], continuum_obs, label='Continuum fit', color='green', linestyle='--')
        ax.axhline(y=1.0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlim(6000, 7500)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel(r'Wavelength, $\AA$')
        ax.set_ylabel('Normalized flux')
        ax.legend()
        ax.set_title('Full spectrum comparison')
        
        # Область с сильным молекулярным поглощением
        ax = axes[1]
        ax.plot(obs_norm[:, 0], obs_norm[:, 1], label='Before correction', color='navy', alpha=0.5)
        ax.plot(obs_norm_zro[:, 0], obs_norm_zro[:, 1], label='After correction', color='crimson', alpha=0.7)
        ax.plot(obs_norm[:, 0], continuum_obs, label='Continuum fit', color='green', linestyle='--')
        ax.axhline(y=1.0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlim(6450, 6650)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel(r'Wavelength, $\AA$')
        ax.set_ylabel('Normalized flux')
        ax.legend()
        ax.set_title('Zoom: ZrO band region (6450-6650 $\AA$)')
        
        # Другая область с поглощением
        ax = axes[2]
        ax.plot(obs_norm[:, 0], obs_norm[:, 1], label='Before correction', color='navy', alpha=0.5)
        ax.plot(obs_norm_zro[:, 0], obs_norm_zro[:, 1], label='After correction', color='crimson', alpha=0.7)
        ax.plot(obs_norm[:, 0], continuum_obs, label='Continuum fit', color='green', linestyle='--')
        ax.axhline(y=1.0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlim(7000, 7200)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel(r'Wavelength, $\AA$')
        ax.set_ylabel('Normalized flux')
        ax.legend()
        ax.set_title('Zoom: 7000-7200 $\AA$ region')
        
        plt.tight_layout()
        
        # 3. Гистограмма распределения
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(obs_norm_corrected, bins=50, alpha=0.7, color='crimson', edgecolor='black')
        ax.axvline(x=1.0, color='red', linestyle='--', label='Continuum')
        ax.axvline(x=np.median(obs_norm_corrected), color='blue', linestyle='--', 
                  label=f'Median: {np.median(obs_norm_corrected):.3f}')
        ax.set_xlabel('Normalized flux')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of normalized flux values')
        ax.legend()
        
        plt.tight_layout()
        plt.show()

# Сохранение результата
if save:
    # Сохраняем исправленный спектр
    np.savetxt("obs_norm_zro_corrected.txt", obs_norm_zro, 
               header="wavelength flux_norm_zro_corrected")
    
    # Сохраняем континуум
    np.savetxt("continuum_fit.txt", np.column_stack((obs_norm[:, 0], continuum_obs)), 
               header="wavelength continuum")
    
    # Сохраняем маску доверенных точек
    np.savetxt("trust_mask.txt", np.column_stack((obs_norm[:, 0], trust_mask.astype(int))),
               header="wavelength trust_mask")
    
    # Сохраняем нормированный спектр ZrO
    np.savetxt("zro_normalized.txt", np.column_stack((molecular_wave, zro_norm_spectrum)), 
               header="wavelength zro_norm")
    
    # Сохраняем информацию о нормировке
    with open("normalization_info.txt", "w") as f:
        f.write(f"Number of trusted points: {np.sum(trust_mask)} out of {len(obs_norm)}\n")
        f.write(f"Max normalized flux: {np.max(obs_norm_corrected):.6f}\n")
        f.write(f"Min normalized flux: {np.min(obs_norm_corrected):.6f}\n")
        f.write(f"Median normalized flux: {np.median(obs_norm_corrected):.6f}\n")
        f.write(f"Points above 1: {np.sum(obs_norm_corrected > 1.0)}\n")
        f.write(f"ZrO temperature: {T_zro} K\n")
        f.write(f"ZrO column density: {column_density_zro:.0e} cm^-2\n")
