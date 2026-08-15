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
plot = True  # Включим для отладки
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

    Parameters:
    wavelength: длина волны в ангстремах
    temperature: температура в Кельвинах

    Returns:
    интенсивность излучения черного тела
    """
    wl_m = wavelength * 1e-10  # перевод в метры
    # Избегаем деления на ноль
    with np.errstate(divide="ignore", invalid="ignore"):
        intensity = (
            (2 * h * c**2) / (wl_m**5) / (np.exp(h * c / (wl_m * k * temperature)) - 1)
        )
        intensity = np.nan_to_num(intensity, nan=0.0, posinf=0.0)
    return intensity


def normalize_molecular_spectrum(
    mol_wave, mol_cross_section, temp=1500, column_density=1e16
):
    """
    Нормировка молекулярного спектра на черное тело

    Returns:
    normalized_spectrum: нормированный спектр (поток / континуум)
    transmission: пропускание слоя = exp(-tau)
    """
    # Оптическая толщина: tau = N * sigma
    optical_depth = column_density * mol_cross_section

    # Пропускание слоя
    transmission = np.exp(-optical_depth)

    # Интенсивность черного тела
    bb_intensity = planck_function(mol_wave, temp)

    # Поток с учетом поглощения: I = I0 * transmission
    transmitted_intensity = bb_intensity * transmission

    # Нормировка на максимум черного тела (континуум)
    # Важно: нормируем на сам континуум, а не на максимум!
    normalized_spectrum = transmitted_intensity / bb_intensity
    normalized_spectrum = np.nan_to_num(normalized_spectrum, nan=1.0, posinf=1.0)

    return normalized_spectrum, transmission, bb_intensity


# Загрузка кросс-секции ZrO
mol_data = np.genfromtxt("/home/delta/exocross/input/ZrO_all.xsec")
molecular_wave = 1e8 / mol_data[:, 0][::-1]  # перевод из обратных см в ангстремы
molecular_cross_section = mol_data[:, 1][::-1]

# Параметры для ZrO
T_zro = 1500  # температура для ZrO
column_density_zro = 1e16  # столбиковая концентрация

# Получение нормированного спектра ZrO
zro_norm_spectrum, zro_transmission, zro_bb = normalize_molecular_spectrum(
    molecular_wave, molecular_cross_section, T_zro, column_density_zro
)

# Интерполяция молекулярного спектра на длины волн реперного спектра
zro_norm_interp = np.interp(
    rep_wave, molecular_wave, zro_norm_spectrum, left=1.0, right=1.0
)

# Интерполяция молекулярного спектра на длины волн obs_norm
zro_norm_obs_interp = np.interp(
    obs_norm[:, 0], molecular_wave, zro_norm_spectrum, left=1.0, right=1.0
)

# ==================== КОРРЕКЦИЯ НАБЛЮДАЕМОГО СПЕКТРА ====================


def find_continuum_robust(wavelength, flux, molecular_spectrum, window_size=100):
    """
    Находит континуум как максимум в скользящем окне с учетом молекулярного поглощения
    """
    n_points = len(wavelength)
    continuum = np.zeros(n_points)

    # Создаем маску для областей без сильного молекулярного поглощения
    # Молекулярное поглощение считается сильным, если нормированный поток < 0.8
    molecular_mask = molecular_spectrum > 0.8

    for i in range(n_points):
        # Определяем окно
        half_window = window_size // 2
        start = max(0, i - half_window)
        end = min(n_points, i + half_window)

        # Берем только точки с малым молекулярным поглощением
        window_mask = molecular_mask[start:end]
        if np.sum(window_mask) > 0:
            window_flux = flux[start:end][window_mask]
            # Континуум - это максимум в окне (или 95-й перцентиль для устойчивости)
            continuum[i] = np.percentile(window_flux, 95)
        else:
            # Если нет точек с малым поглощением, берем среднее
            continuum[i] = np.mean(flux[start:end])

    return continuum


def normalize_with_molecular_spectrum(
    obs_wave, obs_flux, zro_spectrum, poly_order=3, smooth_window=50
):
    """
    Нормировка спектра с использованием молекулярного спектра ZrO

    Идея:
    1. Находим континуум как огибающую максимумов
    2. Учитываем, что в областях сильного поглощения ZrO континуум должен быть выше
    3. Делаем итеративную коррекцию
    """

    # Шаг 1: Находим начальную оценку континуума
    # Используем скользящее окно для поиска максимумов
    n_points = len(obs_wave)
    continuum = np.zeros(n_points)
    window_size = max(51, int(n_points * 0.01))  # 1% от спектра

    for i in range(n_points):
        half = window_size // 2
        start = max(0, i - half)
        end = min(n_points, i + half)
        continuum[i] = np.percentile(obs_flux[start:end], 90)

    # Шаг 2: Сглаживаем континуум полиномом
    # Берем точки, где молекулярное поглощение слабое (zro_spectrum > 0.85)
    good_mask = zro_spectrum > 0.85
    if np.sum(good_mask) > 10:
        coeffs = np.polyfit(obs_wave[good_mask], continuum[good_mask], poly_order)
        continuum_smooth = np.polyval(coeffs, obs_wave)
    else:
        coeffs = np.polyfit(obs_wave, continuum, poly_order)
        continuum_smooth = np.polyval(coeffs, obs_wave)

    # Шаг 3: Первая нормировка
    normalized = obs_flux / continuum_smooth

    # Шаг 4: Итеративная коррекция
    for iteration in range(3):
        # Находим точки, где normalized близок к 1 (потенциальный континуум)
        # и где молекулярное поглощение слабое
        good_mask = (np.abs(normalized - 1.0) < 0.1) & (zro_spectrum > 0.8)

        if np.sum(good_mask) > 10:
            # Подгоняем континуум заново по этим точкам
            coeffs = np.polyfit(
                obs_wave[good_mask],
                obs_flux[good_mask] / zro_spectrum[good_mask],
                poly_order,
            )
            continuum_new = np.polyval(coeffs, obs_wave)

            # Проверяем, что континуум не уходит ниже спектра
            # Континуум должен быть выше или равен спектру
            continuum_new = np.maximum(continuum_new, obs_flux * 1.01)
            continuum_new = np.maximum(continuum_new, continuum_smooth * 0.9)

            # Обновляем
            continuum_smooth = continuum_new
            normalized = obs_flux / continuum_smooth

            # Отладочная печать
            print(
                f"Iteration {iteration}: median continuum = {np.median(continuum_smooth):.3f}"
            )

    # Финальная проверка: континуум должен быть >= 1 после нормировки
    # Это значит, что нормированный спектр не должен превышать 1
    # Но для реальных данных это не всегда так из-за шумов

    return normalized, continuum_smooth


# Применяем коррекцию с учетом ZrO
obs_norm_corrected, continuum_obs = normalize_with_molecular_spectrum(
    obs_norm[:, 0], obs_norm[:, 1], zro_norm_obs_interp, poly_order=3, smooth_window=50
)

# Проверяем, что континуум не превышает 1
# Если нужно, делаем дополнительную нормировку
if np.median(obs_norm_corrected) > 1.0:
    print(
        f"Медиана спектра = {np.median(obs_norm_corrected):.3f}, > 1. Дополнительная нормировка..."
    )
    # Находим медиану в областях без сильного поглощения
    no_abs_mask = zro_norm_obs_interp > 0.9
    if np.sum(no_abs_mask) > 10:
        median_flux = np.median(obs_norm_corrected[no_abs_mask])
        obs_norm_corrected = obs_norm_corrected / median_flux
        continuum_obs = continuum_obs / median_flux
        print(f"Нормировка на {median_flux:.3f}")

# Создаем исправленный спектр
obs_norm_zro = np.column_stack((obs_norm[:, 0], obs_norm_corrected))

# ==================== ВИЗУАЛИЗАЦИЯ ====================

if plot:
    with plt.style.context(["science"]):
        # 1. Молекулярный спектр ZrO
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        ax = axes[0, 0]
        ax.plot(molecular_wave, zro_bb / np.max(zro_bb), label="Black body (1500K)")
        ax.set_xlabel(r"Wavelength, $\AA$")
        ax.set_ylabel("Intensity (norm)")
        ax.set_title("Planck function")
        ax.legend()

        ax = axes[0, 1]
        ax.plot(molecular_wave, zro_transmission)
        ax.set_xlabel(r"Wavelength, $\AA$")
        ax.set_ylabel("Transmission")
        ax.set_title(f"ZrO transmission (N = {column_density_zro:.0e})")
        ax.set_ylim(0, 1.1)

        ax = axes[1, 0]
        ax.plot(molecular_wave, zro_norm_spectrum, color="darkred")
        ax.set_xlabel(r"Wavelength, $\AA$")
        ax.set_ylabel("Normalized flux")
        ax.set_title("ZrO normalized spectrum")
        ax.set_ylim(0, 1.1)

        ax = axes[1, 1]
        ax.plot(obs_norm[:, 0], obs_norm[:, 1], label="Before correction", alpha=0.5)
        ax.plot(
            obs_norm_zro[:, 0],
            obs_norm_zro[:, 1],
            label="After ZrO correction",
            alpha=0.7,
        )
        ax.plot(
            obs_norm[:, 0],
            continuum_obs,
            label="Continuum",
            color="green",
            linestyle="--",
            alpha=0.5,
        )
        ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel(r"Wavelength, $\AA$")
        ax.set_ylabel("Normalized flux")
        ax.set_title("Spectrum correction with ZrO")
        ax.legend()
        ax.set_xlim(6000, 7000)
        ax.set_ylim(0, 1.5)

        plt.tight_layout()

        # 2. Детальный вид в области полос ZrO
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Область с сильными полосами ZrO
        ax = axes[0]
        ax.plot(
            obs_norm[:, 0], obs_norm[:, 1], label="Original", color="navy", alpha=0.5
        )
        ax.plot(
            obs_norm_zro[:, 0],
            obs_norm_zro[:, 1],
            label="ZrO-corrected",
            color="crimson",
            alpha=0.7,
        )
        ax.plot(
            obs_norm[:, 0],
            continuum_obs,
            label="Continuum",
            color="green",
            linestyle="--",
            alpha=0.5,
        )
        ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlim(6000, 6700)
        ax.set_ylim(0, 1.5)
        ax.set_xlabel(r"Wavelength, $\AA$")
        ax.set_ylabel("Normalized flux")
        ax.legend()
        ax.set_title("ZrO band region")

        # Узкая область для детального сравнения
        ax = axes[1]
        ax.plot(
            obs_norm[:, 0], obs_norm[:, 1], label="Original", color="navy", alpha=0.5
        )
        ax.plot(
            obs_norm_zro[:, 0],
            obs_norm_zro[:, 1],
            label="ZrO-corrected",
            color="crimson",
            alpha=0.7,
        )
        ax.plot(
            obs_norm[:, 0],
            zro_norm_obs_interp * 1.0,
            label="ZrO template",
            color="orange",
            alpha=0.5,
        )
        ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlim(6450, 6550)
        ax.set_ylim(0.2, 1.3)
        ax.set_xlabel(r"Wavelength, $\AA$")
        ax.set_ylabel("Normalized flux")
        ax.legend()
        ax.set_title("Zoom in strong ZrO band")

        plt.tight_layout()
        plt.show()

# Сохранение результата
if save:
    np.savetxt(
        "obs_norm_zro_corrected.txt",
        obs_norm_zro,
        header="wavelength flux_norm_zro_corrected",
    )
    np.savetxt(
        "continuum_fit.txt",
        np.column_stack((obs_norm[:, 0], continuum_obs)),
        header="wavelength continuum",
    )
    np.savetxt(
        "zro_normalized.txt",
        np.column_stack((molecular_wave, zro_norm_spectrum)),
        header="wavelength zro_norm",
    )
    np.savetxt(
        "zro_transmission.txt",
        np.column_stack((molecular_wave, zro_transmission)),
        header="wavelength zro_transmission",
    )

    print("Файлы сохранены:")
    print("  - obs_norm_zro_corrected.txt")
    print("  - continuum_fit.txt")
    print("  - zro_normalized.txt")
    print("  - zro_transmission.txt")
