from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import os
import sys
from scipy.constants import h, c, k
from scipy.optimize import minimize_scalar, differential_evolution
import PyAstronomy.pyasl as pyasl


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
_, data[:, 0] = pyasl.dopplerShift(
    data[:, 0], data[:, 1], bcvr_arr[6] / 1000 + 36.5, edgeHandling="firstlast"
)


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


# ==================== ФУНКЦИИ ДЛЯ РАБОТЫ С МОЛЕКУЛЯРНЫМИ СПЕКТРАМИ ====================


def planck_function(wavelength, temperature):
    """
    Функция Планка для черного тела
    """
    wl_m = wavelength * 1e-10  # перевод в метры
    intensity = (
        (2 * h * c**2) / (wl_m**5) / (np.exp(h * c / (wl_m * k * temperature)) - 1)
    )
    return intensity


def normalize_molecular_spectrum(
    mol_wave, mol_cross_section, temp=1500, column_density=1e16
):
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


def combine_molecular_spectra(obs_wave, molecular_spectra_list, method="sum"):
    """
    Комбинирует несколько молекулярных спектров

    Parameters:
    -----------
    obs_wave : array
        Длины волн наблюдений
    molecular_spectra_list : list of tuples (wave, spectrum, name)
        Список молекулярных спектров
    method : str
        Метод комбинирования: 'multiply', 'sum', 'min'

    Returns:
    --------
    combined_spectrum : array
        Комбинированный спектр
    """
    # Интерполируем все спектры на сетку наблюдений
    interpolated_spectra = []

    for wave, spectrum, name in molecular_spectra_list:
        interp = np.interp(obs_wave, wave, spectrum, left=1.0, right=1.0)
        interpolated_spectra.append(interp)
        print(f"  {name}: интерполирован на сетку наблюдений")

    # Комбинируем
    if method == "multiply":
        # Перемножение - для независимых поглотителей
        combined = np.ones_like(obs_wave)
        for spec in interpolated_spectra:
            combined *= spec
    elif method == "sum":
        # Суммирование оптических глубин: I = exp(-sum(tau_i))
        total_optical_depth = np.zeros_like(obs_wave)
        for spec in interpolated_spectra:
            total_optical_depth += -np.log(np.clip(spec, 1e-10, 1.0))
        combined = np.exp(-total_optical_depth)
    elif method == "min":
        # Минимальное пропускание
        combined = np.ones_like(obs_wave)
        for spec in interpolated_spectra:
            combined = np.minimum(combined, spec)
    else:
        raise ValueError(f"Unknown method: {method}")

    return combined


# ==================== ФУНКЦИИ ДЛЯ ОПТИМИЗАЦИИ COLUMN_DENSITY ====================


def optimize_column_density(
    obs_wave,
    obs_flux,
    mol_wave,
    mol_cross_section,
    temperature,
    column_density_range=(1e16, 1e18),
    method="brent",
    mask_threshold=0.1,
    use_delta=True,
    delta_interp=None,
    rep_flux=None,
):
    """
    Автоматический подбор column_density путем минимизации невязки
    между наблюдаемым и модельным спектром.

    Parameters:
    -----------
    obs_wave, obs_flux : array
        Наблюдаемый спектр
    mol_wave, mol_cross_section : array
        Молекулярные данные
    temperature : float
        Температура для функции Планка
    column_density_range : tuple
        Диапазон поиска (min, max)
    method : str
        Метод оптимизации: 'brent', 'differential_evolution', 'grid'
    mask_threshold : float
        Порог для маскирования сильных линий
    use_delta : bool
        Использовать ли delta массив для взвешивания
    delta_interp : array
        Интерполированный delta массив
    rep_flux : array
        Реперный спектр
    """

    def compute_residual(log10_N):
        """Вычисляет невязку для заданной column_density"""
        N = 10**log10_N

        # Получаем нормированный спектр для данной N
        norm_spectrum, _, _ = normalize_molecular_spectrum(
            mol_wave, mol_cross_section, temperature, N
        )

        # Интерполируем на сетку наблюдений
        mol_interp = np.interp(obs_wave, mol_wave, norm_spectrum, left=1.0, right=1.0)

        # Создаем маску для точек, где молекулярное поглощение не слишком сильное
        # (избегаем насыщенных линий)
        weak_absorption_mask = mol_interp > mask_threshold

        # Если используем delta, добавляем веса
        if use_delta and delta_interp is not None:
            # Точки с малой delta имеют больший вес
            weights = 1.0 / (delta_interp + 0.01)  # добавляем epsilon для стабильности
            weights = weights / np.max(weights)  # нормализуем
            weights = weights * weak_absorption_mask
        else:
            weights = weak_absorption_mask.astype(float)

        # Нормируем наблюдаемый спектр на молекулярный
        # (предполагаем, что континуум уже определен)
        corrected_flux = obs_flux / mol_interp

        # Невязка: отклонение от 1 (идеальный континуум)
        # Используем только те точки, где поглощение не слишком сильное
        valid_points = weights > 0

        if np.sum(valid_points) < 10:
            return 1e10  # слишком мало точек

        # Взвешенная невязка
        residuals = (corrected_flux[valid_points] - 1.0) ** 2
        weighted_residual = np.sum(residuals * weights[valid_points]) / np.sum(
            weights[valid_points]
        )

        # Штраф за выход за пределы разумного
        if N < column_density_range[0] or N > column_density_range[1]:
            weighted_residual += (
                1e6 * (np.log10(N) - np.log10(column_density_range[0])) ** 2
            )

        return weighted_residual

    # Поиск оптимального значения
    log10_min = np.log10(column_density_range[0])
    log10_max = np.log10(column_density_range[1])

    if method == "brent":
        # Метод Брента (быстрый, но может застрять в локальном минимуме)
        result = minimize_scalar(
            compute_residual, bounds=(log10_min, log10_max), method="bounded"
        )
        optimal_log10_N = result.x
        grid_results = None

    elif method == "differential_evolution":
        # Дифференциальная эволюция (глобальная оптимизация, но медленнее)
        result = differential_evolution(
            compute_residual, bounds=[(log10_min, log10_max)], maxiter=50, popsize=15
        )
        optimal_log10_N = result.x[0]
        grid_results = None

    elif method == "grid":
        # Полный перебор по сетке (медленно, но надежно)
        n_points = 50
        log10_N_values = np.linspace(log10_min, log10_max, n_points)
        residuals = [compute_residual(log10_N) for log10_N in log10_N_values]
        optimal_idx = np.argmin(residuals)
        optimal_log10_N = log10_N_values[optimal_idx]

        grid_results = {
            "log10_N": log10_N_values,
            "residuals": residuals,
            "optimal_idx": optimal_idx,
        }

    optimal_N = 10**optimal_log10_N

    # Дополнительная информация
    result_info = {
        "optimal_N": optimal_N,
        "optimal_log10_N": optimal_log10_N,
        "residual": compute_residual(optimal_log10_N),
    }

    if grid_results is not None:
        result_info["grid_results"] = grid_results

    return result_info


def optimize_multiple_molecules(
    obs_wave,
    obs_flux,
    molecules_info,
    column_density_ranges,
    temperatures,
    delta_interp=None,
    rep_flux=None,
    method="grid",
    n_iter=3,
):
    """
    Последовательная оптимизация column_density для нескольких молекул.

    Parameters:
    -----------
    molecules_info : list of tuples
        [(wave, cross_section, name), ...]
    column_density_ranges : list of tuples
        [(min, max), ...] для каждой молекулы
    temperatures : list
        Температуры для каждой молекулы
    n_iter : int
        Количество итераций для итеративного уточнения
    """

    results = {}
    current_flux = obs_flux.copy()
    molecular_spectra = []

    for i, (mol_wave, mol_cross_section, name) in enumerate(molecules_info):
        print(f"\n=== Оптимизация для {name} ===")

        # Оптимизация для текущей молекулы
        opt_result = optimize_column_density(
            obs_wave,
            current_flux,
            mol_wave,
            mol_cross_section,
            temperatures[i],
            column_density_ranges[i],
            method=method,
            use_delta=True,
            delta_interp=delta_interp,
            rep_flux=rep_flux,
        )

        optimal_N = opt_result["optimal_N"]
        results[name] = {
            "optimal_N": optimal_N,
            "log10_N": np.log10(optimal_N),
            "residual": opt_result["residual"],
        }

        if "grid_results" in opt_result:
            results[name]["grid_results"] = opt_result["grid_results"]

        # Получаем спектр с оптимальной N
        norm_spectrum, _, _ = normalize_molecular_spectrum(
            mol_wave, mol_cross_section, temperatures[i], optimal_N
        )
        mol_interp = np.interp(obs_wave, mol_wave, norm_spectrum, left=1.0, right=1.0)
        molecular_spectra.append(mol_interp)

        # Убираем вклад текущей молекулы для следующей итерации
        current_flux = current_flux / mol_interp

        print(f"  Оптимальная N = {optimal_N:.2e} cm^-2")
        print(f"  log10(N) = {np.log10(optimal_N):.2f}")
        print(f"  Невязка = {opt_result['residual']:.6f}")

    # Итеративное уточнение
    for iteration in range(n_iter - 1):
        print(f"\n=== Итерация {iteration + 2} ===")
        current_flux = obs_flux.copy()

        for i, (mol_wave, mol_cross_section, name) in enumerate(molecules_info):
            # Используем предыдущие результаты как начальное приближение
            opt_result = optimize_column_density(
                obs_wave,
                current_flux,
                mol_wave,
                mol_cross_section,
                temperatures[i],
                column_density_ranges[i],
                method=method,
                use_delta=True,
                delta_interp=delta_interp,
                rep_flux=rep_flux,
            )

            optimal_N = opt_result["optimal_N"]
            results[name][f"N_iter_{iteration + 2}"] = optimal_N
            results[name][f"residual_iter_{iteration + 2}"] = opt_result["residual"]

            # Обновляем спектр
            norm_spectrum, _, _ = normalize_molecular_spectrum(
                mol_wave, mol_cross_section, temperatures[i], optimal_N
            )
            mol_interp = np.interp(
                obs_wave, mol_wave, norm_spectrum, left=1.0, right=1.0
            )
            molecular_spectra[i] = mol_interp
            current_flux = current_flux / mol_interp

            print(
                f"  {name}: N = {optimal_N:.2e} cm^-2, residual = {opt_result['residual']:.6f}"
            )

    # Комбинируем все молекулярные спектры
    combined = np.ones_like(obs_wave)
    for spec in molecular_spectra:
        combined *= spec

    results["combined_spectrum"] = combined
    results["individual_spectra"] = molecular_spectra

    return results


# ==================== ОСНОВНАЯ ЧАСТЬ: РАБОТА С МОЛЕКУЛЯРНЫМИ СПЕКТРАМИ ====================

# Загрузка молекулярных данных
print("\n=== ЗАГРУЗКА МОЛЕКУЛЯРНЫХ ДАННЫХ ===")

# 1. ZrO
zro_data = np.genfromtxt("/home/delta/exocross/input/ZrO_all.xsec")
zro_wave = 1e8 / zro_data[:, 0][::-1]
zro_cross_section = zro_data[:, 1][::-1]
print(f"ZrO: загружено {len(zro_wave)} точек")

# 2. TiO
tio_data = np.genfromtxt("/home/delta/exocross/input/TiO_all.xsec")
tio_wave = 1e8 / tio_data[:, 0][::-1]
tio_cross_section = tio_data[:, 1][::-1]
print(f"TiO: загружено {len(tio_wave)} точек")

# ============ АВТОМАТИЧЕСКИЙ ПОДБОР COLUMN_DENSITY ============

# Начальные температуры (можно оставить фиксированными или тоже оптимизировать)
T_zro = 1500  # K
T_tio = 1800  # K

# Подготавливаем данные для оптимизации
molecules_info = [
    (zro_wave, zro_cross_section, "ZrO"),
    (tio_wave, tio_cross_section, "TiO"),
]

column_density_ranges = [
    (1e16, 1e18),  # диапазон для ZrO
    (1e16, 1e18),  # диапазон для TiO
]

temperatures = [T_zro, T_tio]

# Получаем интерполированный delta массив
delta_interp = np.interp(obs_norm[:, 0], rep_wave, delta_arr_mean, left=1.0, right=1.0)
rep_interp = np.interp(obs_norm[:, 0], rep_wave, rep_flux, left=1.0, right=1.0)

# Запускаем оптимизацию
print("\n=== АВТОМАТИЧЕСКИЙ ПОДБОР COLUMN_DENSITY ===")
print("Метод: grid (полный перебор)")
print("Диапазон: 1e14 - 1e17 cm^-2")

optimization_results = optimize_multiple_molecules(
    obs_norm[:, 0],
    obs_norm[:, 1],
    molecules_info,
    column_density_ranges,
    temperatures,
    delta_interp=delta_interp,
    rep_flux=rep_interp,
    method="grid",
    n_iter=2,
)

# Извлекаем оптимальные значения
optimal_zro_N = optimization_results["ZrO"]["optimal_N"]
optimal_tio_N = optimization_results["TiO"]["optimal_N"]

print("\n=== ОПТИМАЛЬНЫЕ ЗНАЧЕНИЯ ===")
print(f"ZrO: N = {optimal_zro_N:.2e} cm^-2, log10(N) = {np.log10(optimal_zro_N):.3f}")
print(f"TiO: N = {optimal_tio_N:.2e} cm^-2, log10(N) = {np.log10(optimal_tio_N):.3f}")

# Используем оптимальные значения для финальной нормировки
zro_norm_spectrum, zro_optical_depth, zro_bb = normalize_molecular_spectrum(
    zro_wave, zro_cross_section, T_zro, optimal_zro_N
)

tio_norm_spectrum, tio_optical_depth, tio_bb = normalize_molecular_spectrum(
    tio_wave, tio_cross_section, T_tio, optimal_tio_N
)

# Интерполяция на сетку obs_norm
zro_norm_obs_interp = np.interp(
    obs_norm[:, 0], zro_wave, zro_norm_spectrum, left=1.0, right=1.0
)

tio_norm_obs_interp = np.interp(
    obs_norm[:, 0], tio_wave, tio_norm_spectrum, left=1.0, right=1.0
)

# Комбинированный молекулярный спектр
combined_molecular_spectrum = optimization_results["combined_spectrum"]

print("\n=== МОЛЕКУЛЯРНЫЕ ПАРАМЕТРЫ (ОПТИМИЗИРОВАННЫЕ) ===")
print(f"ZrO: T = {T_zro} K, N = {optimal_zro_N:.2e} cm^-2")
print(f"TiO: T = {T_tio} K, N = {optimal_tio_N:.2e} cm^-2")
print(f"Метод комбинирования: multiply (независимые поглотители)")

# ==================== ИСПРАВЛЕННАЯ НОРМИРОВКА С УЧЕТОМ ОБОИХ МОЛЕКУЛ ====================


def normalize_with_delta_and_molecular(
    obs_wave,
    obs_flux,
    molecular_spectrum,
    rep_wave,
    rep_flux,
    delta_arr_mean,
    delta_threshold=0.1,
    molecular_threshold=0.02,
    poly_order=3,
    n_iter=3,
):
    """
    Нормировка спектра с использованием:
    1. Массива delta для определения доверенных точек (где модели близки к реперу)
    2. Комбинированного молекулярного спектра (ZrO + TiO) для учета поглощения
    3. Итеративной подгонки континуума с принудительным ограничением flux <= 1
    """

    # Интерполяция delta на длины волн obs
    delta_interp = np.interp(obs_wave, rep_wave, delta_arr_mean, left=1.0, right=1.0)

    # Интерполяция реперного спектра
    rep_interp = np.interp(obs_wave, rep_wave, rep_flux, left=1.0, right=1.0)

    # СОЗДАНИЕ МАСКИ ДОВЕРЕННЫХ ТОЧЕК
    # 1. Точки, где delta мала (модели близки к реперу)
    good_delta_mask = delta_interp < delta_threshold

    # 2. Точки, где молекулярное поглощение слабое (используем комбинированный спектр)
    good_molecular_mask = np.abs(molecular_spectrum - 1.0) < molecular_threshold

    # 3. Точки, где реперный спектр близок к континууму (нет сильных линий)
    good_rep_mask = rep_interp > 0.95

    # 4. Объединенная маска - точки, которым можно доверять
    trust_mask = good_delta_mask & good_molecular_mask & good_rep_mask

    print(f"\n=== МАСКА ДОВЕРЕННЫХ ТОЧЕК ===")
    print(f"Доверенных точек: {np.sum(trust_mask)} из {len(obs_wave)}")
    print(f"  - По delta: {np.sum(good_delta_mask)}")
    print(f"  - По молекулярному спектру: {np.sum(good_molecular_mask)}")
    print(f"  - По реперу: {np.sum(good_rep_mask)}")

    if np.sum(trust_mask) < 20:
        print("Предупреждение: слишком мало доверенных точек, расширяем критерии")
        trust_mask = good_delta_mask | (good_molecular_mask & good_rep_mask)
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
                    final_normalized[i] = (1 - weight) * final_normalized[
                        left
                    ] + weight * final_normalized[right]
                elif len(left_idx) > 0:
                    final_normalized[i] = final_normalized[left_idx[-1]]
                elif len(right_idx) > 0:
                    final_normalized[i] = final_normalized[right_idx[0]]
                else:
                    final_normalized[i] = 0.99

    return final_normalized, continuum, trust_mask, delta_interp


# Применяем исправленную нормировку с комбинированным молекулярным спектром
obs_norm_corrected, continuum_obs, trust_mask, delta_interp = (
    normalize_with_delta_and_molecular(
        obs_norm[:, 0],
        obs_norm[:, 1],
        combined_molecular_spectrum,
        rep_wave,
        rep_flux,
        delta_arr_mean,
        delta_threshold=0.1,
        molecular_threshold=0.02,
        poly_order=3,
        n_iter=5,
    )
)

# Проверяем, что все значения <= 1
print(f"\n=== РЕЗУЛЬТАТЫ НОРМИРОВКИ ===")
print(f"Максимальное значение после нормировки: {np.max(obs_norm_corrected)}")
print(f"Минимальное значение после нормировки: {np.min(obs_norm_corrected)}")
print(
    f"Доля точек > 1: {np.sum(obs_norm_corrected > 1.0) / len(obs_norm_corrected) * 100:.2f}%"
)

# Если все еще есть точки > 1, принудительно ограничиваем
obs_norm_corrected = np.clip(obs_norm_corrected, 0.0, 1.0)

# Создаем исправленный спектр
obs_norm_molecular = np.column_stack((obs_norm[:, 0], obs_norm_corrected))

# ==================== ВИЗУАЛИЗАЦИЯ ====================

if plot:
    with plt.style.context(["science"]):
        # 1. Маска доверенных точек и молекулярные спектры
        fig, axes = plt.subplots(4, 1, figsize=(12, 14))

        ax = axes[0]
        ax.plot(obs_norm[:, 0], delta_interp, color="blue", alpha=0.7)
        ax.axhline(y=0.1, color="red", linestyle="--", label="delta threshold")
        ax.set_xlabel(r"Wavelength, $\AA$")
        ax.set_ylabel("Delta")
        ax.set_title("Delta array (model differences from reference)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(
            obs_norm[:, 0], zro_norm_obs_interp, color="darkred", alpha=0.7, label="ZrO"
        )
        ax.plot(
            obs_norm[:, 0],
            tio_norm_obs_interp,
            color="darkblue",
            alpha=0.7,
            label="TiO",
        )
        ax.plot(
            obs_norm[:, 0],
            combined_molecular_spectrum,
            color="green",
            alpha=0.5,
            linestyle="--",
            linewidth=2,
            label="Combined",
        )
        ax.axhline(y=0.98, color="black", linestyle=":", label="threshold")
        ax.set_xlabel(r"Wavelength, $\AA$")
        ax.set_ylabel("Normalized flux")
        ax.set_title(
            f"Molecular spectra (ZrO: N={optimal_zro_N:.2e}, TiO: N={optimal_tio_N:.2e})"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.scatter(
            obs_norm[:, 0][trust_mask],
            obs_norm[:, 1][trust_mask],
            s=1,
            color="green",
            alpha=0.5,
            label="Trusted points",
        )
        ax.plot(
            obs_norm[:, 0], obs_norm[:, 1], color="navy", alpha=0.3, label="Original"
        )
        ax.plot(
            obs_norm[:, 0],
            continuum_obs,
            color="red",
            linestyle="--",
            label="Continuum fit",
        )
        ax.set_xlabel(r"Wavelength, $\AA$")
        ax.set_ylabel("Flux")
        ax.set_title("Trusted points for continuum fitting")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[3]
        # Показываем вклад каждой молекулы
        zro_abs = 1 - zro_norm_obs_interp
        tio_abs = 1 - tio_norm_obs_interp
        ax.fill_between(
            obs_norm[:, 0],
            0,
            zro_abs,
            alpha=0.3,
            color="darkred",
            label="ZrO absorption",
        )
        ax.fill_between(
            obs_norm[:, 0],
            0,
            tio_abs,
            alpha=0.3,
            color="darkblue",
            label="TiO absorption",
        )
        ax.set_xlabel(r"Wavelength, $\AA$")
        ax.set_ylabel("Absorption depth")
        ax.set_title("Molecular absorption contributions")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 2. Графики оптимизации
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # График невязки для ZrO
        if "grid_results" in optimization_results["ZrO"]:
            grid_res = optimization_results["ZrO"]["grid_results"]
            ax = axes[0, 0]
            ax.plot(grid_res["log10_N"], grid_res["residuals"], "b-", linewidth=2)
            ax.axvline(
                grid_res["log10_N"][grid_res["optimal_idx"]],
                color="r",
                linestyle="--",
                label=f"Optimal: {10 ** grid_res['log10_N'][grid_res['optimal_idx']]:.2e}",
            )
            ax.set_xlabel("log10(N)")
            ax.set_ylabel("Residual")
            ax.set_title("ZrO: Column density optimization")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # График невязки для TiO
        if "grid_results" in optimization_results["TiO"]:
            grid_res = optimization_results["TiO"]["grid_results"]
            ax = axes[0, 1]
            ax.plot(grid_res["log10_N"], grid_res["residuals"], "g-", linewidth=2)
            ax.axvline(
                grid_res["log10_N"][grid_res["optimal_idx"]],
                color="r",
                linestyle="--",
                label=f"Optimal: {10 ** grid_res['log10_N'][grid_res['optimal_idx']]:.2e}",
            )
            ax.set_xlabel("log10(N)")
            ax.set_ylabel("Residual")
            ax.set_title("TiO: Column density optimization")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Сравнение спектров с оптимальными параметрами
        ax = axes[1, 0]
        ax.plot(obs_norm[:, 0], obs_norm[:, 1], "navy", alpha=0.5, label="Original")
        ax.plot(
            obs_norm[:, 0],
            optimization_results["combined_spectrum"],
            "crimson",
            alpha=0.7,
            label="Combined molecular",
        )
        ax.set_xlim(6000, 7500)
        ax.set_xlabel(r"Wavelength, $\AA$")
        ax.set_ylabel("Flux")
        ax.set_title("Molecular spectrum with optimal parameters")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Сходимость итераций
        ax = axes[1, 1]
        molecules = ["ZrO", "TiO"]
        colors = ["red", "blue"]
        for i, mol in enumerate(molecules):
            N_values = []
            for key in optimization_results[mol].keys():
                if key.startswith("N_iter_"):
                    N_values.append(optimization_results[mol][key])
                elif key == "optimal_N":
                    N_values.insert(0, optimization_results[mol][key])

            if len(N_values) > 0:
                ax.plot(
                    range(len(N_values)),
                    np.log10(N_values),
                    "o-",
                    color=colors[i],
                    label=mol,
                )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("log10(N)")
        ax.set_title("Convergence of column density")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 3. Сравнение спектров до и после коррекции
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))

        # Полный спектр
        ax = axes[0]
        ax.plot(
            obs_norm[:, 0],
            obs_norm[:, 1],
            label="Before correction",
            color="navy",
            alpha=0.5,
        )
        ax.plot(
            obs_norm_molecular[:, 0],
            obs_norm_molecular[:, 1],
            label="After correction (ZrO+TiO)",
            color="crimson",
            alpha=0.7,
        )
        ax.plot(
            obs_norm[:, 0],
            continuum_obs,
            label="Continuum fit",
            color="green",
            linestyle="--",
        )
        ax.axhline(y=1.0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlim(6000, 7500)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel(r"Wavelength, $\AA$")
        ax.set_ylabel("Normalized flux")
        ax.legend()
        ax.set_title("Full spectrum comparison (ZrO + TiO)")
        ax.grid(True, alpha=0.3)

        # Область с сильным молекулярным поглощением (ZrO)
        ax = axes[1]
        ax.plot(
            obs_norm[:, 0],
            obs_norm[:, 1],
            label="Before correction",
            color="navy",
            alpha=0.5,
        )
        ax.plot(
            obs_norm_molecular[:, 0],
            obs_norm_molecular[:, 1],
            label="After correction (ZrO+TiO)",
            color="crimson",
            alpha=0.7,
        )
        ax.plot(
            obs_norm[:, 0],
            continuum_obs,
            label="Continuum fit",
            color="green",
            linestyle="--",
        )
        ax.axhline(y=1.0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlim(6450, 6650)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel(r"Wavelength, $\AA$")
        ax.set_ylabel("Normalized flux")
        ax.legend()
        ax.set_title("Zoom: ZrO band region (6450-6650 $\AA$)")
        ax.grid(True, alpha=0.3)

        # Область с TiO поглощением
        ax = axes[2]
        ax.plot(
            obs_norm[:, 0],
            obs_norm[:, 1],
            label="Before correction",
            color="navy",
            alpha=0.5,
        )
        ax.plot(
            obs_norm_molecular[:, 0],
            obs_norm_molecular[:, 1],
            label="After correction (ZrO+TiO)",
            color="crimson",
            alpha=0.7,
        )
        ax.plot(
            obs_norm[:, 0],
            continuum_obs,
            label="Continuum fit",
            color="green",
            linestyle="--",
        )
        ax.axhline(y=1.0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlim(7000, 7200)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel(r"Wavelength, $\AA$")
        ax.set_ylabel("Normalized flux")
        ax.legend()
        ax.set_title("Zoom: TiO region (7000-7200 $\AA$)")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 4. Гистограмма распределения
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(
            obs_norm_corrected, bins=50, alpha=0.7, color="crimson", edgecolor="black"
        )
        ax.axvline(x=1.0, color="red", linestyle="--", label="Continuum")
        ax.axvline(
            x=np.median(obs_norm_corrected),
            color="blue",
            linestyle="--",
            label=f"Median: {np.median(obs_norm_corrected):.3f}",
        )
        ax.set_xlabel("Normalized flux")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of normalized flux values (ZrO + TiO)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# Сохранение результата
if save:
    print("\n=== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ===")

    # Сохраняем исправленный спектр
    np.savetxt(
        "obs_norm_molecular_corrected.txt",
        obs_norm_molecular,
        header="wavelength flux_norm_molecular_corrected (ZrO+TiO)",
    )
    print("  - obs_norm_molecular_corrected.txt")

    # Сохраняем континуум
    np.savetxt(
        "continuum_fit.txt",
        np.column_stack((obs_norm[:, 0], continuum_obs)),
        header="wavelength continuum",
    )
    print("  - continuum_fit.txt")

    # Сохраняем маску доверенных точек
    np.savetxt(
        "trust_mask.txt",
        np.column_stack((obs_norm[:, 0], trust_mask.astype(int))),
        header="wavelength trust_mask",
    )
    print("  - trust_mask.txt")

    # Сохраняем отдельные молекулярные спектры
    np.savetxt(
        "zro_normalized.txt",
        np.column_stack((zro_wave, zro_norm_spectrum)),
        header="wavelength zro_norm",
    )
    print("  - zro_normalized.txt")

    np.savetxt(
        "tio_normalized.txt",
        np.column_stack((tio_wave, tio_norm_spectrum)),
        header="wavelength tio_norm",
    )
    print("  - tio_normalized.txt")

    # Сохраняем комбинированный спектр
    np.savetxt(
        "molecular_combined.txt",
        np.column_stack((obs_norm[:, 0], combined_molecular_spectrum)),
        header="wavelength combined_molecular_spectrum (ZrO+TiO)",
    )
    print("  - molecular_combined.txt")

    # Сохраняем результаты оптимизации
    with open("column_density_optimization.txt", "w") as f:
        f.write("=== РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ COLUMN_DENSITY ===\n\n")
        f.write("ZrO:\n")
        f.write(f"  Оптимальная N = {optimal_zro_N:.2e} cm^-2\n")
        f.write(f"  log10(N) = {np.log10(optimal_zro_N):.3f}\n")
        f.write(f"  Невязка = {optimization_results['ZrO']['residual']:.6f}\n")
        f.write(f"  Температура = {T_zro} K\n\n")
        f.write("TiO:\n")
        f.write(f"  Оптимальная N = {optimal_tio_N:.2e} cm^-2\n")
        f.write(f"  log10(N) = {np.log10(optimal_tio_N):.3f}\n")
        f.write(f"  Невязка = {optimization_results['TiO']['residual']:.6f}\n")
        f.write(f"  Температура = {T_tio} K\n\n")
        f.write("=== ИНФОРМАЦИЯ О НОРМИРОВКЕ ===\n")
        f.write(
            f"Number of trusted points: {np.sum(trust_mask)} out of {len(obs_norm)}\n"
        )
        f.write(f"Max normalized flux: {np.max(obs_norm_corrected):.6f}\n")
        f.write(f"Min normalized flux: {np.min(obs_norm_corrected):.6f}\n")
        f.write(f"Median normalized flux: {np.median(obs_norm_corrected):.6f}\n")
        f.write(f"Points above 1: {np.sum(obs_norm_corrected > 1.0)}\n")
        f.write(f"Метод комбинирования: multiply\n")
        f.write(f"Метод оптимизации: grid\n")
        f.write(f"Диапазон поиска: 1e14 - 1e17 cm^-2\n")

    print("  - column_density_optimization.txt")
    print("\nСохранение завершено!")

print("\n=== СКРИПТ ЗАВЕРШЕН ===")
