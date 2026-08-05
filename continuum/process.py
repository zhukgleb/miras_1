from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from scipy.optimize import curve_fit
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "spectra"))
from dech_processing import make_txt_from_spectra
from general_processing import (
    median_normalization,
    full_pipeline,
    normalize_orders_median,
)
from typing import List
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from Model_extractor import ModelGridExtractor
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar


from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


def fit_parabola(x, y):
    def parabola(x, a, b, c):
        return a * x**2 + b * x + c

    popt, pcov = curve_fit(parabola, x, y)
    a, b, c = popt

    y_fit = parabola(x, a, b, c)

    residuals = y - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    xb = -b / (2 * a)
    yb = a * xb**2 + b * xb + c
    yn = y / yb
    yn_p = y_fit / yb

    return {
        "a": a,
        "b": b,
        "c": c,
        "r2": r2,
        "y_fit": y_fit,
        "func": lambda x_val: parabola(x_val, a, b, c),
        "yn": yn,
        "yn_p": yn_p,
    }


def split_spectral_orders(
    data: np.ndarray, zero_threshold: float = 1e-6
) -> List[np.ndarray]:

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


def diagnose_normalization(obs_wave, obs_flux, normalized_obs, x_fit, y_fit, poly_func):
    """
    Диагностика качества нормировки
    """
    print("=" * 60)
    print("NORMALIZATION DIAGNOSTICS")
    print("=" * 60)

    # 1. Медиана нормированного спектра
    median_norm = np.median(normalized_obs)
    print(f"Median of normalized spectrum: {median_norm:.4f}")
    print(f"Expected: ~1.0 (deviation: {median_norm - 1.0:.4f})")

    # 2. Медиана отношений в реперных точках
    median_ratio = np.median(y_fit)
    print(f"Median ratio (Obs/Model) in reference points: {median_ratio:.4f}")

    # 3. Значения полинома в реперных точках
    poly_at_ref = poly_func(x_fit)
    mean_residual = np.mean(y_fit - poly_at_ref)
    print(f"Mean residual in reference points: {mean_residual:.6f}")
    print(f"RMS residual: {np.std(y_fit - poly_at_ref):.6f}")

    # 4. Проверка на систематический тренд остатков
    coeffs_trend = np.polyfit(x_fit, y_fit - poly_at_ref, 1)
    print(f"Linear trend in residuals: {coeffs_trend[0]:.6f} (should be near 0)")

    # 5. Полином на краях спектра
    poly_start = poly_func(obs_wave[0])
    poly_end = poly_func(obs_wave[-1])
    poly_mid = poly_func(obs_wave[len(obs_wave) // 2])
    print(
        f"Polynomial values: start={poly_start:.4f}, mid={poly_mid:.4f}, end={poly_end:.4f}"
    )
    print(f"Dynamic range of polynomial: {poly_end / poly_start:.4f}")

    print("=" * 60)

    # Рекомендации
    if median_norm > 1.05:
        print("⚠️  WARNING: Normalized spectrum is systematically >1")
        print("   Possible causes:")
        print("   - Outliers in reference points (use sigma-clipping)")
        print("   - Model is not normalized to continuum")
        print("   - Polynomial degree too low")
    elif median_norm < 0.95:
        print("⚠️  WARNING: Normalized spectrum is systematically <1")
        print("   Possible causes:")
        print("   - Systematic offset in model")
        print("   - Wrong interpolation")

    return {
        "median": median_norm,
        "median_ratio": median_ratio,
        "mean_residual": mean_residual,
        "rms_residual": np.std(y_fit - poly_at_ref),
    }


def normalize_with_poly(
    model_wave,
    model_flux,
    model_diff,
    obs_wave,
    obs_flux,
    threshold=0.1,
    poly_degree=3,
    sigma_clip=3.0,
    plot=True,
    force_normalize=True,
):
    """
    Нормировка наблюдаемого спектра с помощью полиномиальной подгонки
    по реперным точкам (где model_diff < threshold)

    Parameters:
    -----------
    model_wave : array - длины волн модели
    model_flux : array - поток модели
    model_diff : array - массив "отличий" (чем меньше, тем стабильнее)
    obs_wave : array - длины волн наблюдений
    obs_flux : array - поток наблюдений
    threshold : float - порог для реперных точек (default: 0.1)
    poly_degree : int - степень полинома (default: 3)
    sigma_clip : float - сигма для отбраковки выбросов (default: 3.0)
    plot : bool - показывать графики (default: True)
    force_normalize : bool - принудительно нормировать на медиану 1 (default: True)

    Returns:
    --------
    normalized_obs : array - нормированный наблюдаемый спектр
    poly_func : function - полиномиальная функция
    obs_flux_interp : array - интерполированный на сетку модели спектр наблюдений
    """

    print("\n" + "=" * 60)
    print("NORMALIZATION WITH POLYNOMIAL")
    print("=" * 60)

    # ============================================================
    # 1. Интерполяция наблюдений на сетку модели
    # ============================================================
    obs_flux_interp = np.interp(model_wave, obs_wave, obs_flux)
    print(f"Model median {np.median(model_flux)}")

    # 2. Реперные точки
    mask = model_diff < threshold

    n_points = np.sum(mask)
    print(f"Initial reference points (threshold={threshold}): {n_points}")

    if n_points < 5:
        # Если точек мало - ослабляем порог
        mask = model_diff < threshold * 1.5
        n_points = np.sum(mask)
        print(f"Relaxed threshold to {threshold * 1.5:.2f}, found {n_points} points")

    if n_points < 3:
        raise ValueError(
            f"Too few reference points ({n_points}) for polynomial fitting!"
        )

    x_fit = model_wave[mask]
    y_fit = obs_flux_interp[mask] / model_flux[mask]

    # 3. Отбрасываем выбросы по Y (итеративно)
    print("Cleaning outliers...")

    # Первичная очистка по медиане
    median_y = np.median(y_fit)
    mad = np.median(np.abs(y_fit - median_y))
    if mad > 0:
        clip_mask = np.abs(y_fit - median_y) < sigma_clip * mad
    else:
        clip_mask = np.ones(len(y_fit), dtype=bool)

    x_fit = x_fit[clip_mask]
    y_fit = y_fit[clip_mask]
    print(f"  After median clipping: {len(x_fit)} points")

    # Итеративная очистка (3 итерации)
    for i in range(3):
        if len(x_fit) < 5:
            break
        # Временная подгонка полиномом 2-го порядка
        try:
            coeffs_temp = np.polyfit(x_fit, y_fit, min(2, poly_degree))
            poly_temp = np.poly1d(coeffs_temp)
            residuals = y_fit - poly_temp(x_fit)
            std_res = np.std(residuals)

            if std_res > 0:
                mask_clean = np.abs(residuals) < sigma_clip * std_res
                x_fit = x_fit[mask_clean]
                y_fit = y_fit[mask_clean]
            else:
                break
        except:
            break

    print(f"  After iterative cleaning: {len(x_fit)} points")

    # 4. Подгонка полинома
    print(f"Fitting polynomial degree {poly_degree}...")

    try:
        coeffs = np.polyfit(x_fit, y_fit, poly_degree)
        poly_func = np.poly1d(coeffs)
    except np.linalg.LinAlgError:
        print(
            f"  WARNING: Polynomial degree {poly_degree} failed, trying lower degree..."
        )
        poly_degree = min(3, poly_degree)
        coeffs = np.polyfit(x_fit, y_fit, poly_degree)
        poly_func = np.poly1d(coeffs)

    # 5. Проверка качества подгонки
    y_fit_poly = poly_func(x_fit)
    residuals = y_fit - y_fit_poly
    rms_res = np.std(residuals)
    mean_res = np.mean(residuals)

    print(f"  RMS residual: {rms_res:.6f}")
    print(f"  Mean residual: {mean_res:.6f}")

    # 6. Визуализация
    if plot:
        plt.figure(figsize=(14, 8))

        # График 1: Подгонка
        plt.subplot(2, 2, 1)
        plt.plot(x_fit, y_fit, "bo", markersize=3, alpha=0.7, label="Reference points")

        # Гладкая кривая полинома
        x_smooth = np.linspace(min(x_fit), max(x_fit), 500)
        plt.plot(
            x_smooth,
            poly_func(x_smooth),
            "r-",
            linewidth=2,
            label=f"Polynomial (deg={poly_degree})",
        )

        plt.xlabel("Wavelength")
        plt.ylabel("Observed / Model")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title("Polynomial fit to reference points")

        # График 2: Остатки
        plt.subplot(2, 2, 2)
        plt.plot(x_fit, residuals, "go", markersize=3, alpha=0.7)
        plt.axhline(y=0, color="r", linestyle="--", linewidth=1.5)
        plt.axhline(y=3 * rms_res, color="orange", linestyle=":", alpha=0.7)
        plt.axhline(y=-3 * rms_res, color="orange", linestyle=":", alpha=0.7)
        plt.xlabel("Wavelength")
        plt.ylabel("Residuals")
        plt.grid(True, alpha=0.3)
        plt.title(f"Residuals (RMS = {rms_res:.4f})")

        # График 3: Гистограмма остатков
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
        plt.axvline(x=0, color="r", linestyle="--")
        plt.xlabel("Residual")
        plt.ylabel("Count")
        plt.title(f"Mean residual = {mean_res:.6f}")
        plt.grid(True, alpha=0.3)

        # График 4: Весь спектр (маленький участок для проверки)
        plt.subplot(2, 2, 4)
        # Показываем небольшой участок в центре спектра
        mid_idx = len(obs_wave) // 2
        half_range = min(500, len(obs_wave) // 4)
        plot_range = slice(mid_idx - half_range, mid_idx + half_range)

        plt.plot(
            obs_wave[plot_range],
            obs_flux[plot_range],
            "b-",
            alpha=0.5,
            label="Original",
        )
        plt.plot(
            obs_wave[plot_range],
            obs_flux[plot_range] / poly_func(obs_wave)[plot_range],
            "r-",
            alpha=0.7,
            label="Normalized",
        )
        plt.axhline(y=1.0, color="green", linestyle="--", alpha=0.5)
        plt.xlabel("Wavelength")
        plt.ylabel("Flux")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title("Spectrum before and after normalization (center)")

        plt.tight_layout()

    # 7. Применяем полином ко всему спектру
    poly_full = poly_func(obs_wave)
    normalized_obs = obs_flux / poly_full

    # 8. Принудительная нормализация (если нужно)
    if force_normalize:
        median_norm = np.median(normalized_obs)
        if abs(median_norm - np.median(model_flux)) > 0.02:
            print(
                f"  Forcing normalization: median {median_norm:.4f} -> {np.median(model_flux)} (model)"
            )
            correction = np.median(model_flux) / median_norm
            normalized_obs = normalized_obs * correction
            # Корректируем полином (для согласованности)
            poly_func = lambda x: poly_func(x) * correction

    # 9. Финальная диагностика
    final_median = np.median(normalized_obs)
    print(f"\nFinal median of normalized spectrum: {final_median:.4f}")

    if abs(final_median - np.median(model_flux)) > 0.05:
        print("  ⚠️  WARNING: Normalized spectrum deviates from median model")
        print("     Possible issues:")
        print("     - Outliers in reference points not fully removed")
        print("     - Model not properly normalized")
        print("     - Polynomial degree too high (overfitting)")
        print(
            f"Median model {np.median(model_flux)}, median observation flux {final_median}"
        )
    else:
        print("  ✅ Normalization successful!")

    print("=" * 60 + "\n")

    return normalized_obs, poly_func, obs_flux_interp


def iterative_flexure_correction(
    model_wave,
    model_flux,
    model_diff,
    obs_wave,
    obs_flux,
    threshold=0.1,
    poly_degree=7,
    n_iterations=5,
    flexure_smooth_width=100,
    sigma_clip=2.5,
    plot=True,
    verbose=True,
):
    """
    Итеративное исправление гнутия спектральных порядков.
    """

    if verbose:
        print("\n" + "=" * 70)
        print("ITERATIVE FLEXURE CORRECTION")
        print("=" * 70)

    # === ШАГ 0: Интерполяция наблюдений на сетку модели ===
    obs_flux_interp = np.interp(model_wave, obs_wave, obs_flux)

    # === ШАГ 1: Начальные реперные точки ===
    # Создаём маску один раз
    initial_mask = model_diff < threshold
    if np.sum(initial_mask) < 5:
        initial_mask = model_diff < threshold * 1.5
        if verbose:
            print(f"Relaxed threshold to {threshold * 1.5:.2f}")

    # Сохраняем индексы реперных точек
    ref_indices = np.where(initial_mask)[0]
    x_ref = model_wave[ref_indices]

    if verbose:
        print(f"Initial reference points: {len(x_ref)}")

    # === ШАГ 2: Итеративная коррекция ===
    current_obs_flux = obs_flux.copy()
    corrections = []
    correction_polys = []

    history = {
        "iterations": [],
        "rms_residuals": [],
        "median_spectrum": [],
        "flexure_amplitude": [],
    }

    for iteration in range(n_iterations):
        if verbose:
            print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")

        # 2.1 Интерполируем текущий спектр на сетку модели
        current_obs_interp = np.interp(model_wave, obs_wave, current_obs_flux)

        # 2.2 Вычисляем отношения для реперных точек
        y_fit = current_obs_interp[ref_indices] / model_flux[ref_indices]

        # 2.3 Очистка выбросов
        median_y = np.median(y_fit)
        mad = np.median(np.abs(y_fit - median_y))
        if mad > 0:
            clip_mask = np.abs(y_fit - median_y) < sigma_clip * mad
        else:
            clip_mask = np.ones(len(y_fit), dtype=bool)

        x_fit = x_ref[clip_mask]
        y_fit = y_fit[clip_mask]

        if verbose:
            print(f"  After sigma-clipping: {len(x_fit)} points")

        # 2.4 Дополнительная итеративная очистка
        for _ in range(2):
            if len(x_fit) < 10:
                break
            try:
                temp_degree = min(2, poly_degree)
                coeffs_temp = np.polyfit(x_fit, y_fit, temp_degree)
                poly_temp = np.poly1d(coeffs_temp)
                residuals = y_fit - poly_temp(x_fit)
                std_res = np.std(residuals)
                if std_res > 0:
                    mask_clean = np.abs(residuals) < sigma_clip * std_res
                    x_fit = x_fit[mask_clean]
                    y_fit = y_fit[mask_clean]
                else:
                    break
            except:
                break

        if verbose:
            print(f"  After iterative cleaning: {len(x_fit)} points")

        # 2.5 Подгонка полинома
        try:
            coeffs = np.polyfit(x_fit, y_fit, poly_degree)
            poly_func = np.poly1d(coeffs)
        except np.linalg.LinAlgError:
            if verbose:
                print(f"  WARNING: Polyfit failed, reducing degree")
            poly_degree = max(3, poly_degree - 1)
            coeffs = np.polyfit(x_fit, y_fit, poly_degree)
            poly_func = np.poly1d(coeffs)

        # 2.6 Измеряем остаточное гнутие
        poly_full = poly_func(obs_wave)
        temp_normalized = current_obs_flux / poly_full

        # Сглаживаем для измерения гнутия
        flexure_smooth = gaussian_filter1d(
            temp_normalized, sigma=flexure_smooth_width / 2.0, mode="reflect"
        )

        flexure_amplitude = np.std(flexure_smooth - 1.0)
        if verbose:
            print(f"  Flexure amplitude: {flexure_amplitude:.6f}")

        # 2.7 Вычисляем финальную коррекцию для этой итерации
        correction = poly_full * flexure_smooth
        corrections.append(correction)
        correction_polys.append(poly_func)

        # 2.8 Применяем коррекцию
        current_obs_flux = current_obs_flux / correction

        # 2.9 Сохраняем историю
        history["iterations"].append(iteration + 1)
        history["rms_residuals"].append(np.std(y_fit - poly_func(x_fit)))
        history["median_spectrum"].append(np.median(current_obs_flux))
        history["flexure_amplitude"].append(flexure_amplitude)

        # 2.10 Проверка сходимости
        if iteration > 0:
            improvement = (
                history["flexure_amplitude"][-2] - history["flexure_amplitude"][-1]
            ) / (history["flexure_amplitude"][-2] + 1e-10)
            if improvement < 0.01 and flexure_amplitude < 0.001:
                if verbose:
                    print(f"  Converged after {iteration + 1} iterations")
                break

        # 2.11 Обновление реперных точек для следующей итерации
        # Пересчитываем diff для текущего спектра
        current_diff = np.abs(current_obs_interp / model_flux - 1.0)

        if iteration < 2:
            # На первых итерациях используем исходный порог
            new_mask = model_diff < threshold
        else:
            # На поздних итерациях используем более строгий порог
            new_mask = (model_diff < threshold * 0.8) & (current_diff < 0.15)
            if np.sum(new_mask) < 100:
                new_mask = model_diff < threshold * 0.8

        # Обновляем индексы реперных точек
        ref_indices = np.where(new_mask)[0]
        x_ref = model_wave[ref_indices]

        if verbose:
            print(f"  Updated reference points: {len(x_ref)}")

    # === ШАГ 3: Финальная коррекция ===
    final_normalized = obs_flux.copy()
    for correction in corrections:
        final_normalized = final_normalized / correction

    # === ШАГ 4: Принудительная нормализация ===
    median_final = np.median(final_normalized)
    if abs(median_final - 1.0) > 0.01:
        if verbose:
            print(f"\nFinal median correction: {median_final:.4f} -> 1.0")
        final_normalized = final_normalized / median_final

    # === ШАГ 5: Визуализация ===
    if plot:
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        # 1. Эволюция RMS
        ax = axes[0, 0]
        ax.plot(history["iterations"], history["rms_residuals"], "bo-", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("RMS Residuals")
        ax.set_title("Evolution of fit quality")
        ax.grid(True, alpha=0.3)

        # 2. Эволюция гнутия
        ax = axes[0, 1]
        ax.plot(history["iterations"], history["flexure_amplitude"], "ro-", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Flexure amplitude (std)")
        ax.set_title("Evolution of flexure amplitude")
        ax.grid(True, alpha=0.3)

        # 3. Нормированный спектр (финальный)
        ax = axes[1, 0]
        mid = len(obs_wave) // 2
        window = min(500, len(obs_wave) // 4)
        idx_range = slice(mid - window, mid + window)
        ax.plot(
            obs_wave[idx_range],
            final_normalized[idx_range],
            "b-",
            linewidth=1,
            alpha=0.7,
            label="Final",
        )
        ax.axhline(y=1.0, color="r", linestyle="--", linewidth=2)
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Normalized Flux")
        ax.set_title("Final normalized spectrum (center)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Сравнение до/после (весь спектр)
        ax = axes[1, 1]
        smooth_final = gaussian_filter1d(final_normalized, sigma=20, mode="reflect")
        orig_norm = obs_flux / np.median(obs_flux)

        ax.plot(obs_wave, orig_norm, "b-", alpha=0.3, label="Original (scaled)")
        ax.plot(obs_wave, final_normalized, "r-", alpha=0.7, label="Corrected")
        ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5)
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Flux")
        ax.set_title("Spectrum before and after correction")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Остаточное гнутие
        ax = axes[2, 0]
        smooth_final_flux = gaussian_filter1d(
            final_normalized, sigma=50, mode="reflect"
        )
        residual_flexure = smooth_final_flux - 1.0
        ax.plot(obs_wave, residual_flexure, "g-", linewidth=1, alpha=0.7)
        ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
        ax.axhline(y=0.01, color="orange", linestyle=":", alpha=0.5)
        ax.axhline(y=-0.01, color="orange", linestyle=":", alpha=0.5)
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Residual flexure")
        ax.set_title(f"Residual flexure (RMS = {np.std(residual_flexure):.6f})")
        ax.grid(True, alpha=0.3)

        # 6. Гистограмма финального спектра
        ax = axes[2, 1]
        hist_data = final_normalized[
            (final_normalized > 0.5) & (final_normalized < 1.5)
        ]
        ax.hist(hist_data, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(x=1.0, color="r", linestyle="--", linewidth=2)
        ax.axvline(
            x=np.median(hist_data),
            color="orange",
            linestyle="--",
            label=f"Median = {np.median(hist_data):.4f}",
        )
        ax.set_xlabel("Normalized Flux")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of final spectrum")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Статистика
        print("\n" + "=" * 70)
        print("FINAL STATISTICS")
        print("=" * 70)
        print(f"Number of iterations: {len(history['iterations'])}")
        print(f"Final median: {np.median(final_normalized):.6f}")
        print(f"Final RMS (spectrum): {np.std(final_normalized):.6f}")
        print(f"Final flexure amplitude: {np.std(smooth_final_flux - 1.0):.6f}")
        if len(history["flexure_amplitude"]) > 1:
            print(
                f"Improvement factor: {history['flexure_amplitude'][0] / history['flexure_amplitude'][-1]:.2f}x"
            )
        print("=" * 70)

    return final_normalized, correction_polys, history


def correct_molecular_bands(
    obs_wave,
    obs_flux,
    molecular_template_wave,
    molecular_template_flux,
    continuum_regions=None,
    smooth_width=50,
    polynomial_degree=5,
    n_iterations=5,
    plot=True,
    verbose=True,
):
    """
    Коррекция спектра с молекулярными полосами.

    Алгоритм:
    1. Находит участки континуума (где нет молекулярного поглощения)
    2. По точкам континуума строит полиномиальную модель континуума
    3. Делит спектр на модель континуума
    4. Итеративно уточняет, используя молекулярный шаблон

    Parameters:
    -----------
    obs_wave : array - длины волн наблюдаемого спектра
    obs_flux : array - поток наблюдаемого спектра
    molecular_template_wave : array - длины волн шаблона молекулы
    molecular_template_flux : array - нормированный шаблон молекулы (0-1, где 1 = континуум)
    continuum_regions : list of tuples - [(wl1, wl2), ...] участки континуума
                         Если None, определяются автоматически
    smooth_width : int - ширина сглаживания для поиска континуума
    polynomial_degree : int - степень полинома для модели континуума
    n_iterations : int - число итераций
    plot : bool - показывать графики
    verbose : bool - печатать информацию

    Returns:
    --------
    normalized_flux : array - нормированный спектр
    continuum_model : array - модель континуума
    molecular_depth : array - глубина молекулярного поглощения
    """

    if verbose:
        print("\n" + "=" * 70)
        print("MOLECULAR BAND CORRECTION")
        print("=" * 70)

    # === ШАГ 1: Интерполяция молекулярного шаблона на сетку наблюдений ===
    molecular_template_interp = np.interp(
        obs_wave, molecular_template_wave, molecular_template_flux
    )

    # === ШАГ 2: Итеративное определение континуума ===

    # 2.1 Если участки континуума не заданы, определяем автоматически
    if continuum_regions is None:
        if verbose:
            print("Determining continuum regions automatically...")

        # Находим области, где молекулярное поглощение минимально
        # Сглаживаем спектр
        smooth_flux = gaussian_filter1d(obs_flux, sigma=smooth_width, mode="reflect")

        # Нормализуем сглаженный спектр
        smooth_norm = smooth_flux / np.median(smooth_flux)

        # Находим пики (участки с максимальным потоком)
        from scipy.signal import find_peaks

        peaks, properties = find_peaks(
            smooth_norm, height=0.95, distance=len(obs_wave) // 20, prominence=0.05
        )

        # Группируем пики в участки
        continuum_regions = []
        if len(peaks) > 2:
            # Сортируем пики
            peak_positions = obs_wave[peaks]

            # Группируем близкие пики
            current_region = [peak_positions[0]]
            for i in range(1, len(peak_positions)):
                if (
                    peak_positions[i] - peak_positions[i - 1]
                    < (obs_wave[-1] - obs_wave[0]) / 10
                ):
                    current_region.append(peak_positions[i])
                else:
                    if len(current_region) > 1:
                        continuum_regions.append(
                            (current_region[0], current_region[-1])
                        )
                    current_region = [peak_positions[i]]

            if len(current_region) > 1:
                continuum_regions.append((current_region[0], current_region[-1]))

        # Если не нашли, используем весь спектр
        if len(continuum_regions) < 2:
            continuum_regions = [(obs_wave[0], obs_wave[-1])]

        if verbose:
            print(f"  Found {len(continuum_regions)} continuum regions")

    # === ШАГ 3: Итеративная коррекция ===

    # Копируем исходный спектр
    current_flux = obs_flux.copy()

    # Сохраняем историю
    history = {"iterations": [], "rms_continuum": [], "molecular_depth": []}

    # Маска для точек, которые не являются молекулярными линиями (континуум)
    continuum_mask = np.zeros(len(obs_wave), dtype=bool)
    for wl1, wl2 in continuum_regions:
        continuum_mask |= (obs_wave >= wl1) & (obs_wave <= wl2)

    # Дополнительная маска: удаляем сильные линии (где шаблон < 0.7)
    line_mask = molecular_template_interp > 0.7
    continuum_mask = continuum_mask & line_mask

    if verbose:
        print(f"Initial continuum points: {np.sum(continuum_mask)}")

    for iteration in range(n_iterations):
        if verbose:
            print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")

        # 3.1 Выбираем точки континуума
        x_cont = obs_wave[continuum_mask]
        y_cont = current_flux[continuum_mask]

        # 3.2 Подгоняем полином по точкам континуума
        if len(x_cont) > polynomial_degree + 1:
            try:
                coeffs = np.polyfit(x_cont, y_cont, polynomial_degree)
                continuum_poly = np.poly1d(coeffs)
                continuum_model = continuum_poly(obs_wave)
            except:
                # Если подгонка не удалась, используем медиану
                if verbose:
                    print("  WARNING: Polyfit failed, using median")
                continuum_model = np.ones(len(obs_wave)) * np.median(y_cont)
        else:
            continuum_model = np.ones(len(obs_wave)) * np.median(y_cont)

        # 3.3 Нормируем спектр на континуум
        normalized_flux = current_flux / continuum_model

        # 3.4 Вычисляем глубину молекулярного поглощения
        molecular_depth = 1.0 - normalized_flux / molecular_template_interp
        molecular_depth = np.clip(molecular_depth, 0, 1)

        # 3.5 Обновляем маску континуума
        # Ищем точки, где молекулярное поглощение минимально
        # и спектр близок к континууму
        threshold = 0.05 + 0.05 * (1 - np.exp(-iteration / 3))
        new_continuum_mask = (molecular_depth < threshold) & (continuum_mask)

        # Добавляем точки, где шаблон близок к континууму
        new_continuum_mask |= molecular_template_interp > 0.9

        # Добавляем немного сглаживания для непрерывности
        # Расширяем маску на соседние точки (для континуума)
        kernel = np.ones(5) / 5
        smoothed_mask = (
            np.convolve(new_continuum_mask.astype(float), kernel, mode="same") > 0.3
        )
        new_continuum_mask = smoothed_mask

        # Обновляем маску
        continuum_mask = new_continuum_mask

        if verbose:
            print(f"  Continuum points: {np.sum(continuum_mask)}")
            print(f"  Median continuum: {np.median(continuum_model):.4f}")
            print(f"  Mean molecular depth: {np.mean(molecular_depth):.4f}")

        # Сохраняем историю
        history["iterations"].append(iteration + 1)
        history["rms_continuum"].append(
            np.std(
                y_cont - continuum_poly(x_cont)
                if len(x_cont) > polynomial_degree
                else 0
            )
        )
        history["molecular_depth"].append(np.mean(molecular_depth))

        # Проверка сходимости
        if iteration > 0:
            depth_change = abs(
                history["molecular_depth"][-1] - history["molecular_depth"][-2]
            )
            if depth_change < 0.001:
                if verbose:
                    print(f"  Converged after {iteration + 1} iterations")
                break

    # === ШАГ 4: Финальная коррекция ===

    # 4.1 Финальная модель континуума (усредняем по последним итерациям)
    # или берём последнюю
    continuum_model_final = continuum_model

    # 4.2 Финальная нормировка
    normalized_flux_final = obs_flux / continuum_model_final

    # 4.3 Вычитаем молекулярный континуум (для измерения EW)
    # Это спектр без молекулярных полос (только линии)
    molecular_corrected = normalized_flux_final / molecular_template_interp

    # === ШАГ 5: Визуализация ===
    if plot:
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))

        # 1. Исходный спектр с континуумом
        ax = axes[0, 0]
        ax.plot(obs_wave, obs_flux, "b-", alpha=0.5, label="Original")
        ax.plot(
            obs_wave, continuum_model_final, "r-", linewidth=2, label="Continuum model"
        )

        # Отмечаем точки континуума
        cont_wave = obs_wave[continuum_mask]
        cont_flux = obs_flux[continuum_mask]
        ax.scatter(
            cont_wave, cont_flux, c="green", s=10, alpha=0.5, label="Continuum points"
        )

        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Flux")
        ax.set_title("Original spectrum with continuum model")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Молекулярный шаблон
        ax = axes[0, 1]
        ax.plot(
            obs_wave,
            molecular_template_interp,
            "g-",
            linewidth=1.5,
            label="Molecular template",
        )
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Template flux")
        ax.set_title("Molecular template (normalized)")
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Нормированный спектр
        ax = axes[1, 0]
        ax.plot(obs_wave, normalized_flux_final, "b-", alpha=0.7, label="Normalized")
        ax.axhline(y=1.0, color="r", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Normalized Flux")
        ax.set_title("Normalized spectrum (continuum = 1)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Спектр после вычитания молекулярных полос
        ax = axes[1, 1]
        # Показываем небольшой участок для наглядности
        mid = len(obs_wave) // 2
        window = min(200, len(obs_wave) // 8)
        idx_range = slice(mid - window, mid + window)

        ax.plot(
            obs_wave[idx_range],
            normalized_flux_final[idx_range],
            "b-",
            alpha=0.5,
            label="Normalized",
        )
        ax.plot(
            obs_wave[idx_range],
            molecular_template_interp[idx_range],
            "g--",
            alpha=0.7,
            label="Molecular template",
        )
        ax.plot(
            obs_wave[idx_range],
            molecular_corrected[idx_range],
            "r-",
            alpha=0.7,
            label="Molecular corrected",
        )
        ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Flux")
        ax.set_title("Zoom: molecular band correction")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Глубина молекулярного поглощения
        ax = axes[2, 0]
        ax.plot(obs_wave, molecular_depth, "r-", alpha=0.7, label="Molecular depth")
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Depth (1 - normalized/template)")
        ax.set_title("Molecular absorption depth")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Гистограмма
        ax = axes[2, 1]
        hist_data = normalized_flux_final[
            (normalized_flux_final > 0.5) & (normalized_flux_final < 1.5)
        ]
        ax.hist(hist_data, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(x=1.0, color="r", linestyle="--", linewidth=2)
        ax.axvline(
            x=np.median(hist_data),
            color="orange",
            linestyle="--",
            label=f"Median = {np.median(hist_data):.4f}",
        )
        ax.set_xlabel("Normalized Flux")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of normalized spectrum")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Статистика
        print("\n" + "=" * 70)
        print("FINAL STATISTICS")
        print("=" * 70)
        print(f"Number of iterations: {len(history['iterations'])}")
        print(f"Continuum points used: {np.sum(continuum_mask)}")
        print(f"Median normalized flux: {np.median(normalized_flux_final):.6f}")
        print(f"Mean molecular depth: {np.mean(molecular_depth):.6f}")
        print(f"Max molecular depth: {np.max(molecular_depth):.6f}")
        print("=" * 70)

    return normalized_flux_final, continuum_model_final, molecular_depth


def create_molecular_template(
    wavelength,
    cross_section,
    column_density=None,
    normalize=True,
    smooth_sigma=None,
    max_optical_depth=20,  # Ограничение на оптическую глубину
    plot=True,
    verbose=True,
):
    """
    Создание нормированного шаблона молекулярного поглощения из кросс-секции.
    Исправленная версия с защитой от численных переполнений.
    """

    if verbose:
        print("\n" + "=" * 70)
        print("CREATING MOLECULAR TEMPLATE")
        print("=" * 70)

    # === ШАГ 1: Нормализация кросс-секции ===
    # Используем логарифмическую нормализацию для больших динамических диапазонов
    cs_min = (
        np.min(cross_section[cross_section > 0]) if np.any(cross_section > 0) else 1e-10
    )
    cs_max = np.max(cross_section)

    if verbose:
        print(f"Cross-section range: {cs_min:.2e} - {cs_max:.2e}")
        print(f"Dynamic range: {cs_max / cs_min:.2e}")

    # Логарифмическая нормализация для лучшего распределения
    # cross_section_norm = (cross_section - cs_min) / (cs_max - cs_min)
    # Но лучше использовать относительную нормализацию
    cross_section_norm = cross_section / cs_max
    cross_section_norm = np.clip(cross_section_norm, 0, 1)

    # === ШАГ 2: Определение столбцовой плотности ===
    if column_density is None:
        # Автоматический подбор с учётом численной стабильности

        # Находим 95-й процентиль (игнорируем выбросы)
        sorted_cs = np.sort(cross_section_norm)
        percentile_95 = np.percentile(sorted_cs, 95)
        percentile_50 = np.percentile(sorted_cs, 50)

        # Используем средний уровень для определения N
        # Целевая глубина: хотим, чтобы на пиках было ~10-20% пропускания
        target_depth = 0.15  # 15% пропускания на пиках

        # N = -ln(target_depth) / percentile_95
        if percentile_95 > 1e-10:
            column_density = -np.log(target_depth) / percentile_95
        else:
            column_density = 1.0

        # Ограничиваем N, чтобы избежать переполнения
        max_N = max_optical_depth / (percentile_95 + 1e-10)
        column_density = min(column_density, max_N)

        if verbose:
            print(f"  Auto-determined column density: {column_density:.2e}")
            print(f"  Peak optical depth: {column_density * percentile_95:.2f}")
            print(f"  Max allowed optical depth: {max_optical_depth}")
    else:
        if verbose:
            print(f"Using provided column density: {column_density:.2e}")

    # === ШАГ 3: Вычисление коэффициента пропускания ===
    # Вычисляем оптическую глубину с ограничением
    optical_depth = column_density * cross_section_norm
    optical_depth = np.clip(optical_depth, 0, max_optical_depth)

    # T(λ) = exp(-τ)
    template_flux = np.exp(-optical_depth)

    # Защита от численных проблем
    template_flux = np.nan_to_num(template_flux, nan=0.0, posinf=0.0, neginf=0.0)
    template_flux = np.clip(template_flux, 1e-10, 1.0)

    if verbose:
        print(
            f"Template flux range: {np.min(template_flux):.6f} - {np.max(template_flux):.6f}"
        )
        print(f"Mean template flux: {np.mean(template_flux):.6f}")
        print(f"Points with zero flux: {np.sum(template_flux < 1e-10)}")

    # === ШАГ 4: Опциональное сглаживание ===
    if smooth_sigma is not None and smooth_sigma > 0:
        template_flux = gaussian_filter1d(
            template_flux, sigma=smooth_sigma, mode="reflect"
        )
        template_flux = np.clip(template_flux, 1e-10, 1.0)
        if verbose:
            print(f"Applied Gaussian smoothing (sigma={smooth_sigma})")

    # === ШАГ 5: Нормировка ===
    if normalize:
        max_flux = np.max(template_flux)
        if max_flux > 0:
            template_flux = template_flux / max_flux
        if verbose:
            print("Template normalized to maximum = 1.0")

    # === ШАГ 6: Фильтрация шума ===
    # Удаляем изолированные пики (артефакты)
    from scipy.signal import medfilt

    template_flux = medfilt(template_flux, kernel_size=3)
    template_flux = np.clip(template_flux, 0, 1)

    # === ШАГ 7: Сохранение параметров ===
    params = {
        "column_density": column_density,
        "max_cross_section": cs_max,
        "mean_cross_section": np.mean(cross_section_norm),
        "min_flux": np.min(template_flux),
        "max_flux": np.max(template_flux),
        "mean_flux": np.mean(template_flux),
        "total_absorption": np.sum(1.0 - template_flux) * np.gradient(wavelength)[0],
    }

    # === ШАГ 8: Визуализация ===
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Кросс-секция (лог-масштаб для наглядности)
        ax = axes[0, 0]
        ax.plot(wavelength, cross_section_norm, "b-", linewidth=1, alpha=0.7)
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Normalized Cross-section")
        ax.set_title("Cross-section (normalized)")
        ax.grid(True, alpha=0.3)

        # 2. Шаблон пропускания
        ax = axes[0, 1]
        ax.plot(wavelength, template_flux, "r-", linewidth=1.5)
        ax.axhline(y=1.0, color="k", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.axhline(y=0.0, color="k", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Flux (Transmission)")
        ax.set_title(f"Template (N={column_density:.2e})")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        # 3. Оптическая глубина
        ax = axes[1, 0]
        optical_depth_plot = -np.log(template_flux + 1e-10)
        ax.plot(wavelength, optical_depth_plot, "g-", linewidth=1.5)
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Optical Depth")
        ax.set_title(f"Optical Depth (max={np.max(optical_depth_plot):.2f})")
        ax.grid(True, alpha=0.3)

        # 4. Гистограмма шаблона
        ax = axes[1, 1]
        ax.hist(template_flux, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(x=1.0, color="r", linestyle="--", label="Continuum")
        ax.axvline(
            x=np.median(template_flux),
            color="orange",
            linestyle="--",
            label=f"Median = {np.median(template_flux):.4f}",
        )
        ax.set_xlabel("Transmission")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of transmission")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Статистика
        print("\n" + "=" * 70)
        print("TEMPLATE STATISTICS")
        print("=" * 70)
        print(f"Column density: {params['column_density']:.2e}")
        print(f"Minimum flux: {params['min_flux']:.6f}")
        print(f"Maximum flux: {params['max_flux']:.6f}")
        print(f"Mean flux: {params['mean_flux']:.6f}")
        print(f"Total absorption (EW-like): {params['total_absorption']:.3f}")
        print("=" * 70)

    return template_flux, params


def fit_column_density(
    obs_wave,
    obs_flux,
    wavelength_cs,
    cross_section,
    continuum_regions=None,
    n_iterations=20,
    plot=True,
    verbose=True,
):
    """
    Автоматическая подгонка столбцовой плотности.
    Исправленная версия с защитой от nan.
    """

    if verbose:
        print("\n" + "=" * 70)
        print("FITTING COLUMN DENSITY")
        print("=" * 70)

    # Интерполяция кросс-секции на сетку наблюдений
    cross_section_interp = np.interp(obs_wave, wavelength_cs, cross_section)

    # Нормализация (защита от нулевых значений)
    cs_max = np.max(cross_section_interp)
    if cs_max > 0:
        cross_section_norm = cross_section_interp / cs_max
    else:
        cross_section_norm = cross_section_interp

    # Ограничиваем значения для численной стабильности
    cross_section_norm = np.clip(cross_section_norm, 0, 1)

    # Пробуем разные значения N в логарифмическом масштабе
    # Ограничиваем диапазон, чтобы избежать переполнения
    logN_min = -3  # N = 0.001
    logN_max = 5  # N = 100000

    N_values = np.logspace(logN_min, logN_max, n_iterations)

    # Метрики качества
    smoothness = []
    median_spectrum = []
    valid_indices = []

    for i, N in enumerate(N_values):
        try:
            # Создаём шаблон с защитой от переполнения
            optical_depth = N * cross_section_norm
            optical_depth = np.clip(optical_depth, 0, 20)  # Ограничиваем

            template = np.exp(-optical_depth)
            template = np.nan_to_num(template, nan=0.0, posinf=0.0, neginf=0.0)

            max_template = np.max(template)
            if max_template > 0:
                template = template / max_template
            else:
                template = np.ones_like(template)

            # Нормируем спектр
            # Защита от деления на ноль
            template_safe = np.where(template > 1e-10, template, 1e-10)
            normalized = obs_flux / template_safe

            # Удаляем выбросы для стабильности
            normalized_clipped = np.clip(normalized, 0.1, 10)

            # Вычисляем гладкость (стандартное отклонение сглаженного спектра)
            smooth = gaussian_filter1d(normalized_clipped, sigma=50, mode="reflect")
            smoothness_value = np.std(smooth - 1.0)

            if not np.isnan(smoothness_value) and not np.isinf(smoothness_value):
                smoothness.append(smoothness_value)
                median_spectrum.append(np.median(normalized_clipped))
                valid_indices.append(i)

                if verbose:
                    print(f"N={N:.2e}: smoothness={smoothness_value:.6f}")
        except Exception as e:
            if verbose:
                print(f"N={N:.2e}: failed - {str(e)}")
            continue

    # Проверка, что есть валидные результаты
    if len(smoothness) == 0:
        print("WARNING: No valid results! Using default N=1.0")
        return 1.0, [], N_values

    # Находим N, дающее максимальную гладкость
    best_idx = np.argmin(smoothness)
    best_N = N_values[valid_indices[best_idx]]
    best_smoothness = smoothness[best_idx]

    if verbose:
        print(f"\nBest column density: {best_N:.2e}")
        print(f"Best smoothness: {best_smoothness:.6f}")

    # Визуализация
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax = axes[0]
        ax.plot(N_values[valid_indices], smoothness, "bo-", linewidth=1.5)
        ax.axvline(x=best_N, color="r", linestyle="--", label=f"Best N = {best_N:.2e}")
        ax.set_xscale("log")
        ax.set_xlabel("Column Density (N)")
        ax.set_ylabel("Smoothness (std)")
        ax.set_title("Smoothness vs Column Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(N_values[valid_indices], median_spectrum, "go-", linewidth=1.5)
        ax.axvline(x=best_N, color="r", linestyle="--", label=f"Best N = {best_N:.2e}")
        ax.axhline(y=1.0, color="k", linestyle=":", alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("Column Density (N)")
        ax.set_ylabel("Median Spectrum")
        ax.set_title("Median Spectrum vs Column Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return best_N, smoothness, N_values[valid_indices]


def correct_spectrum_with_molecular_bands(
    obs_wave,
    obs_flux,
    model_wave,
    model_flux,
    molecular_template_wave,
    molecular_template_flux,
    continuum_regions=None,
    poly_degree=5,
    n_iterations=5,
    smooth_width=30,
    target_median=None,
    plot=True,
    verbose=True,
):
    """
    Коррекция спектра с молекулярными полосами.

    Алгоритм:
    1. Находит участки чистого континуума (где нет молекулярных полос)
    2. Строит модель континуума по этим участкам
    3. Нормирует спектр на модельный континуум
    4. Использует молекулярный шаблон для коррекции внутри полос
    5. Приводит медиану к значению модельного континуума

    Parameters:
    -----------
    obs_wave, obs_flux : массивы наблюдаемого спектра
    model_wave, model_flux : массивы модельного спектра (нормированного)
    molecular_template_wave, molecular_template_flux : шаблон молекулы (0-1)
    continuum_regions : list of tuples - [(wl1, wl2), ...] участки континуума
    poly_degree : int - степень полинома для континуума
    n_iterations : int - число итераций
    smooth_width : int - ширина сглаживания
    target_median : float - целевая медиана (если None, берётся из модели)
    """

    if verbose:
        print("\n" + "=" * 70)
        print("MOLECULAR BAND CORRECTION WITH CONTINUUM CONSTRAINT")
        print("=" * 70)

    # === ШАГ 1: Интерполяция на общую сетку ===
    # Модель на сетку наблюдений
    model_flux_interp = np.interp(obs_wave, model_wave, model_flux)

    # Молекулярный шаблон на сетку наблюдений
    molecular_template_interp = np.interp(
        obs_wave, molecular_template_wave, molecular_template_flux
    )

    # === ШАГ 2: Определение целевой медианы ===
    if target_median is None:
        # Берём медиану модельного спектра в областях континуума
        if continuum_regions is not None:
            cont_mask = np.zeros(len(obs_wave), dtype=bool)
            for wl1, wl2 in continuum_regions:
                cont_mask |= (obs_wave >= wl1) & (obs_wave <= wl2)
            target_median = np.median(model_flux_interp[cont_mask])
        else:
            target_median = np.median(model_flux_interp)

        if verbose:
            print(f"Target median (from model): {target_median:.4f}")

    # === ШАГ 3: Итеративное определение континуума ===

    # 3.1 Начальная маска континуума
    if continuum_regions is not None:
        continuum_mask = np.zeros(len(obs_wave), dtype=bool)
        for wl1, wl2 in continuum_regions:
            continuum_mask |= (obs_wave >= wl1) & (obs_wave <= wl2)
        if verbose:
            print(f"Using provided continuum regions: {len(continuum_regions)} regions")
    else:
        # Автоматическое определение
        if verbose:
            print("Automatically determining continuum regions...")

        # Находим участки, где молекулярный шаблон близок к 1 (нет поглощения)
        continuum_mask = molecular_template_interp > 0.95

        # Дополнительно: ищем пики в спектре
        smooth_flux = gaussian_filter1d(obs_flux, sigma=smooth_width, mode="reflect")
        smooth_norm = smooth_flux / np.median(smooth_flux)

        # Находим пики
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(smooth_norm, height=0.95, distance=len(obs_wave) // 20)

        # Добавляем пики в маску континуума
        peak_mask = np.zeros(len(obs_wave), dtype=bool)
        for peak in peaks:
            # Расширяем окно вокруг пика
            half_width = max(5, int(len(obs_wave) / 200))
            start = max(0, peak - half_width)
            end = min(len(obs_wave), peak + half_width)
            peak_mask[start:end] = True

        continuum_mask = continuum_mask | peak_mask

    # Количество точек континуума
    n_cont = np.sum(continuum_mask)
    if verbose:
        print(f"Initial continuum points: {n_cont}")

    if n_cont < 10:
        print("WARNING: Too few continuum points! Using full spectrum.")
        continuum_mask = np.ones(len(obs_wave), dtype=bool)

    # === ШАГ 4: Итеративная коррекция ===

    current_flux = obs_flux.copy()
    current_mask = continuum_mask.copy()

    history = {
        "iterations": [],
        "median_continuum": [],
        "rms_continuum": [],
        "median_spectrum": [],
    }

    for iteration in range(n_iterations):
        if verbose:
            print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")

        # 4.1 Выбираем точки континуума
        x_cont = obs_wave[current_mask]
        y_cont = current_flux[current_mask]

        if len(x_cont) < poly_degree + 2:
            if verbose:
                print(f"  Too few points ({len(x_cont)}), using median")
            continuum_model = np.ones(len(obs_wave)) * np.median(y_cont)
        else:
            try:
                # Подгонка полинома
                coeffs = np.polyfit(x_cont, y_cont, poly_degree)
                continuum_poly = np.poly1d(coeffs)
                continuum_model = continuum_poly(obs_wave)
            except:
                if verbose:
                    print("  Polyfit failed, using median")
                continuum_model = np.ones(len(obs_wave)) * np.median(y_cont)

        # 4.2 Нормируем спектр на континуум
        normalized_flux = current_flux / continuum_model

        # 4.3 Учитываем молекулярный шаблон
        # Ищем точки, где шаблон показывает поглощение, но спектр их не показывает
        # (т.е. континуум занижен)

        # Вычисляем ожидаемый поток с учётом молекулярного поглощения
        expected_flux = continuum_model * molecular_template_interp

        # Там где наблюдаемый поток > ожидаемый, континуум занижен
        correction_mask = (current_flux > expected_flux * 1.05) & (
            molecular_template_interp < 0.95
        )

        if np.sum(correction_mask) > 0:
            if verbose:
                print(
                    f"  Found {np.sum(correction_mask)} points with underestimated continuum"
                )

            # Корректируем эти точки
            # Повышаем континуум в этих областях
            correction_factor = (
                current_flux[correction_mask] / expected_flux[correction_mask]
            )
            median_correction = np.median(correction_factor)

            if not np.isnan(median_correction) and median_correction > 1.0:
                # Применяем коррекцию к модели континуума
                continuum_model[correction_mask] = (
                    continuum_model[correction_mask] * median_correction
                )

                if verbose:
                    print(f"  Correction factor: {median_correction:.4f}")

        # 4.4 Обновляем маску континуума
        # Используем молекулярный шаблон для идентификации чистого континуума
        new_mask = molecular_template_interp > 0.9

        # Добавляем точки, где спектр близок к континууму
        norm_diff = np.abs(normalized_flux - 1.0)
        new_mask |= (norm_diff < 0.05) & (molecular_template_interp > 0.7)

        # Расширяем маску для непрерывности
        kernel = np.ones(5) / 5
        smoothed_mask = np.convolve(new_mask.astype(float), kernel, mode="same") > 0.3
        new_mask = smoothed_mask

        # Обновляем
        current_mask = new_mask

        # 4.5 Применяем коррекцию к спектру
        current_flux = obs_flux / continuum_model

        # 4.6 Сохраняем историю
        history["iterations"].append(iteration + 1)
        history["median_continuum"].append(np.median(continuum_model))
        history["rms_continuum"].append(
            np.std(y_cont - continuum_poly(x_cont) if len(x_cont) > poly_degree else 0)
        )
        history["median_spectrum"].append(np.median(current_flux))

        if verbose:
            print(f"  Continuum median: {np.median(continuum_model):.4f}")
            print(f"  Spectrum median: {np.median(current_flux):.4f}")
            print(f"  Continuum points: {np.sum(current_mask)}")

        # Проверка сходимости
        if iteration > 0:
            change = abs(
                history["median_spectrum"][-1] - history["median_spectrum"][-2]
            )
            if change < 0.001:
                if verbose:
                    print(f"  Converged after {iteration + 1} iterations")
                break

    # === ШАГ 5: Финальная нормализация на целевое значение ===

    # 5.1 Финальная модель континуума
    continuum_model_final = continuum_model

    # 5.2 Нормируем спектр
    normalized_flux_final = obs_flux / continuum_model_final

    # 5.3 Приводим медиану к целевому значению
    current_median = np.median(normalized_flux_final)

    # Используем только области континуума для определения медианы
    cont_median = np.median(normalized_flux_final[current_mask])

    if verbose:
        print(f"\nFinal median (spectrum): {current_median:.4f}")
        print(f"Final median (continuum): {cont_median:.4f}")
        print(f"Target median: {target_median:.4f}")

    # Нормируем на медиану континуума, а затем масштабируем к целевому значению
    if cont_median > 0:
        # Сначала приводим континуум к 1
        scale_to_cont = 1.0 / cont_median
        normalized_flux_final = normalized_flux_final * scale_to_cont

        # Затем масштабируем к целевому значению
        scale_to_target = target_median / 1.0  # т.к. континуум теперь = 1
        normalized_flux_final = normalized_flux_final * scale_to_target

        if verbose:
            print(f"\nApplied scaling:")
            print(f"  Scale to continuum = 1: {scale_to_cont:.4f}")
            print(f"  Scale to target: {scale_to_target:.4f}")

    # === ШАГ 6: Финальная проверка ===
    final_median = np.median(normalized_flux_final)
    final_cont_median = np.median(normalized_flux_final[current_mask])

    if verbose:
        print(f"\nFinal statistics:")
        print(f"  Spectrum median: {final_median:.4f}")
        print(f"  Continuum median: {final_cont_median:.4f}")
        print(f"  Target median: {target_median:.4f}")
        print(f"  Error: {final_cont_median - target_median:.4f}")

    # === ШАГ 7: Визуализация ===
    if plot:
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))

        # 1. Исходный спектр с континуумом
        ax = axes[0, 0]
        ax.plot(obs_wave, obs_flux, "b-", alpha=0.5, label="Original")
        ax.plot(obs_wave, continuum_model_final, "r-", linewidth=2, label="Continuum")

        # Отмечаем точки континуума
        cont_wave = obs_wave[current_mask]
        cont_flux = obs_flux[current_mask]
        ax.scatter(
            cont_wave, cont_flux, c="green", s=5, alpha=0.5, label="Continuum points"
        )

        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Flux")
        ax.set_title("Original spectrum with continuum")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Молекулярный шаблон
        ax = axes[0, 1]
        ax.plot(obs_wave, molecular_template_interp, "g-", linewidth=1.5)
        ax.axhline(
            y=0.9, color="orange", linestyle="--", alpha=0.5, label="Threshold 0.9"
        )
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Transmission")
        ax.set_title("Molecular template")
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Нормированный спектр (финальный)
        ax = axes[1, 0]
        ax.plot(obs_wave, normalized_flux_final, "b-", alpha=0.7, label="Normalized")
        ax.axhline(
            y=target_median,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Target = {target_median:.4f}",
        )
        ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Normalized Flux")
        ax.set_title("Final normalized spectrum")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Zoom с молекулярными полосами
        ax = axes[1, 1]
        mid = len(obs_wave) // 2
        window = min(200, len(obs_wave) // 10)
        idx_range = slice(mid - window, mid + window)

        ax.plot(
            obs_wave[idx_range],
            normalized_flux_final[idx_range],
            "b-",
            alpha=0.7,
            label="Normalized",
        )
        ax.plot(
            obs_wave[idx_range],
            molecular_template_interp[idx_range] * target_median,
            "g--",
            alpha=0.7,
            label="Template",
        )
        ax.axhline(y=target_median, color="r", linestyle="--", alpha=0.5)
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Flux")
        ax.set_title("Zoom: molecular bands")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Эволюция
        ax = axes[2, 0]
        ax.plot(
            history["iterations"],
            history["median_spectrum"],
            "bo-",
            linewidth=1.5,
            label="Spectrum median",
        )
        ax.plot(
            history["iterations"],
            history["median_continuum"],
            "ro-",
            linewidth=1.5,
            label="Continuum median",
        )
        ax.axhline(y=target_median, color="k", linestyle="--", label="Target")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Median")
        ax.set_title("Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Гистограмма
        ax = axes[2, 1]
        hist_data = normalized_flux_final[
            (normalized_flux_final > 0.3) & (normalized_flux_final < 1.5)
        ]
        ax.hist(hist_data, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(x=target_median, color="r", linestyle="--", linewidth=2)
        ax.axvline(
            x=np.median(hist_data),
            color="orange",
            linestyle="--",
            label=f"Median = {np.median(hist_data):.4f}",
        )
        ax.set_xlabel("Normalized Flux")
        ax.set_ylabel("Count")
        ax.set_title("Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return normalized_flux_final, continuum_model_final, current_mask


def normalize_echelle_order(
    obs_wave,
    obs_flux,
    molecular_template_wave,
    molecular_template_flux,
    order_range=None,
    poly_degree=5,
    n_iterations=5,
    smooth_width=30,
    target_median=1.0,
    plot=True,
    verbose=True,
):
    """
    Нормировка одного порядка эшелле-спектра с учётом молекулярных полос.

    Parameters:
    -----------
    obs_wave, obs_flux : массивы для одного порядка
    molecular_template_wave, molecular_template_flux : шаблон молекулы
    order_range : tuple - (wl_min, wl_max) для порядка
    poly_degree : int - степень полинома для локального континуума
    n_iterations : int - число итераций
    smooth_width : int - ширина сглаживания
    target_median : float - целевая медиана континуума (обычно 1.0)
    """

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"NORMALIZING ECHELLE ORDER")
        if order_range:
            print(f"Range: {order_range[0]:.1f} - {order_range[1]:.1f} Å")
        print(f"{'=' * 70}")

    # === ШАГ 1: Интерполяция шаблона на сетку порядка ===
    molecular_template_interp = np.interp(
        obs_wave, molecular_template_wave, molecular_template_flux
    )

    # === ШАГ 2: Итеративное определение локального континуума ===

    # 2.1 Начальная маска: точки, где молекулярное поглощение минимально
    # Используем более высокий порог для эшелле (0.85 вместо 0.9)
    continuum_mask = molecular_template_interp > 0.85

    # 2.2 Добавляем точки с высоким потоком (потенциальный континуум)
    smooth_flux = gaussian_filter1d(obs_flux, sigma=smooth_width, mode="reflect")

    # Находим пики в сглаженном спектре
    from scipy.signal import find_peaks

    peaks, properties = find_peaks(
        smooth_flux,
        height=np.percentile(smooth_flux, 70),
        distance=max(10, len(obs_wave) // 50),
        prominence=np.std(smooth_flux) * 0.5,
    )

    if len(peaks) > 3:
        peak_mask = np.zeros(len(obs_wave), dtype=bool)
        for peak in peaks:
            half_width = max(3, int(len(obs_wave) / 100))
            start = max(0, peak - half_width)
            end = min(len(obs_wave), peak + half_width)
            peak_mask[start:end] = True
        continuum_mask = continuum_mask | peak_mask

    # 2.3 Убеждаемся, что есть достаточно точек
    if np.sum(continuum_mask) < 10:
        if verbose:
            print("  Too few continuum points, using broader mask")
        continuum_mask = molecular_template_interp > 0.7

    if verbose:
        print(f"  Initial continuum points: {np.sum(continuum_mask)}")

    # === ШАГ 3: Итеративная подгонка ===

    current_flux = obs_flux.copy()
    current_mask = continuum_mask.copy()

    history = {
        "iterations": [],
        "continuum_median": [],
        "spectrum_median": [],
        "rms_continuum": [],
        "n_points": [],
    }

    for iteration in range(n_iterations):
        if verbose:
            print(f"\n  Iteration {iteration + 1}/{n_iterations}")

        # 3.1 Точки континуума
        x_cont = obs_wave[current_mask]
        y_cont = current_flux[current_mask]

        if len(x_cont) < poly_degree + 2:
            if verbose:
                print(f"    Too few points ({len(x_cont)}), using median")
            continuum_model = np.ones(len(obs_wave)) * np.median(y_cont)
        else:
            try:
                # Взвешенная подгонка: больший вес точкам с меньшим молекулярным поглощением
                weights = molecular_template_interp[current_mask]
                weights = weights / np.max(weights)
                weights = np.clip(weights, 0.1, 1.0)

                coeffs = np.polyfit(x_cont, y_cont, poly_degree, w=weights)
                continuum_poly = np.poly1d(coeffs)
                continuum_model = continuum_poly(obs_wave)

                # Защита от экстраполяции на краях
                # Используем медиану на краях, где полином может быть нестабильным
                edge_width = len(obs_wave) // 20
                edge_median = np.median(y_cont)
                continuum_model[:edge_width] = np.clip(
                    continuum_model[:edge_width], 0.5 * edge_median, 2 * edge_median
                )
                continuum_model[-edge_width:] = np.clip(
                    continuum_model[-edge_width:], 0.5 * edge_median, 2 * edge_median
                )

            except Exception as e:
                if verbose:
                    print(f"    Polyfit failed: {e}, using median")
                continuum_model = np.ones(len(obs_wave)) * np.median(y_cont)

        # 3.2 Нормируем спектр
        normalized_flux = current_flux / continuum_model

        # 3.3 Обновляем маску континуума
        # Критерии для точек континуума:
        # 1. Малое молекулярное поглощение
        # 2. Нормированный поток близок к 1
        # 3. Не является выбросом

        condition1 = molecular_template_interp > 0.85
        condition2 = np.abs(normalized_flux - 1.0) < 0.1
        condition3 = np.abs(normalized_flux - 1.0) < 0.2  # более мягкое условие

        new_mask = condition1 & (condition2 | condition3)

        # Добавляем точки с высоким потоком (потенциальный континуум)
        flux_percentile = np.percentile(current_flux, 80)
        new_mask |= (current_flux > flux_percentile) & (molecular_template_interp > 0.7)

        # Расширяем маску для непрерывности
        kernel = np.ones(5) / 5
        smoothed_mask = np.convolve(new_mask.astype(float), kernel, mode="same") > 0.3
        new_mask = smoothed_mask & (molecular_template_interp > 0.6)

        # Убеждаемся, что есть достаточно точек
        if np.sum(new_mask) < 5:
            if verbose:
                print("    Too few points after update, keeping old mask")
            new_mask = current_mask

        # 3.4 Применяем коррекцию
        current_flux = obs_flux / continuum_model
        current_mask = new_mask

        # 3.5 Сохраняем историю
        history["iterations"].append(iteration + 1)
        history["continuum_median"].append(np.median(continuum_model))
        history["spectrum_median"].append(np.median(current_flux))
        history["n_points"].append(np.sum(current_mask))

        if verbose:
            print(f"    Continuum median: {np.median(continuum_model):.4f}")
            print(f"    Spectrum median: {np.median(current_flux):.4f}")
            print(f"    Continuum points: {np.sum(current_mask)}")

        # Проверка сходимости
        if iteration > 0:
            change = abs(
                history["spectrum_median"][-1] - history["spectrum_median"][-2]
            )
            if change < 0.001:
                if verbose:
                    print(f"    Converged")
                break

    # === ШАГ 4: Финальная нормализация ===

    # 4.1 Финальная модель континуума
    continuum_model_final = continuum_model

    # 4.2 Нормируем спектр
    normalized_flux_final = obs_flux / continuum_model_final

    # 4.3 Приводим медиану к целевому значению
    # Используем только точки континуума
    cont_median = np.median(normalized_flux_final[current_mask])

    if cont_median > 0:
        scale = target_median / cont_median
        normalized_flux_final = normalized_flux_final * scale

        if verbose:
            print(f"\n  Scaling to target: {scale:.4f}")
            print(
                f"  Final continuum median: {np.median(normalized_flux_final[current_mask]):.4f}"
            )

    # === ШАГ 5: Визуализация ===
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Исходный спектр с континуумом
        ax = axes[0, 0]
        ax.plot(obs_wave, obs_flux, "b-", alpha=0.7, label="Original")
        ax.plot(obs_wave, continuum_model_final, "r-", linewidth=2, label="Continuum")

        # Точки континуума
        cont_wave = obs_wave[current_mask]
        cont_flux = obs_flux[current_mask]
        ax.scatter(
            cont_wave,
            cont_flux,
            c="green",
            s=20,
            alpha=0.7,
            label=f"Continuum points (n={np.sum(current_mask)})",
        )

        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Flux")
        ax.set_title("Order with continuum")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Молекулярный шаблон
        ax = axes[0, 1]
        ax.plot(obs_wave, molecular_template_interp, "g-", linewidth=1.5)
        ax.axhline(y=0.85, color="orange", linestyle="--", alpha=0.5, label="Threshold")
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Transmission")
        ax.set_title("Molecular template")
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Нормированный спектр
        ax = axes[1, 0]
        ax.plot(obs_wave, normalized_flux_final, "b-", alpha=0.7, label="Normalized")
        ax.axhline(
            y=target_median,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Target = {target_median}",
        )
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Normalized Flux")
        ax.set_title("Normalized order")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Проверка: остатки
        ax = axes[1, 1]
        # Показываем небольшой участок
        mid = len(obs_wave) // 2
        window = min(100, len(obs_wave) // 10)
        idx_range = slice(mid - window, mid + window)

        ax.plot(
            obs_wave[idx_range],
            normalized_flux_final[idx_range],
            "b-",
            alpha=0.7,
            label="Normalized",
        )
        ax.plot(
            obs_wave[idx_range],
            molecular_template_interp[idx_range] * target_median,
            "g--",
            alpha=0.7,
            label="Template",
        )
        ax.axhline(y=target_median, color="r", linestyle="--", alpha=0.5)

        # Отмечаем точки континуума
        cont_idx = current_mask[idx_range]
        ax.scatter(
            obs_wave[idx_range][cont_idx],
            normalized_flux_final[idx_range][cont_idx],
            c="green",
            s=30,
            alpha=0.7,
            label="Continuum",
        )

        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Flux")
        ax.set_title("Zoom: molecular bands")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return normalized_flux_final, continuum_model_final, current_mask, history


def normalize_all_orders(
    obs_wave,
    obs_flux,
    order_boundaries,  # список [(wl_min, wl_max), ...]
    molecular_template_wave,
    molecular_template_flux,
    poly_degree=5,
    n_iterations=5,
    target_median=1.0,
    plot=True,
    verbose=True,
):
    """
    Нормировка всех порядков эшелле-спектра.

    Parameters:
    -----------
    order_boundaries : list of tuples - границы каждого порядка
                       [(wl_min1, wl_max1), (wl_min2, wl_max2), ...]
    """

    if verbose:
        print("\n" + "=" * 70)
        print("NORMALIZING ALL ECHELLE ORDERS")
        print(f"Number of orders: {len(order_boundaries)}")
        print("=" * 70)

    normalized_spectrum = np.zeros_like(obs_flux)
    continuum_models = []
    masks = []
    histories = []

    for i, (wl_min, wl_max) in enumerate(order_boundaries):
        if verbose:
            print(f"\n{'=' * 50}")
            print(
                f"Order {i + 1}/{len(order_boundaries)}: {wl_min:.1f} - {wl_max:.1f} Å"
            )
            print(f"{'=' * 50}")

        # Выбираем точки порядка
        mask = (obs_wave >= wl_min) & (obs_wave <= wl_max)

        if np.sum(mask) < 10:
            if verbose:
                print(f"  Skipping: too few points ({np.sum(mask)})")
            normalized_spectrum[mask] = 1.0
            continue

        wave_order = obs_wave[mask]
        flux_order = obs_flux[mask]

        # Нормируем порядок
        try:
            norm_flux, continuum, cont_mask, history = normalize_echelle_order(
                wave_order,
                flux_order,
                molecular_template_wave,
                molecular_template_flux,
                order_range=(wl_min, wl_max),
                poly_degree=poly_degree,
                n_iterations=n_iterations,
                target_median=target_median,
                plot=False,
                verbose=verbose,
            )

            normalized_spectrum[mask] = norm_flux
            continuum_models.append(continuum)
            masks.append(cont_mask)
            histories.append(history)

        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            normalized_spectrum[mask] = 1.0

    # === Визуализация всех порядков ===
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        # 1. Весь спектр
        ax = axes[0]
        ax.plot(obs_wave, obs_flux, "b-", alpha=0.5, label="Original")
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Flux")
        ax.set_title("Full spectrum")
        ax.grid(True, alpha=0.3)

        # 2. Нормированный спектр
        ax = axes[1]
        ax.plot(obs_wave, normalized_spectrum, "r-", alpha=0.7, label="Normalized")
        ax.axhline(y=target_median, color="k", linestyle="--", alpha=0.5)
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Normalized Flux")
        ax.set_title("Normalized full spectrum")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return normalized_spectrum, continuum_models, masks, histories


def detect_orders(wavelength, flux, min_order_width=50, max_order_width=500):
    """
    Автоматическое определение границ порядков по провалам между ними.
    """
    # Вычисляем градиент
    grad = np.gradient(flux)
    grad_smooth = gaussian_filter1d(np.abs(grad), sigma=5, mode="reflect")

    # Находим провалы между порядками (где градиент минимален)
    threshold = np.percentile(grad_smooth, 30)

    # Находим границы
    boundaries = []
    in_order = False
    order_start = 0

    for i in range(len(wavelength)):
        if grad_smooth[i] < threshold and not in_order:
            # Начало порядка
            order_start = wavelength[i]
            in_order = True
        elif grad_smooth[i] > threshold and in_order:
            # Конец порядка
            if wavelength[i] - order_start > min_order_width:
                boundaries.append((order_start, wavelength[i]))
            in_order = False

    # Добавляем последний порядок
    if in_order and wavelength[-1] - order_start > min_order_width:
        boundaries.append((order_start, wavelength[-1]))

    return boundaries


def normalize_with_molecular_continuum_cross_section(
    model_wave,
    model_flux,
    model_diff,
    obs_wave,
    obs_flux,
    molecular_wave,
    molecular_cross_section,
    threshold=0.1,
    poly_degree=5,
    column_density_range=(1e14, 1e20),  # диапазон column density в см⁻²
    n_iterations=3,
    sigma_clip=2.5,
    smooth_molecular=True,
    molecular_smooth_sigma=5,
    plot=True,
    verbose=True,
):
    """
    Нормировка спектра с учётом молекулярного континуума.
    ЯВНО учитывает column density (N) и cross-section (σ).

    Parameters:
    -----------
    model_wave : array - длины волн модели (реперные точки)
    model_flux : array - поток модели
    model_diff : array - маска отличий (реперные точки)
    obs_wave : array - длины волн наблюдений
    obs_flux : array - поток наблюдений
    molecular_wave : array - длины волн молекулярного шаблона
    molecular_cross_section : array - коэффициент поглощения (см²/молекулу)
    threshold : float - порог для реперных точек
    poly_degree : int - степень полинома для континуума
    column_density_range : tuple - диапазон column density (N) в см⁻²
    n_iterations : int - число итераций
    sigma_clip : float - сигма для отбраковки выбросов
    smooth_molecular : bool - сглаживать молекулярный шаблон
    molecular_smooth_sigma : float - сигма сглаживания
    plot : bool - показывать графики
    verbose : bool - печатать информацию

    Returns:
    --------
    normalized_obs : array - нормированный спектр
    continuum_model : array - модель континуума
    column_density : float - оптимальная столбовая плотность (N) в см⁻²
    optical_depth : float - максимальная оптическая глубина (τ_max = N * σ_max)
    history : dict - история итераций
    """

    if verbose:
        print("\n" + "=" * 70)
        print("NORMALIZATION WITH MOLECULAR CROSS-SECTION")
        print("(with explicit Column Density)")
        print("=" * 70)

    # === ШАГ 1: Проверка входных данных ===
    if verbose:
        print(f"\nInput shapes:")
        print(f"  model_wave: {len(model_wave)}")
        print(f"  obs_wave: {len(obs_wave)}")
        print(f"  molecular_wave: {len(molecular_wave)}")
        print(f"  molecular_cross_section: {len(molecular_cross_section)}")

    # Проверяем cross-section
    if np.max(molecular_cross_section) == 0:
        if verbose:
            print("\n⚠️  WARNING: Molecular cross-section contains only zeros!")
            print("   Proceeding without molecular correction.")

        result = normalize_without_molecular_fallback(
            model_wave,
            model_flux,
            model_diff,
            obs_wave,
            obs_flux,
            threshold,
            poly_degree,
            n_iterations,
            sigma_clip,
            plot,
            verbose,
        )
        return result[0], result[1], 0.0, 0.0, result[2]

    # === ШАГ 2: Интерполяция молекулярного шаблона ===
    molecular_cross_interp = np.interp(
        obs_wave, molecular_wave, molecular_cross_section, left=0.0, right=0.0
    )

    if np.max(molecular_cross_interp) == 0:
        if verbose:
            print(
                "\n⚠️  WARNING: No overlap between molecular and observed wavelengths!"
            )
            print(
                f"   Molecular range: {molecular_wave[0]:.2f} - {molecular_wave[-1]:.2f}"
            )
            print(f"   Observed range: {obs_wave[0]:.2f} - {obs_wave[-1]:.2f}")

        result = normalize_without_molecular_fallback(
            model_wave,
            model_flux,
            model_diff,
            obs_wave,
            obs_flux,
            threshold,
            poly_degree,
            n_iterations,
            sigma_clip,
            plot,
            verbose,
        )
        return result[0], result[1], 0.0, 0.0, result[2]

    # Сглаживание
    if smooth_molecular:
        molecular_cross_interp = gaussian_filter1d(
            molecular_cross_interp, sigma=molecular_smooth_sigma, mode="reflect"
        )

    molecular_cross_interp = np.maximum(molecular_cross_interp, 0.0)

    nonzero_mask = molecular_cross_interp > 0
    if np.sum(nonzero_mask) == 0:
        if verbose:
            print("\n⚠️  WARNING: Interpolated cross-section has no positive values!")

        result = normalize_without_molecular_fallback(
            model_wave,
            model_flux,
            model_diff,
            obs_wave,
            obs_flux,
            threshold,
            poly_degree,
            n_iterations,
            sigma_clip,
            plot,
            verbose,
        )
        return result[0], result[1], 0.0, 0.0, result[2]

    # Статистика cross-section
    sigma_max = np.max(molecular_cross_interp)
    sigma_mean = np.mean(molecular_cross_interp[nonzero_mask])

    if verbose:
        print(f"\nMolecular cross-section statistics:")
        print(f"  σ_max: {sigma_max:.4e} cm²")
        print(f"  σ_mean: {sigma_mean:.4e} cm²")
        print(
            f"  σ_min (positive): {np.min(molecular_cross_interp[nonzero_mask]):.4e} cm²"
        )

    # === ШАГ 3: Интерполяция на сетку модели ===
    obs_flux_interp = np.interp(model_wave, obs_wave, obs_flux)
    molecular_cross_model = np.interp(
        model_wave, obs_wave, molecular_cross_interp, left=0.0, right=0.0
    )
    molecular_cross_model = np.maximum(molecular_cross_model, 0.0)

    # === ШАГ 4: Начальные реперные точки ===
    initial_mask = model_diff < threshold
    if np.sum(initial_mask) < 5:
        initial_mask = model_diff < threshold * 1.5
        if verbose:
            print(f"Relaxed threshold to {threshold * 1.5:.2f}")

    ref_indices = np.where(initial_mask)[0]
    x_ref = model_wave[ref_indices]

    if verbose:
        print(f"\nInitial reference points: {len(x_ref)}")

    # === ШАГ 5: Поиск оптимальной column density (N) ===
    def transmission_from_N(N, sigma):
        """
        Вычисление пропускания из column density и cross-section.
        Transmission = exp(-N * σ)
        """
        return np.exp(-N * sigma)

    def objective_function(log10_N):
        """
        Целевая функция для поиска оптимальной column density.
        Используем log10 для лучшего масштабирования.
        """
        N = 10**log10_N

        # Вычисляем пропускание на сетке модели
        transmission = np.exp(-N * molecular_cross_model)
        transmission = np.clip(transmission, 1e-6, 1.0)

        # Учитываем только реперные точки
        transmission_ref = transmission[ref_indices]

        # Вычисляем отношения
        y_ratios = obs_flux_interp[ref_indices] / (
            model_flux[ref_indices] * transmission_ref
        )

        # Медиана должна быть близка к 1
        median_ratio = np.median(y_ratios)

        # Штраф за отклонение от 1
        return (median_ratio - 1.0) ** 2

    # Поиск оптимальной column density
    log10_N_range = (
        np.log10(column_density_range[0]),
        np.log10(column_density_range[1]),
    )

    if len(ref_indices) > 5:
        try:
            result = minimize_scalar(
                objective_function,
                bounds=log10_N_range,
                method="bounded",
                options={"xatol": 0.01},
            )
            log10_N_opt = result.x
            column_density = 10**log10_N_opt
        except Exception as e:
            if verbose:
                print(f"  Optimization failed: {e}")
                print("  Using default column density")
            column_density = 10 ** ((log10_N_range[0] + log10_N_range[1]) / 2)
    else:
        if verbose:
            print("  Too few reference points, using default column density")
        column_density = 10 ** ((log10_N_range[0] + log10_N_range[1]) / 2)

    # Вычисляем оптическую глубину
    optical_depth_max = column_density * sigma_max
    optical_depth_mean = column_density * sigma_mean

    if verbose:
        print(f"\nOptimal Column Density (N): {column_density:.4e} cm⁻²")
        print(f"  τ_max = N × σ_max = {optical_depth_max:.4f}")
        print(f"  τ_mean = N × σ_mean = {optical_depth_mean:.4f}")
        print(f"  Max molecular absorption: {1.0 - np.exp(-optical_depth_max):.2%}")
        print(f"  Mean molecular absorption: {1.0 - np.exp(-optical_depth_mean):.2%}")

    # === ШАГ 6: Итеративная коррекция ===
    current_obs_flux = obs_flux.copy()
    corrections = []
    correction_polys = []

    history = {
        "iterations": [],
        "rms_residuals": [],
        "median_spectrum": [],
        "column_density": column_density,
        "optical_depth_max": optical_depth_max,
        "flexure_amplitude": [],
    }

    for iteration in range(n_iterations):
        if verbose:
            print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")

        # Интерполяция текущего спектра
        current_obs_interp = np.interp(model_wave, obs_wave, current_obs_flux)

        # Вычисляем молекулярное пропускание с текущей column density
        transmission_model = np.exp(-column_density * molecular_cross_model)
        transmission_model = np.clip(transmission_model, 1e-6, 1.0)

        # Вычисляем отношения
        y_fit = current_obs_interp[ref_indices] / (
            model_flux[ref_indices] * transmission_model[ref_indices]
        )

        # Очистка выбросов
        median_y = np.median(y_fit)
        mad = np.median(np.abs(y_fit - median_y))
        if mad > 0:
            clip_mask = np.abs(y_fit - median_y) < sigma_clip * mad
        else:
            clip_mask = np.ones(len(y_fit), dtype=bool)

        x_fit = x_ref[clip_mask]
        y_fit = y_fit[clip_mask]

        if verbose:
            print(f"  After sigma-clipping: {len(x_fit)} points")

        # Дополнительная очистка
        for _ in range(2):
            if len(x_fit) < 10:
                break
            try:
                temp_degree = min(2, poly_degree)
                coeffs_temp = np.polyfit(x_fit, y_fit, temp_degree)
                poly_temp = np.poly1d(coeffs_temp)
                residuals = y_fit - poly_temp(x_fit)
                std_res = np.std(residuals)
                if std_res > 0:
                    mask_clean = np.abs(residuals) < sigma_clip * std_res
                    x_fit = x_fit[mask_clean]
                    y_fit = y_fit[mask_clean]
                else:
                    break
            except:
                break

        if verbose:
            print(f"  After iterative cleaning: {len(x_fit)} points")

        # Подгонка полинома
        try:
            coeffs = np.polyfit(x_fit, y_fit, poly_degree)
            poly_func = np.poly1d(coeffs)
        except np.linalg.LinAlgError:
            if verbose:
                print(f"  WARNING: Polyfit failed, reducing degree")
            poly_degree = max(3, poly_degree - 1)
            coeffs = np.polyfit(x_fit, y_fit, poly_degree)
            poly_func = np.poly1d(coeffs)

        # Строим полную модель континуума
        poly_full = poly_func(obs_wave)
        transmission_obs = np.exp(-column_density * molecular_cross_interp)
        transmission_obs = np.clip(transmission_obs, 1e-6, 1.0)
        continuum_model = poly_full * transmission_obs

        # Временная нормировка
        temp_normalized = current_obs_flux / continuum_model

        # Измеряем остаточное гнутие
        flexure_smooth = gaussian_filter1d(
            temp_normalized, sigma=100 / 2.0, mode="reflect"
        )
        flexure_amplitude = np.std(flexure_smooth - 1.0)
        if verbose:
            print(f"  Flexure amplitude: {flexure_amplitude:.6f}")

        # Коррекция
        correction = continuum_model * flexure_smooth
        corrections.append(correction)
        correction_polys.append(poly_func)

        current_obs_flux = current_obs_flux / correction

        # Сохраняем историю
        history["iterations"].append(iteration + 1)
        history["rms_residuals"].append(np.std(y_fit - poly_func(x_fit)))
        history["median_spectrum"].append(np.median(current_obs_flux))
        history["flexure_amplitude"].append(flexure_amplitude)

        # Проверка сходимости
        if iteration > 0:
            improvement = (
                history["flexure_amplitude"][-2] - history["flexure_amplitude"][-1]
            ) / (history["flexure_amplitude"][-2] + 1e-10)
            if improvement < 0.01 and flexure_amplitude < 0.001:
                if verbose:
                    print(f"  Converged after {iteration + 1} iterations")
                break

        # Обновление реперных точек
        current_diff = np.abs(
            current_obs_interp / (model_flux * transmission_model) - np.mean(model_flux)
        )

        if iteration < 2:
            new_mask = model_diff < threshold
        else:
            new_mask = (model_diff < threshold * 0.8) & (current_diff < 0.15)
            if np.sum(new_mask) < 100:
                new_mask = model_diff < threshold * 0.8

        ref_indices = np.where(new_mask)[0]
        x_ref = model_wave[ref_indices]

        if verbose:
            print(f"  Updated reference points: {len(x_ref)}")

    # === ШАГ 7: Финальная модель континуума ===
    final_poly = correction_polys[-1] if correction_polys else poly_func
    poly_full_final = final_poly(obs_wave)
    transmission_obs_final = np.exp(-column_density * molecular_cross_interp)
    transmission_obs_final = np.clip(transmission_obs_final, 1e-6, 1.0)
    final_continuum = poly_full_final * transmission_obs_final

    # === ШАГ 8: Финальная нормировка ===
    final_normalized = obs_flux / final_continuum

    # Принудительная нормализация
    median_final = np.median(final_normalized)
    if abs(median_final - np.median(model_flux)) > 0.01:
        if verbose:
            print(
                f"\nFinal median correction: {median_final:.4f} -> {np.median(model_flux)}"
            )
        final_normalized = final_normalized / median_final
        final_continuum = final_continuum * median_final

    # === ШАГ 9: Визуализация ===
    if plot:
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))

        # 1. Молекулярный шаблон
        ax = axes[0, 0]
        ax2 = ax.twinx()
        ax.plot(obs_wave, molecular_cross_interp, "b-", linewidth=1, alpha=0.7)
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("σ(λ) (cm²)", color="b")
        ax.tick_params(axis="y", labelcolor="b")
        ax2.plot(obs_wave, transmission_obs_final, "r-", linewidth=1.5, alpha=0.7)
        ax2.set_ylabel("Transmission", color="r")
        ax2.tick_params(axis="y", labelcolor="r")
        ax2.set_ylim(0, 1.05)
        ax.set_title("Cross-section and Transmission")
        ax.grid(True, alpha=0.3)

        # 2. Column density
        ax = axes[0, 1]
        ax.text(
            0.1,
            0.5,
            f"N = {column_density:.4e} cm⁻²",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.text(
            0.1,
            0.3,
            f"τ_max = {optical_depth_max:.4f}",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.text(
            0.1,
            0.1,
            f"τ_mean = {optical_depth_mean:.4f}",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.axis("off")
        ax.set_title("Column Density Parameters")

        # 3. Эволюция RMS
        ax = axes[0, 2]
        ax.plot(history["iterations"], history["rms_residuals"], "bo-", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("RMS Residuals")
        ax.set_title("Evolution of fit quality")
        ax.grid(True, alpha=0.3)

        # 4. Исходный спектр с моделью
        ax = axes[1, 0]
        mid = len(obs_wave) // 2
        window = min(800, len(obs_wave) // 3)
        idx_range = slice(mid - window, mid + window)

        ax.plot(
            obs_wave[idx_range],
            obs_flux[idx_range],
            "b-",
            linewidth=0.8,
            alpha=0.5,
            label="Original",
        )
        ax.plot(
            obs_wave[idx_range],
            final_continuum[idx_range],
            "r-",
            linewidth=1.5,
            label="Continuum model",
        )
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Flux")
        ax.set_title("Spectrum with continuum model")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Нормированный спектр
        ax = axes[1, 1]
        ax.plot(
            obs_wave[idx_range],
            final_normalized[idx_range],
            "g-",
            linewidth=0.8,
            alpha=0.7,
        )
        ax.axhline(y=1.0, color="r", linestyle="--", linewidth=2)
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Normalized Flux")
        ax.set_title("Normalized spectrum")
        ax.grid(True, alpha=0.3)

        # 6. Остаточное гнутие
        ax = axes[1, 2]
        smooth_final = gaussian_filter1d(final_normalized, sigma=50, mode="reflect")
        residual_flexure = smooth_final - 1.0
        ax.plot(obs_wave, residual_flexure, "g-", linewidth=1, alpha=0.7)
        ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
        ax.axhline(y=0.01, color="orange", linestyle=":", alpha=0.5)
        ax.axhline(y=-0.01, color="orange", linestyle=":", alpha=0.5)
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Residual flexure")
        ax.set_title(f"Residual flexure (RMS = {np.std(residual_flexure):.6f})")
        ax.grid(True, alpha=0.3)

        # 7. Гистограмма
        ax = axes[2, 0]
        hist_data = final_normalized[
            (final_normalized > 0.5) & (final_normalized < 1.5)
        ]
        ax.hist(hist_data, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(x=1.0, color="r", linestyle="--", linewidth=2)
        ax.axvline(
            x=np.median(hist_data),
            color="orange",
            linestyle="--",
            label=f"Median = {np.median(hist_data):.4f}",
        )
        ax.set_xlabel("Normalized Flux")
        ax.set_ylabel("Count")
        ax.set_title("Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 8. Зависимость χ² от N (если есть)
        ax = axes[2, 1]
        # Вычисляем χ² для разных N
        N_values = np.logspace(
            np.log10(column_density_range[0]), np.log10(column_density_range[1]), 20
        )
        chi2_values = []
        for N_test in N_values:
            transmission_test = np.exp(-N_test * molecular_cross_model)
            transmission_test = np.clip(transmission_test, 1e-6, 1.0)
            y_test = obs_flux_interp[ref_indices] / (
                model_flux[ref_indices] * transmission_test[ref_indices]
            )
            chi2 = np.std(y_test - 1.0) ** 2
            chi2_values.append(chi2)

        ax.plot(N_values, chi2_values, "b-", linewidth=1.5)
        ax.axvline(
            x=column_density,
            color="r",
            linestyle="--",
            label=f"N_opt = {column_density:.2e}",
        )
        ax.set_xscale("log")
        ax.set_xlabel("Column Density N (cm⁻²)")
        ax.set_ylabel("χ²")
        ax.set_title("χ² vs Column Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 9. Молекулярные полосы (детально)
        ax = axes[2, 2]
        # Показываем участок с сильными молекулярными полосами
        # Ищем область с максимальным cross-section
        max_idx = np.argmax(molecular_cross_interp)
        half_window = min(500, len(obs_wave) // 6)
        idx_range_mol = slice(
            max(0, max_idx - half_window), min(len(obs_wave), max_idx + half_window)
        )

        ax.plot(
            obs_wave[idx_range_mol],
            obs_flux[idx_range_mol],
            "b-",
            linewidth=0.8,
            alpha=0.5,
            label="Original",
        )
        ax.plot(
            obs_wave[idx_range_mol],
            final_continuum[idx_range_mol],
            "r-",
            linewidth=1.5,
            label="Continuum",
        )
        ax.plot(
            obs_wave[idx_range_mol],
            final_normalized[idx_range_mol],
            "g-",
            linewidth=0.8,
            alpha=0.7,
            label="Normalized",
        )
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Flux")
        ax.set_title("Molecular band region (detail)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Статистика
        print("\n" + "=" * 70)
        print("FINAL STATISTICS")
        print("=" * 70)
        print(f"Number of iterations: {len(history['iterations'])}")
        print(f"Column Density (N): {column_density:.4e} cm⁻²")
        print(f"τ_max = N × σ_max: {optical_depth_max:.4f}")
        print(f"τ_mean = N × σ_mean: {optical_depth_mean:.4f}")
        print(f"Max absorption: {1.0 - np.exp(-optical_depth_max):.2%}")
        print(f"Final median: {np.median(final_normalized):.6f}")
        print(f"Final flexure amplitude: {np.std(smooth_final - 1.0):.6f}")
        if len(history["flexure_amplitude"]) > 1:
            print(
                f"Improvement factor: {history['flexure_amplitude'][0] / history['flexure_amplitude'][-1]:.2f}x"
            )
        print("=" * 70)

    # Возвращаем 5 значений
    return final_normalized, final_continuum, column_density, optical_depth_max, history


def normalize_without_molecular_fallback(
    model_wave,
    model_flux,
    model_diff,
    obs_wave,
    obs_flux,
    threshold,
    poly_degree,
    n_iterations,
    sigma_clip,
    plot,
    verbose,
):
    """Fallback без молекулярного континуума"""
    if verbose:
        print("\n" + "=" * 70)
        print("FALLBACK: NORMALIZATION WITHOUT MOLECULAR CONTINUUM")
        print("=" * 70)

    obs_flux_interp = np.interp(model_wave, obs_wave, obs_flux)

    mask = model_diff < threshold
    if np.sum(mask) < 5:
        mask = model_diff < threshold * 1.5

    x_fit = model_wave[mask]
    y_fit = obs_flux_interp[mask] / model_flux[mask]

    median_y = np.median(y_fit)
    mad = np.median(np.abs(y_fit - median_y))
    if mad > 0:
        clip_mask = np.abs(y_fit - median_y) < sigma_clip * mad
        x_fit = x_fit[clip_mask]
        y_fit = y_fit[clip_mask]

    try:
        coeffs = np.polyfit(x_fit, y_fit, poly_degree)
        poly_func = np.poly1d(coeffs)
    except:
        poly_degree = min(3, poly_degree)
        coeffs = np.polyfit(x_fit, y_fit, poly_degree)
        poly_func = np.poly1d(coeffs)

    normalized = obs_flux / poly_func(obs_wave)

    median_norm = np.median(normalized)
    if abs(median_norm - 1.0) > 0.01:
        normalized = normalized / median_norm

    history = {
        "iterations": [1],
        "rms_residuals": [np.std(y_fit - poly_func(x_fit))],
        "median_spectrum": [np.median(normalized)],
        "flexure_amplitude": [np.std(normalized - 1.0)],
    }

    return normalized, poly_func, history


def plot_normalized_spectrum(
    obs_wave,
    obs_flux,
    normalized_flux,
    continuum_model=None,
    molecular_cross=None,
    column_density=None,
    optical_depth=None,
    title="Normalized Spectrum",
    save_path=None,
    show_plot=True,
    zoom_window=3000,
    ylim_norm=(0.7, 1.3),
    ylim_residual=(-0.05, 0.05),
):
    """
    Построение нормированного спектра с детальной информацией.

    Parameters:
    -----------
    obs_wave : array - длины волн наблюдений
    obs_flux : array - исходный поток
    normalized_flux : array - нормированный поток
    continuum_model : array - модель континуума (опционально)
    molecular_cross : array - молекулярный шаблон (опционально)
    column_density : float - столбовая плотность (опционально)
    optical_depth : float - оптическая глубина (опционально)
    title : str - заголовок
    save_path : str - путь для сохранения (опционально)
    show_plot : bool - показывать график
    zoom_window : int - ширина окна для отображения (в пикселях)
    ylim_norm : tuple - пределы для нормированного спектра
    ylim_residual : tuple - пределы для остатков
    """

    # Определяем, показывать ли весь спектр или его часть
    if len(obs_wave) > 5000:
        mid = len(obs_wave) // 2
        window = min(zoom_window, len(obs_wave) // 3)
        idx_range = slice(mid - window, mid + window)
        wave_plot = obs_wave[idx_range]
        flux_plot = obs_flux[idx_range]
        norm_plot = normalized_flux[idx_range]
        if continuum_model is not None:
            cont_plot = continuum_model[idx_range]
        else:
            cont_plot = None
    else:
        wave_plot = obs_wave
        flux_plot = obs_flux
        norm_plot = normalized_flux
        cont_plot = continuum_model

    # Создаём фигуру
    if continuum_model is not None:
        fig, axes = plt.subplots(
            3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [2, 1, 1]}
        )
    else:
        fig, axes = plt.subplots(
            2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]}
        )

    # === ВЕРХНИЙ ГРАФИК: Исходный и нормированный спектр ===
    ax1 = axes[0]

    # Исходный спектр
    ax1.plot(wave_plot, flux_plot, "b-", linewidth=1, alpha=0.6, label="Original")

    # Модель континуума (если есть)
    if cont_plot is not None:
        ax1.plot(
            wave_plot, cont_plot, "r-", linewidth=2, alpha=0.8, label="Continuum model"
        )

    # Нормированный спектр (смещён вверх для наглядности)
    offset = 0.2 * np.max(flux_plot) if np.max(flux_plot) > 0 else 0.5
    ax1.plot(
        wave_plot,
        norm_plot + offset,
        "g-",
        linewidth=1.2,
        alpha=0.8,
        label="Normalized (+offset)",
    )

    # Линия континуума для нормированного спектра
    ax1.axhline(y=1.0 + offset, color="g", linestyle="--", alpha=0.5, linewidth=1)

    ax1.set_xlabel("Wavelength (Å)", fontsize=12)
    ax1.set_ylabel("Flux", fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Добавляем информацию о параметрах
    info_text = []
    if column_density is not None:
        info_text.append(f"N = {column_density:.3e} cm⁻²")
    if optical_depth is not None:
        info_text.append(f"τ_max = {optical_depth:.4f}")
    if info_text:
        ax1.text(
            0.02,
            0.98,
            "\n".join(info_text),
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # === СРЕДНИЙ ГРАФИК: Нормированный спектр (увеличенный) ===
    ax2 = axes[1]

    ax2.plot(wave_plot, norm_plot, "g-", linewidth=1.2, alpha=0.8)
    ax2.axhline(y=1.0, color="r", linestyle="--", linewidth=2, label="Continuum (1.0)")

    # Добавляем области ±3% для оценки качества
    ax2.axhline(y=1.03, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
    ax2.axhline(y=0.97, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
    ax2.fill_between(wave_plot, 0.97, 1.03, color="gray", alpha=0.1)

    ax2.set_xlabel("Wavelength (Å)", fontsize=12)
    ax2.set_ylabel("Normalized Flux", fontsize=12)
    ax2.set_title("Normalized Spectrum (zoom)", fontsize=12)
    ax2.set_ylim(ylim_norm)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Добавляем статистику (по всему спектру, не только по отображаемой части)
    median_norm = np.median(normalized_flux)
    std_norm = np.std(normalized_flux)
    rms_norm = np.sqrt(np.mean((normalized_flux - 1.0) ** 2))
    ax2.text(
        0.02,
        0.98,
        f"Median = {median_norm:.4f}\nStd = {std_norm:.4f}\nRMS = {rms_norm:.4f}",
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
    )

    # === ТРЕТИЙ ГРАФИК: Остатки (только если есть континуум) ===
    if continuum_model is not None and len(axes) > 2:
        ax3 = axes[2]

        # Вычисляем сглаженные остатки (гнутие) для ВСЕГО спектра
        smooth_flux_full = gaussian_filter1d(normalized_flux, sigma=100, mode="reflect")
        residuals_full = smooth_flux_full - 1.0

        # БЕРЁМ ТОЛЬКО ТУ ЖЕ ОБЛАСТЬ, ЧТО И ДЛЯ СПЕКТРА
        residuals_plot = residuals_full[idx_range]

        ax3.plot(wave_plot, residuals_plot, "b-", linewidth=1, alpha=0.7)
        ax3.axhline(y=0, color="r", linestyle="--", linewidth=1.5)
        ax3.axhline(y=0.01, color="orange", linestyle=":", alpha=0.5)
        ax3.axhline(y=-0.01, color="orange", linestyle=":", alpha=0.5)

        ax3.set_xlabel("Wavelength (Å)", fontsize=12)
        ax3.set_ylabel("Residual flexure", fontsize=12)
        ax3.set_title(
            f"Residual flexure (RMS = {np.std(residuals_full):.6f})", fontsize=12
        )
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(ylim_residual)

        # Информация о качестве
        ax3.text(
            0.02,
            0.98,
            f"RMS flexure = {np.std(residuals_full):.6f}\nMax = {np.max(np.abs(residuals_full)):.6f}",
            transform=ax3.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        )

    plt.tight_layout()

    # Сохраняем если нужно
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig


def plot_full_spectrum_comparison(
    obs_wave,
    obs_flux,
    normalized_flux,
    continuum_model=None,
    molecular_cross=None,
    column_density=None,
    optical_depth=None,
    save_path=None,
    show_plot=True,
):
    """
    Построение полного спектра с несколькими панелями для детального анализа.
    """

    fig = plt.figure(figsize=(16, 14))

    # === Панель 1: Полный спектр ===
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(obs_wave, obs_flux, "b-", linewidth=0.8, alpha=0.6, label="Original")
    if continuum_model is not None:
        ax1.plot(
            obs_wave, continuum_model, "r-", linewidth=1.5, alpha=0.8, label="Continuum"
        )
    ax1.set_xlabel("Wavelength (Å)")
    ax1.set_ylabel("Flux")
    ax1.set_title("Full Spectrum with Continuum")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # === Панель 2: Нормированный спектр (полный) ===
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(obs_wave, normalized_flux, "g-", linewidth=0.8, alpha=0.7)
    ax2.axhline(y=1.0, color="r", linestyle="--", linewidth=2)
    ax2.set_xlabel("Wavelength (Å)")
    ax2.set_ylabel("Normalized Flux")
    ax2.set_title("Normalized Spectrum")
    ax2.set_ylim(0.5, 1.5)
    ax2.grid(True, alpha=0.3)

    # === Панель 3: Участок с молекулярными полосами ===
    ax3 = plt.subplot(3, 2, 3)
    # Находим область с максимальным молекулярным поглощением
    if molecular_cross is not None:
        # Если передан молекулярный шаблон для всего спектра
        max_idx = np.argmax(molecular_cross)
        half_window = min(500, len(obs_wave) // 6)
        idx_range = slice(
            max(0, max_idx - half_window), min(len(obs_wave), max_idx + half_window)
        )

        ax3.plot(
            obs_wave[idx_range],
            obs_flux[idx_range],
            "b-",
            linewidth=0.8,
            alpha=0.5,
            label="Original",
        )
        if continuum_model is not None:
            ax3.plot(
                obs_wave[idx_range],
                continuum_model[idx_range],
                "r-",
                linewidth=1.5,
                label="Continuum",
            )
        ax3.set_xlabel("Wavelength (Å)")
        ax3.set_ylabel("Flux")
        ax3.set_title("Molecular Band Region")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # === Панель 4: Нормированный участок с молекулярными полосами ===
        ax4 = plt.subplot(3, 2, 4)
        ax4.plot(
            obs_wave[idx_range],
            normalized_flux[idx_range],
            "g-",
            linewidth=0.8,
            alpha=0.7,
        )
        ax4.axhline(y=1.0, color="r", linestyle="--", linewidth=2)
        ax4.set_xlabel("Wavelength (Å)")
        ax4.set_ylabel("Normalized Flux")
        ax4.set_title("Normalized - Molecular Region")
        ax4.set_ylim(0.5, 1.5)
        ax4.grid(True, alpha=0.3)
    else:
        # Если нет молекулярного шаблона, показываем центральную область
        mid = len(obs_wave) // 2
        window = min(1000, len(obs_wave) // 4)
        idx_range = slice(mid - window, mid + window)

        ax3.plot(
            obs_wave[idx_range],
            obs_flux[idx_range],
            "b-",
            linewidth=0.8,
            alpha=0.5,
            label="Original",
        )
        if continuum_model is not None:
            ax3.plot(
                obs_wave[idx_range],
                continuum_model[idx_range],
                "r-",
                linewidth=1.5,
                label="Continuum",
            )
        ax3.set_xlabel("Wavelength (Å)")
        ax3.set_ylabel("Flux")
        ax3.set_title("Central Region")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4 = plt.subplot(3, 2, 4)
        ax4.plot(
            obs_wave[idx_range],
            normalized_flux[idx_range],
            "g-",
            linewidth=0.8,
            alpha=0.7,
        )
        ax4.axhline(y=1.0, color="r", linestyle="--", linewidth=2)
        ax4.set_xlabel("Wavelength (Å)")
        ax4.set_ylabel("Normalized Flux")
        ax4.set_title("Normalized - Central Region")
        ax4.set_ylim(0.5, 1.5)
        ax4.grid(True, alpha=0.3)

    # === Панель 5: Остатки (гнутие) ===
    ax5 = plt.subplot(3, 2, 5)
    smooth_flux = gaussian_filter1d(normalized_flux, sigma=100, mode="reflect")
    residuals = smooth_flux - 1.0
    ax5.plot(obs_wave, residuals, "b-", linewidth=0.8, alpha=0.7)
    ax5.axhline(y=0, color="r", linestyle="--", linewidth=1.5)
    ax5.axhline(y=0.01, color="orange", linestyle=":", alpha=0.5)
    ax5.axhline(y=-0.01, color="orange", linestyle=":", alpha=0.5)
    ax5.set_xlabel("Wavelength (Å)")
    ax5.set_ylabel("Residual flexure")
    ax5.set_title(f"Residual Flexure (RMS = {np.std(residuals):.6f})")
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(-0.05, 0.05)

    # === Панель 6: Гистограмма и статистика ===
    ax6 = plt.subplot(3, 2, 6)
    hist_data = normalized_flux[(normalized_flux > 0.5) & (normalized_flux < 1.5)]
    ax6.hist(hist_data, bins=50, edgecolor="black", alpha=0.7, color="green")
    ax6.axvline(x=1.0, color="r", linestyle="--", linewidth=2, label="Continuum")
    ax6.axvline(
        x=np.median(hist_data),
        color="orange",
        linestyle="--",
        label=f"Median = {np.median(hist_data):.4f}",
    )
    ax6.set_xlabel("Normalized Flux")
    ax6.set_ylabel("Count")
    ax6.set_title("Distribution")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Добавляем общую информацию
    info_text = f"""
    Statistics:
    Median: {np.median(normalized_flux):.4f}
    Mean: {np.mean(normalized_flux):.4f}
    Std: {np.std(normalized_flux):.4f}
    RMS: {np.sqrt(np.mean((normalized_flux - 1.0) ** 2)):.4f}
    """
    if column_density is not None:
        info_text += f"\nN = {column_density:.3e} cm⁻²"
    if optical_depth is not None:
        info_text += f"\nτ_max = {optical_depth:.4f}"

    plt.figtext(
        0.02,
        0.02,
        info_text,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig
