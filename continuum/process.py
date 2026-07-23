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


def diagnose_normalization(obs_wave, obs_flux, normalized_obs, 
                           x_fit, y_fit, poly_func):
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
    poly_mid = poly_func(obs_wave[len(obs_wave)//2])
    print(f"Polynomial values: start={poly_start:.4f}, mid={poly_mid:.4f}, end={poly_end:.4f}")
    print(f"Dynamic range of polynomial: {poly_end/poly_start:.4f}")
    
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
        'median': median_norm,
        'median_ratio': median_ratio,
        'mean_residual': mean_residual,
        'rms_residual': np.std(y_fit - poly_at_ref)
    }

def normalize_with_poly(model_wave, model_flux, model_diff, 
                        obs_wave, obs_flux, 
                        threshold=0.1, poly_degree=3, 
                        sigma_clip=3.0, plot=True, 
                        force_normalize=True):
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
    
    print("\n" + "="*60)
    print("NORMALIZATION WITH POLYNOMIAL")
    print("="*60)
    
    # ============================================================
    # 1. Интерполяция наблюдений на сетку модели
    # ============================================================
    obs_flux_interp = np.interp(model_wave, obs_wave, obs_flux)
    
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
        raise ValueError(f"Too few reference points ({n_points}) for polynomial fitting!")
    
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
        print(f"  WARNING: Polynomial degree {poly_degree} failed, trying lower degree...")
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
        plt.plot(x_fit, y_fit, 'bo', markersize=3, alpha=0.7, label='Reference points')
        
        # Гладкая кривая полинома
        x_smooth = np.linspace(min(x_fit), max(x_fit), 500)
        plt.plot(x_smooth, poly_func(x_smooth), 'r-', linewidth=2, 
                 label=f'Polynomial (deg={poly_degree})')
        
        plt.xlabel('Wavelength')
        plt.ylabel('Observed / Model')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title('Polynomial fit to reference points')
        
        # График 2: Остатки
        plt.subplot(2, 2, 2)
        plt.plot(x_fit, residuals, 'go', markersize=3, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
        plt.axhline(y=3*rms_res, color='orange', linestyle=':', alpha=0.7)
        plt.axhline(y=-3*rms_res, color='orange', linestyle=':', alpha=0.7)
        plt.xlabel('Wavelength')
        plt.ylabel('Residuals')
        plt.grid(True, alpha=0.3)
        plt.title(f'Residuals (RMS = {rms_res:.4f})')
        
        # График 3: Гистограмма остатков
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Residual')
        plt.ylabel('Count')
        plt.title(f'Mean residual = {mean_res:.6f}')
        plt.grid(True, alpha=0.3)
        
        # График 4: Весь спектр (маленький участок для проверки)
        plt.subplot(2, 2, 4)
        # Показываем небольшой участок в центре спектра
        mid_idx = len(obs_wave) // 2
        half_range = min(500, len(obs_wave) // 4)
        plot_range = slice(mid_idx - half_range, mid_idx + half_range)
        
        plt.plot(obs_wave[plot_range], obs_flux[plot_range], 
                 'b-', alpha=0.5, label='Original')
        plt.plot(obs_wave[plot_range], obs_flux[plot_range] / poly_func(obs_wave)[plot_range], 
                 'r-', alpha=0.7, label='Normalized')
        plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
        plt.xlabel('Wavelength')
        plt.ylabel('Flux')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title('Spectrum before and after normalization (center)')
        
        plt.tight_layout()
    
    # 7. Применяем полином ко всему спектру
    poly_full = poly_func(obs_wave)
    normalized_obs = obs_flux / poly_full
    
    # 8. Принудительная нормализация (если нужно)
    if force_normalize:
        median_norm = np.median(normalized_obs)
        if abs(median_norm - 1.0) > 0.02:
            print(f"  Forcing normalization: median {median_norm:.4f} -> 1.0")
            correction = 1.0 / median_norm
            normalized_obs = normalized_obs * correction
            # Корректируем полином (для согласованности)
            poly_func = lambda x: poly_func(x) * correction
    
    # 9. Финальная диагностика
    final_median = np.median(normalized_obs)
    print(f"\nFinal median of normalized spectrum: {final_median:.4f}")
    
    if abs(final_median - 1.0) > 0.05:
        print("  ⚠️  WARNING: Normalized spectrum deviates from 1.0")
        print("     Possible issues:")
        print("     - Outliers in reference points not fully removed")
        print("     - Model not properly normalized")
        print("     - Polynomial degree too high (overfitting)")
    else:
        print("  ✅ Normalization successful!")
    
    print("="*60 + "\n")
    
    return normalized_obs, poly_func, obs_flux_interp