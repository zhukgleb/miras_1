import numpy as np
from numpy.polynomial import Polynomial


def get_spectra_cut(start_wl, end_wl, data):
    delta = data[:, 0][1] - data[:, 0][0]

    cut_data = []
    start_idx = np.argmin(np.abs(data[:, 0] - start_wl))
    end_idx = np.argmin(np.abs(data[:, 0] - end_wl))

    cut_data.append(data[int(start_idx) : int(end_idx)])
    return cut_data[0]


def median_normalization(data):
    wavelengths = data[:, 0]
    flux = data[:, 1].copy() 
    
    zero_mask = (flux == 0)
    zero_indices = np.where(zero_mask)[0]
    
    if len(zero_indices) == 0:
        mask = (flux != 0)
        if np.any(mask):
            median_val = np.median(flux[mask])
            if median_val != 0:
                flux[mask] /= median_val
        return np.column_stack((wavelengths, flux))
    

    start_idx = 0
    for i, zero_idx in enumerate(zero_indices):
        order_slice = flux[start_idx:zero_idx]
        
        order_mask = (order_slice != 0)
        if np.any(order_mask):
            median_val = np.median(order_slice[order_mask])
            if median_val != 0:
                flux[start_idx:zero_idx][order_mask] /= median_val
        
        start_idx = zero_idx + 1
    
    if start_idx < len(flux):
        order_slice = flux[start_idx:]
        order_mask = (order_slice != 0)
        if np.any(order_mask):
            median_val = np.median(order_slice[order_mask])
            if median_val != 0:
                flux[start_idx:][order_mask] /= median_val

    return np.column_stack((wavelengths, flux))


def normalize_orders_median(data):
    wavelengths = data[:, 0]
    flux = data[:, 1].copy()
    
    zero_mask = (flux == 0)
    zero_indices = np.where(zero_mask)[0]
    
    if len(zero_indices) == 0:
        mask = (flux != 0)
        if np.any(mask):
            median_val = np.median(flux[mask])
            if median_val != 0:
                flux[mask] /= median_val
        return np.column_stack((wavelengths, flux))
    
    start_idx = 0
    for zero_idx in zero_indices:
        order_slice = flux[start_idx:zero_idx]
        order_mask = (order_slice != 0)
        if np.any(order_mask):
            median_val = np.median(order_slice[order_mask])
            if median_val != 0:
                flux[start_idx:zero_idx][order_mask] /= median_val
        start_idx = zero_idx + 1
    
    if start_idx < len(flux):
        order_slice = flux[start_idx:]
        order_mask = (order_slice != 0)
        if np.any(order_mask):
            median_val = np.median(order_slice[order_mask])
            if median_val != 0:
                flux[start_idx:][order_mask] /= median_val
    
    return np.column_stack((wavelengths, flux))


def subtract_parabolic_continuum(data, degree=2, mask_zero=True):
    """
    Вычитание параболического (полиномиального) инструментального континуума.
    
    Параметры:
        data : np.ndarray, shape (N, 2)
            Первый столбец — длина волны, второй — поток (ADU).
            Порядки разделены нулевыми значениями потока.
        degree : int, default=2
            Степень полинома для аппроксимации континуума (2 — парабола).
        mask_zero : bool, default=True
            Исключать ли нулевые точки из аппроксимации.
    
    Возвращает:
        np.ndarray, shape (N, 2)
            Массив с вычтенным континуумом (поток = поток - континуум).
    """
    wavelengths = data[:, 0].copy()
    flux = data[:, 1].copy()
    corrected_flux = np.zeros_like(flux)
    
    # Находим границы порядков
    zero_mask = (flux == 0)
    zero_indices = np.where(zero_mask)[0]
    
    # Функция для обработки одного порядка
    def process_order(wave_ord, flux_ord, start_idx, end_idx):
        # Создаем маску для ненулевых точек
        if mask_zero:
            valid_mask = (flux_ord != 0)
        else:
            valid_mask = np.ones_like(flux_ord, dtype=bool)
        
        # Если слишком мало точек для аппроксимации — пропускаем
        if np.sum(valid_mask) < degree + 1:
            corrected_flux[start_idx:end_idx] = flux_ord
            return
        
        # Выбираем валидные точки
        wave_valid = wave_ord[valid_mask]
        flux_valid = flux_ord[valid_mask]
        
        try:
            # Аппроксимируем полиномом степени degree
            coeffs = np.polyfit(wave_valid, flux_valid, deg=degree)
            poly = np.poly1d(coeffs)
            
            # Вычисляем континуум для всех точек порядка
            continuum = poly(wave_ord)
            
            corrected_flux[start_idx:end_idx] = flux_ord / continuum
            
        except np.linalg.LinAlgError:
            print("linalg error")
            # Если аппроксимация не удалась (например, вырожденная матрица)
            corrected_flux[start_idx:end_idx] = flux_ord
    
    # Обрабатываем каждый порядок
    if len(zero_indices) == 0:
        # Весь массив — один порядок
        process_order(wavelengths, flux, 0, len(flux))
    else:
        start_idx = 0
        for zero_idx in zero_indices:
            # Порядок от start_idx до zero_idx (не включая zero_idx)
            wave_ord = wavelengths[start_idx:zero_idx]
            flux_ord = flux[start_idx:zero_idx]
            process_order(wave_ord, flux_ord, start_idx, zero_idx)
            
            # Нулевой разделитель остается нулем
            corrected_flux[zero_idx] = 0
            start_idx = zero_idx + 1
        
        # Последний порядок
        if start_idx < len(flux):
            wave_ord = wavelengths[start_idx:]
            flux_ord = flux[start_idx:]
            process_order(wave_ord, flux_ord, start_idx, len(flux))
    
    return np.column_stack((wavelengths, corrected_flux))


def full_pipeline(data, degree=2, do_median=True, do_continuum=True):
    """
    Полный пайплайн обработки:
    1. Медианная нормализация (опционально)
    2. Вычитание параболического континуума (опционально)
    
    Параметры:
        data : np.ndarray, shape (N, 2)
            Исходные данные.
        degree : int, default=2
            Степень полинома для континуума.
        do_median : bool, default=True
            Применять ли медианную нормализацию.
        do_continuum : bool, default=True
            Вычитать ли параболический континуум.
    
    Возвращает:
        np.ndarray, shape (N, 2)
            Обработанные данные.
    """
    result = data.copy()
    
    if do_median:
        result = normalize_orders_median(result)
    
    if do_continuum:
        result = subtract_parabolic_continuum(result, degree=degree)
    
    return result
