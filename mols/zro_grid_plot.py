
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from process 
import os


def load_and_process_spectrum(filepath):
    """
    Загружает спектр из файла, конвертирует волновые числа в ангстремы
    и сортирует по возрастанию длины волны.
    """
    # Загружаем данные
    data = np.loadtxt(filepath)
    
    # Первый столбец - волновые числа (см^-1), второй - кросс-секция
    wavenumber = data[:, 0]  # см^-1
    cross_section = data[:, 1]
    
    # Конвертируем волновые числа в ангстремы
    # 1 см^-1 = 10^8 / wavenumber ангстрем
    # lambda(Å) = 10^8 / wavenumber(cm^-1)
    wavelength_ang = 1e8 / wavenumber
    
    # Сортируем по возрастанию длины волны
    sort_idx = np.argsort(wavelength_ang)
    wavelength_sorted = wavelength_ang[sort_idx]
    cross_section_sorted = cross_section[sort_idx]
    
    return wavelength_sorted, cross_section_sorted

def main():
    # Путь к папке с файлами
    folder_path = "zro_grid"  # измените на ваш путь
    
    # Получаем список всех .xsec файлов
    file_pattern = "ZrO_*.xsec"
    files = list(Path(folder_path).glob(file_pattern))
    
    if not files:
        print(f"Файлы не найдены в папке {folder_path}")
        return
    
    # Словарь для хранения всех спектров
    # Ключ: температура (int), значение: dict с wavelength и normalized_spectrum
    spectra_dict = {}
    
    # Параметры для нормализации
    column_density = 1e16
    temp_for_normalization = 1500  # температура для функции normalize_molecular_spectrum
    
    print(f"Найдено файлов: {len(files)}")
    print("Обработка файлов...")
    
    # Создаем фигуру для графика
    plt.figure(figsize=(12, 8))
    
    # Обрабатываем каждый файл
    for filepath in sorted(files):
        # Извлекаем температуру из имени файла
        # Формат: ZrO_1090.xsec
        filename = filepath.stem  # без расширения
        temp_str = filename.split('_')[1]  # берем часть после подчеркивания
        temperature = int(temp_str)
        
        print(f"Обработка: {filepath.name}, T = {temperature} K")
        
        # Загружаем и обрабатываем спектр
        wavelength, cross_section = load_and_process_spectrum(filepath)
        
        # Нормализуем спектр
        normalized_spectrum = normalize_molecular_spectrum(
            wavelength, 
            cross_section, 
            temp=temp_for_normalization, 
            column_density=column_density
        )
        
        # Сохраняем в словарь
        spectra_dict[temperature] = {
            'wavelength': wavelength,
            'cross_section': cross_section,
            'normalized': normalized_spectrum
        }
        
        # Строим график для каждого спектра
        plt.plot(wavelength, normalized_spectrum, 
                label=f'T = {temperature} K', 
                linewidth=0.8, alpha=0.7)
    
    # Настройка графика
    plt.xlabel('Длина волны (Å)', fontsize=12)
    plt.ylabel('Нормализованный спектр', fontsize=12)
    plt.title('Сетка моделей ZrO при разных температурах', fontsize=14)
    plt.legend(loc='best', fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.xlim(wavelength.min(), wavelength.max())  # ограничиваем по диапазону
    
    # Сохраняем график
    plt.tight_layout()
    plt.savefig('zro_spectral_grid.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Выводим информацию о сохраненных спектрах
    print("\nСпектры сохранены в словаре spectra_dict")
    print(f"Доступные температуры: {sorted(spectra_dict.keys())}")
    
    # Пример доступа к данным для конкретной температуры
    if 1090 in spectra_dict:
        print(f"\nПример данных для T=1090 K:")
        print(f"  Длина волны: {spectra_dict[1090]['wavelength'][:5]} Å")
        print(f"  Нормализованный спектр: {spectra_dict[1090]['normalized'][:5]}")
    
    return spectra_dict

if __name__ == "__main__":
    spectra_dict = main()
