import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

class ModelGridExtractor:
    """
    Класс для извлечения и организации сетки модельных спектров.
    """
    
    def __init__(self, base_path: str):
        """
        Инициализация экстрактора.
        
        Parameters:
        -----------
        base_path : str
            Путь к папке с модельными спектрами и CSV-файлом
        """
        self.base_path = Path(base_path)
        self.spectra_data = {}
        self.parameters_df = None
        self.grid_structure = {}
        
    def load_parameters(self, csv_filename: str = "spectra_parameters.csv") -> pd.DataFrame:
        """
        Загрузка параметров из CSV-файла.
        
        Parameters:
        -----------
        csv_filename : str
            Имя CSV-файла с параметрами
            
        Returns:
        --------
        pd.DataFrame
            DataFrame с параметрами моделей
        """
        csv_path = self.base_path / csv_filename
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV файл не найден: {csv_path}")
        
        # Загрузка CSV с обработкой возможных пустых строк в конце
        self.parameters_df = pd.read_csv(csv_path, skipfooter=1, engine='python')
        
        # Очистка от возможных пустых строк
        self.parameters_df = self.parameters_df.dropna(how='all')
        
        # Приведение specname к строковому типу
        self.parameters_df['specname'] = self.parameters_df['specname'].astype(str)
        
        print(f"Загружено {len(self.parameters_df)} записей параметров")
        print(f"Доступные параметры: {list(self.parameters_df.columns)}")
        
        return self.parameters_df
    
    def load_spectra(self, pattern: str = "*.spec") -> Dict[str, np.ndarray]:
        """
        Загрузка всех спектров из папки.
        
        Parameters:
        -----------
        pattern : str
            Шаблон для поиска файлов спектров
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Словарь {имя_файла: массив_спектра}
        """
        spectrum_files = list(self.base_path.glob(pattern))
        
        if not spectrum_files:
            raise FileNotFoundError(f"Файлы спектров не найдены по шаблону {pattern}")
        
        for spec_file in spectrum_files:
            try:
                # Загрузка спектра (3 столбца: длина волны, норм. поток, ненорм. поток)
                data = np.loadtxt(spec_file)
                
                # Проверка размерности
                if data.ndim == 1:
                    # Если файл с одномерными данными, предполагаем что это только поток
                    # В реальности лучше уточнить формат
                    print(f"Внимание: файл {spec_file.name} имеет одномерные данные")
                    wavelength = np.arange(len(data))
                    flux_norm = data
                    flux_unorm = data
                elif data.shape[1] >= 3:
                    wavelength = data[:, 0]
                    flux_norm = data[:, 1]
                    flux_unorm = data[:, 2]
                else:
                    print(f"Внимание: файл {spec_file.name} имеет нестандартный формат ({data.shape})")
                    continue
                
                self.spectra_data[spec_file.name] = {
                    'wavelength': wavelength,
                    'flux_norm': flux_norm,
                    'flux_unorm': flux_unorm,
                    'full_data': data
                }
                
            except Exception as e:
                print(f"Ошибка при загрузке {spec_file.name}: {e}")
        
        print(f"Загружено {len(self.spectra_data)} спектров")
        return self.spectra_data
    
    def build_grid(self) -> Dict[str, Dict]:
        """
        Построение сетки моделей, объединяя параметры и спектры.
        
        Returns:
        --------
        Dict[str, Dict]
            Структурированная сетка моделей
        """
        if self.parameters_df is None:
            self.load_parameters()
        
        if not self.spectra_data:
            self.load_spectra()
        
        # Объединение параметров и спектров
        for idx, row in self.parameters_df.iterrows():
            spec_name = row['specname']
            
            if spec_name in self.spectra_data:
                # Создаем запись для модели
                model_entry = {
                    'parameters': row.to_dict(),
                    'spectrum': self.spectra_data[spec_name]
                }
                
                # Добавляем в общую структуру
                self.grid_structure[spec_name] = model_entry
            else:
                print(f"Внимание: для {spec_name} не найден спектр")
        
        # Создание многомерной сетки по параметрам
        param_grid = self._create_parameter_grid()
        
        self.grid_structure['param_grid'] = param_grid
        
        print(f"Построена сетка из {len(self.grid_structure)-1} моделей")
        return self.grid_structure
    
    def _create_parameter_grid(self) -> Dict:
        """
        Создание многомерной сетки параметров.
        
        Returns:
        --------
        Dict
            Словарь с уникальными значениями параметров
        """
        if self.parameters_df is None:
            return {}
        
        param_grid = {}
        
        # Исключаем specname и rv из параметров сетки
        param_columns = [col for col in self.parameters_df.columns 
                        if col not in ['specname', 'rv']]
        
        for param in param_columns:
            unique_values = sorted(self.parameters_df[param].unique())
            param_grid[param] = unique_values
        
        return param_grid
    
    def get_model(self, spec_name: str) -> Optional[Dict]:
        """
        Получение конкретной модели по имени файла.
        
        Parameters:
        -----------
        spec_name : str
            Имя файла спектра
            
        Returns:
        --------
        Optional[Dict]
            Данные модели или None
        """
        return self.grid_structure.get(spec_name)
    
    def filter_models(self, **kwargs) -> List[str]:
        """
        Фильтрация моделей по параметрам.
        
        Parameters:
        -----------
        **kwargs : dict
            Параметры для фильтрации (например, teff=5000, logg=1.0)
            
        Returns:
        --------
        List[str]
            Список имен моделей, удовлетворяющих критериям
        """
        if self.parameters_df is None:
            self.load_parameters()
        
        mask = pd.Series([True] * len(self.parameters_df))
        
        for key, value in kwargs.items():
            if key in self.parameters_df.columns:
                mask &= (self.parameters_df[key] == value)
        
        filtered = self.parameters_df[mask]
        return filtered['specname'].tolist()
    
    def get_spectra_by_params(self, **kwargs) -> Dict[str, Dict]:
        """
        Получение спектров, отфильтрованных по параметрам.
        
        Parameters:
        -----------
        **kwargs : dict
            Параметры для фильтрации
            
        Returns:
        --------
        Dict[str, Dict]
            Словарь с отфильтрованными моделями
        """
        model_names = self.filter_models(**kwargs)
        
        result = {}
        for name in model_names:
            if name in self.grid_structure:
                result[name] = self.grid_structure[name]
        
        return result
    
    def get_parameter_grid_summary(self) -> pd.DataFrame:
        """
        Получение сводки по параметрам сетки.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame со статистикой по параметрам
        """
        if self.parameters_df is None:
            self.load_parameters()
        
        summary = {}
        param_columns = [col for col in self.parameters_df.columns 
                        if col not in ['specname', 'rv']]
        
        for param in param_columns:
            summary[param] = {
                'min': self.parameters_df[param].min(),
                'max': self.parameters_df[param].max(),
                'unique_count': self.parameters_df[param].nunique(),
                'values': sorted(self.parameters_df[param].unique())
            }
        
        return pd.DataFrame(summary).T
    
    def export_grid_structure(self, output_file: str = "grid_structure.txt"):
        """
        Экспорт структуры сетки в текстовый файл.
        
        Parameters:
        -----------
        output_file : str
            Имя выходного файла
        """
        if not self.grid_structure:
            self.build_grid()
        
        with open(self.base_path / output_file, 'w') as f:
            f.write("=== СТРУКТУРА СЕТКИ МОДЕЛЕЙ ===\n\n")
            
            # Параметры сетки
            if 'param_grid' in self.grid_structure:
                f.write("Параметры сетки:\n")
                for param, values in self.grid_structure['param_grid'].items():
                    f.write(f"  {param}: {values}\n")
                f.write("\n")
            
            # Список моделей
            f.write("Модели в сетке:\n")
            for spec_name in sorted(self.grid_structure.keys()):
                if spec_name != 'param_grid':
                    model = self.grid_structure[spec_name]
                    params = model['parameters']
                    param_str = ', '.join([f"{k}={v}" for k, v in params.items() 
                                         if k not in ['specname']])
                    f.write(f"  {spec_name}: {param_str}\n")
            
            f.write(f"\nВсего моделей: {len(self.grid_structure)-1}\n")


# Пример использования
if __name__ == "__main__":
    # Путь к папке с данными
    folder_path = "2026-07-20-13-28-24_0.7248514425106289_LTE_synthetic_spectra_parameters"
    
    # Создание экстрактора
    extractor = ModelGridExtractor(folder_path)
    
    # Загрузка параметров
    params_df = extractor.load_parameters()    
    # Загрузка спектров
    spectra = extractor.load_spectra()
    
    # Построение сетки
    grid = extractor.build_grid()
    
    # # Просмотр структуры сетки
    # print("\nСтруктура параметров сетки:")
    # param_grid = grid.get('param_grid', {})
    # for param, values in param_grid.items():
    #     print(f"  {param}: {values}")
    
    # # Фильтрация моделей
    # print("\nМодели с Teff=5000, logg=1.0:")
    # models = extractor.filter_models(teff=5000, logg=1.0)
    # for model in models:
    #     print(f"  {model}")
    
    # # Получение спектра для конкретной модели
    # print("\nДанные модели 0.spec:")
    # model_data = extractor.get_model("0.spec")
    # if model_data:
    #     print(f"  Параметры: {model_data['parameters']}")
    #     print(f"  Длина спектра: {len(model_data['spectrum']['wavelength'])} точек")
    #     print(f"  Диапазон длин волн: {model_data['spectrum']['wavelength'][0]:.2f} - {model_data['spectrum']['wavelength'][-1]:.2f}")
    
    # # Экспорт структуры сетки
    # extractor.export_grid_structure()
    
    # # Сводка по параметрам
    # print("\nСводка по параметрам сетки:")
    # summary = extractor.get_parameter_grid_summary()
    # print(summary)

    for key in grid.keys():
        if key == "param_grid":
            pass
        else:
            wl = grid[key]['spectrum']['wavelength']
            flux = grid[key]['spectrum']['flux_norm']
            params = grid[key]['parameters']
            
    rep_flux = grid['0.spec']['spectrum']['flux_norm']
    flux_arr = []


    for key in grid.keys():
        if key == "param_grid":
            pass
        else:
            flux_arr.append(grid[key]['spectrum']['flux_norm']) 

    delta_arr = np.array([abs(rep_flux - flux_arr[i]) for i in range(len(flux_arr))])
    
    delta_arr_mean = np.mean(delta_arr, axis=0)
    import matplotlib.pyplot as plt
    import scienceplots
    with plt.style.context(["science", "ieee"]):
        fig, ax = plt.subplots()
        scatter = ax.scatter(grid["0.spec"]['spectrum']['wavelength'], grid["0.spec"]['spectrum']['flux_norm'], c=delta_arr_mean, cmap='plasma', s=1, alpha=0.5)
        ax.set_title("Delta graph")
        ax.set_xlabel(r"Wavelength, \AA")
        ax.set_ylabel(r"mean delta flux")
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Delta', fontsize=12)



        fig, ax = plt.subplots()
        good_delta_data = np.where(grid["0.spec"]['spectrum']['flux_norm'] < 0.1)
        sc = ax.scatter(grid["0.spec"]['spectrum']['wavelength'][good_delta_data], grid["0.spec"]['spectrum']['flux_norm'][good_delta_data], c=delta_arr_mean[good_delta_data], cmap='plasma', s=1)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label('Delta', fontsize=12)

    plt.show()