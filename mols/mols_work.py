import numpy as np
import PyAstronomy.pyasl as pyasl

from scipy.interpolate import interp1d


# Чилавек-малекула
class Molecule:
    """
    Класс для представления молекулы с ее спектральными характеристиками.
    
    Attributes:
        name (str): Название молекулы
        velocity (float): Скорость движения в км/с
        column_density (float): Колонковая концентрация
        cross_section (np.ndarray): Массив с длинами волн (см⁻¹) и коэффициентами кросс-секции
        stick_spectrum (np.ndarray): Массив с длинами волн (см⁻¹) и интенсивностями (опционально)
        wavelengths (np.ndarray): Длины волн в ангстремах
        cross_section_values (np.ndarray): Коэффициенты кросс-секции
    """
    
    def __init__(self, name, velocity, column_density, cross_section, stick_spectrum=None):
        """
        Инициализация молекулы.
        
        Args:
            name (str): Название молекулы
            velocity (float): Скорость движения в км/с
            column_density (float): Колонковая концентрация
            cross_section (np.ndarray): Массив [длина_волны_см⁻¹, кросс_секция]
            stick_spectrum (np.ndarray, optional): Массив [длина_волны_см⁻¹, интенсивность]
        """
        self.name = name
        self.velocity = velocity
        self.column_density = column_density
        
        # Сохраняем исходные данные
        self.cross_section = cross_section
        self.stick_spectrum = stick_spectrum
        
        # Конвертируем длины волн из см⁻¹ в ангстремы
        # 1 см⁻¹ = 10^8 / (длина_волны_в_см⁻¹) ангстрем
        # Формула: λ(Å) = 10^8 / ν(см⁻¹)
        wavenumbers = cross_section[:, 0]
        self.wavelengths = 1e8 / wavenumbers  # Перевод в ангстремы
        self.cross_section_values = cross_section[:, 1]
        
        # Если есть stick спектр, конвертируем и его
        if stick_spectrum is not None:
            stick_wavenumbers = stick_spectrum[:, 0]
            self.stick_wavelengths = 1e8 / stick_wavenumbers
            self.stick_intensities = stick_spectrum[:, 1]
        else:
            self.stick_wavelengths = None
            self.stick_intensities = None
            
    def apply_doppler_shift(self, velocity_shift=None):
        """
        Применяет доплеровский сдвиг к длинам волн молекулы.
        
        Args:
            velocity_shift (float, optional): Дополнительное смещение скорости в км/с.
                                            Если None, используется self.velocity.
        
        Returns:
            tuple: (сдвинутые_длины_волн, кросс_секция) для кросс-секции
                и (сдвинутые_длины_волн, интенсивности) для stick спектра (если есть)
        """
        if velocity_shift is None:
            velocity_shift = self.velocity
        
        # Константа скорости света в км/с
        c = 299792.458  # км/с
        
        # Доплеровский фактор: z = v / c
        # Для релятивистского доплер-эффекта: λ_obs = λ_rest * sqrt((1 + β)/(1 - β))
        # где β = v/c
        beta = velocity_shift / c
        
        # Релятивистский фактор
        doppler_factor = np.sqrt((1 + beta) / (1 - beta))
        
        # Применяем сдвиг к длинам волн кросс-секции
        shifted_cross = self.wavelengths * doppler_factor
        
        # Применяем сдвиг к stick спектру, если он есть
        shifted_stick = None
        if self.stick_spectrum is not None:
            shifted_stick = self.stick_wavelengths * doppler_factor
    
        return shifted_cross, shifted_stick
        
    def get_optical_depth(self, velocity_shift=None):
        """
        Вычисляет оптическую глубину молекулы.
        """
        shifted_wavelengths, _ = self.apply_doppler_shift(velocity_shift)
        
        # Сортируем по длинам волн
        sort_idx = np.argsort(shifted_wavelengths)
        shifted_wavelengths = shifted_wavelengths[sort_idx]
        cross_section_sorted = self.cross_section_values[sort_idx]
        
        optical_depth = self.column_density * cross_section_sorted
        
        print(f"  optical_depth min/max: {optical_depth.min():.2e}, {optical_depth.max():.2e}")
        print(f"  optical_depth sum: {optical_depth.sum():.2e}")
        
        return np.column_stack((shifted_wavelengths, optical_depth))
    

    def __add__(self, other):
        """
        Перегрузка оператора + для сложения двух молекул.
        Возвращает оптическую глубину суммы двух молекул.
        
        Args:
            other (Molecule): Другая молекула
            
        Returns:
            np.ndarray: Массив [длина_волны_Å, суммарная_оптическая_глубина]
        """
        if not isinstance(other, Molecule):
            raise TypeError(f"Невозможно сложить Molecule с {type(other)}")
            
        # Получаем оптические глубины обеих молекул
        od1 = self.get_optical_depth()
        od2 = other.get_optical_depth()
        
        # Интерполируем на общую сетку длин волн
        # Берем пересечение длин волн/
        min_wavelength = max(od1[0, 0], od2[0, 0])
        max_wavelength = min(od1[-1, 0], od2[-1, 0])
        
        # Создаем общую сетку длин волн с шагом, соответствующим меньшему шагу
        step1 = od1[1, 0] - od1[0, 0]
        step2 = od2[1, 0] - od2[0, 0]
        step = min(step1, step2)
        
        common_wavelengths = np.arange(min_wavelength, max_wavelength, step)
        
        # Интерполируем оптические глубины на общую сетку
        od1_interp = np.interp(common_wavelengths, od1[:, 0], od1[:, 1])
        od2_interp = np.interp(common_wavelengths, od2[:, 0], od2[:, 1])
        
        # Суммируем оптические глубины
        total_od = od1_interp + od2_interp
        
        return np.column_stack((common_wavelengths, total_od))
    
    def __radd__(self, other):
        """
        Поддержка сложения с левой стороны (для sum() и т.д.)
        """
        if other == 0:
            return self.get_optical_depth()
        return self.__add__(other)
        
    def __repr__(self):
        return f"Molecule(name='{self.name}', velocity={self.velocity} km/s, column_density={self.column_density})"
        
    def __str__(self):
        return f"Molecule {self.name} (v={self.velocity} km/s, N={self.column_density}, AA={min(self.wavelengths)} : {max(self.wavelengths)})"


if __name__ == "__main__":
    cp1 = "/home/delta/exocross/input/TiO_all.xsec"
    cp2 = "/home/delta/exocross/input/ZrO_all.xsec"
    star_spectrum_path = "/home/delta/miras_1/mols/synth_all.spec"
    obs_spectrum_path = "/home/delta/miras_1/mols/norm_spectra.txt"
    cross_section_data_1 = np.genfromtxt(cp1)
    cross_section_data_2 = np.genfromtxt(cp2)
    nu_star, F_star_norm, F_star = np.loadtxt(star_spectrum_path, unpack=True)
    nu_obs, F_obs = np.loadtxt(obs_spectrum_path, unpack=True)

    mol1 = Molecule(
    name="TiO",
    velocity=-0,  
    column_density=2e16,  
    cross_section=cross_section_data_1,
    stick_spectrum=None
)
    # 4619
    mol2 = Molecule(
        name="ZrO",
        velocity=-0,  
        column_density=1e16, 
        cross_section=cross_section_data_2
    )

    optical_depth1 = mol1.get_optical_depth()
    optical_depth2 = mol2.get_optical_depth()

    total_optical_depth = mol1 + mol2



    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=2)
    ax[0].plot(optical_depth1[:, 0], optical_depth1[:, 1], label=f'{mol1.name} (OD)')
    ax[0].plot(optical_depth2[:, 0], optical_depth2[:, 1], label=f'{mol2.name} (OD)')
    ax[0].plot(total_optical_depth[:, 0], total_optical_depth[:, 1], '--', label='Total OD')
    ax[0].set_xlabel('Длина волны (Å)')
    ax[0].set_ylabel('Оптическая глубина')
    ax[0].legend()
    ax[0].set_title('Оптические глубины молекул')
    

    # print(mol1)
    # print(mol2)
    # print(f"Оптическая глубина для {mol1.name}: {len(optical_depth1)} точек")
    # print(f"Суммарная оптическая глубина: {len(total_optical_depth)} точек")

    # fig, ax = plt.subplots()
    # ax.plot(nu_obs,
    #     F_obs,
    #     "k-",
    #     linewidth=1.0,
    #     label="Observation spectra",
    # )
    # ax.plot(nu_star, F_star_norm, label="Model spectra")
    # ax.plot(total_optical_depth[:, 0], (total_optical_depth[:, 1] / np.median(total_optical_depth[:, 1])) / 40, '--', label='Total OD')
    # plt.legend()
    # plt.show()
    

    # График со всем на свете
    F_obs_recalibrated = F_obs * np.median(F_star) * 0.4
    F_interp_func = interp1d(nu_star, F_star, kind="linear", fill_value=0.0, bounds_error=False)
    F_star_interp = F_interp_func(total_optical_depth[:, 0])
    F_transmitted = F_star_interp * np.exp(-total_optical_depth[:, 1])

    # fig, ax = plt.subplots()
    ax[1].plot(nu_obs, F_obs_recalibrated, label="obs data recalibrated", color="black")
    ax[1].plot(total_optical_depth[:, 0], F_transmitted, label="Transmited")

    ax[0].plot([4637.909, 4637.909], [0, max(optical_depth2[:, 1])], label="element 1")
    ax[0].plot([4638.95, 4638.95], [0, max(optical_depth2[:, 1])], label="element 2")
    ax[1].plot([4637.909, 4637.909], [0, max(F_obs_recalibrated)], label="element 1")
    ax[1].plot([4638.95, 4638.95], [0, max(F_obs_recalibrated)], label="element 2")
    

    ax[1].legend()
    # ax[0].set_xlim((4612, 4679))    
    # ax[1].set_xlim((4612, 4679))
    plt.show()


    
    import scienceplots
    with plt.style.context(["science", "ieee"]):
        fig, ax = plt.subplots(ncols=2, figsize=(6, 4))
        ax[0].set_xlim((4220, 4235))
        ax[0].plot(nu_star, F_star_norm, label="Synth data norm", color="crimson", alpha=0.8)
        ax[0].plot(nu_obs, F_obs, label="Obs data norm", color="black", alpha=0.7)
        ax[0].plot([4226.728, 4226.728], [0, 3], label="Ca I 4226.728", color="green", linestyle="-.")
        ax[0].set_title("Ca I line")
        ax[0].set_ylim((0, 1.5))


        ax[1].set_title("Na I line")
        ax[1].set_xlim((5886, 5900))
        ax[1].plot(nu_star, F_star_norm, label="Synth data norm", color="crimson", alpha=0.8)
        ax[1].plot(nu_obs, F_obs, label="Obs data norm", color="black", alpha=0.7)
        ax[1].plot([5889.95, 5889.95], [0, 3], label="Na I 5889.95", color="green", linestyle="-.")
        ax[1].plot([5895.92, 5895.92], [0, 3], label="Na I 5895.92", color="green", linestyle="-.")
        ax[1].set_ylim((0, 1.5))
        ax[0].legend()
        ax[1].legend()
        plt.tight_layout()
        plt.show()
        # plt.savefig("rv_lines.pdf")

