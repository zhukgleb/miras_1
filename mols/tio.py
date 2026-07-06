import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scienceplots

# Загружаем твой спектр
wavelength_obs_mol, intensity_obs_mol, _ = np.genfromtxt(
    "0.spec_h2o", comments="#", unpack=True
)
wavelength_obs, _, intensity_obs = np.genfromtxt(
    "h_alpha_mira.spec", comments="#", unpack=True
)

wavelength_exo, sigma = np.loadtxt(
    "/home/delta/exocross/input/TiO_Mira_2000K_Voigt_halpha.xsec", unpack=True
)

# wavelength_obs, intensity_obs = observed[:, 0], observed[:, 1]
# wavelength_exo, sigma = exocross[:, 0], exocross[:, 1]  # сечение в см²
wavelength_exo = 10e7 / wavelength_exo

sigma_interp = interp1d(wavelength_exo, sigma, bounds_error=False, fill_value=0)(
    wavelength_obs
)

# Столбцовая концентрация (пример: 1e15 молекул/см²)
N = 2e24

# Расчет пропускания
transmission_model = intensity_obs * np.exp(-N * sigma_interp)

# Наложение на наблюдаемый спектр
with plt.style.context(["science", "ieee"]):
    # plt.plot(wavelength_obs_mol, intensity_obs_mol, label="Turbospectrum + Mols")
    # plt.plot(wavelength_obs, intensity_obs, label="Turbospectrum")
    # plt.plot(
    # wavelength_obs, transmission_model, label="Turbospectrum + ExoCross", alpha=0.7
    # )
    plt.plot(wavelength_exo, sigma)
    plt.xlabel("Wavelenght, AA")
    plt.ylabel("Intensity")
    plt.legend()
    plt.xlim((6552.8, 6572.8))
    plt.show()
