import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import PyAstronomy.pyasl as pyasl
from plots import cross_section_plot, optical_depth_plot, complex_graph


# CONFIG
# molecule_spectrum_path = "/home/delta/exocross/input/ZrO_6400.xsec"
molecule_spectrum_path = "/home/delta/exocross/input/TiO_all.xsec"
stick_path = "/home/delta/exocross/input/TiO_6400.stick"
star_spectrum_path = "/home/delta/exocross/python/molecular_region_star.spec"
obs_spectrum_path = "/home/delta/exocross/python/cut_for_molecule.txt.norm"
scale_factor =  10e7
column_density = 5e15
# rv = -125
rv = -100
save = False
plots = True
mol_name = "ZrO"


nu_molecule, sigma_molecule = np.loadtxt(molecule_spectrum_path, unpack=True)
nu_molecule = scale_factor / nu_molecule
nu_molecule_s, stick = np.loadtxt(stick_path, unpack=True)
nu_molecule_s = scale_factor / nu_molecule_s

# rv = 34.13 km/s ?
molecule = np.column_stack((nu_molecule, sigma_molecule))
molecule = molecule[molecule[:, 0].argsort()]
molecules = np.column_stack((nu_molecule_s, stick))
molecules = molecules[molecules[:, 0].argsort()]


nu_molecule, sigma_molecule = molecule[:, 0], molecule[:, 1]
nu_molecule_s, stick = molecules[:, 0], molecules[:, 1]

# rv = -116
_, nu_molecule = pyasl.dopplerShift(
    nu_molecule, sigma_molecule, rv, edgeHandling="firstlast"
)
_, nu_molecule_s = pyasl.dopplerShift(
    nu_molecule_s, stick, rv, edgeHandling="firstlast"
)


nu_star, _, F_star = np.loadtxt(star_spectrum_path, unpack=True)
nu_obs, F_obs = np.loadtxt(obs_spectrum_path, unpack=True)
F_obs = F_obs * np.median(F_star)
F_obs = F_obs * 0.4


# N_TiO = 5e15  # Пример для холодного гиганта, измените под свою задачу

F_interp_func = interp1d(
    nu_star,
    F_star,
    kind="linear",
    fill_value=0.0,  # За пределами диапазона - ноль
    bounds_error=False,
)

# Интерполируем поток на сетку TiO
F_star_interp = F_interp_func(nu_molecule)
# Оптическая толща tau = N * sigma (безразмерная величина)
tau = column_density * sigma_molecule

# ============================================================
# 4. РАСЧЕТ ПОГЛОЩЕННОГО СПЕКТРА (закон Бугера-Ламберта-Бера)
# ============================================================

# Поток, прошедший сквозь слой: F = F0 * exp(-tau)
F_transmitted = F_star_interp * np.exp(-tau)

F_absorbed = F_star_interp * (1.0 - np.exp(-tau))

# 5. РАСЧЕТ ИНТЕГРАЛЬНОЙ МОЩНОСТИ ПОГЛОЩЕНИЯ

# Интегрируем поглощенный спектр по волновому числу, чтобы получить
# полную мощность поглощения на 1 см^2 поверхности (эрг/с/см^2)
P_abs_total = np.trapezoid(F_absorbed, nu_molecule)

# Чтобы получить мощность на одну молекулу (эрг/с/молекулу),
# делим на столбиковую концентрацию
P_abs_per_molecule = P_abs_total / column_density

print(f"Столбиковая концентрация TiO: {column_density:.2e} см^-2")
print(f"Полная поглощенная мощность на 1 см^2: {P_abs_total:.3e} эрг/с/см^2")
print(f"Поглощенная мощность на 1 молекулу: {P_abs_per_molecule:.3e} эрг/с/молекулу")

if save:
    output_data = np.column_stack((nu_molecule, F_absorbed))
    np.savetxt("TiO_absorbed_spectrum.txt", output_data)
    np.savetxt("normalized_obs_na.txt", np.column_stack((nu_obs, F_obs)))
    np.savetxt("TiO_pure.txt", np.column_stack((nu_molecule, tau)))

    print("\nРезультаты сохранены в файлы:")
    print("  - TiO_absorbed_spectrum.dat  (поглощенный спектр)")
    print("  - TiO_absorpmoleculen_spectrum.png (графики)")


# cross_section_plot(nu_molecule, sigma_molecule, mol_name)
# optical_depth_plot(nu_molecule, tau, column_density)
complex_graph(nu_molecule, F_star_interp, F_transmitted, nu_obs, F_obs, nu_molecule_s, stick, fancy=False)
