import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import PyAstronomy.pyasl as pyasl

# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================

# 1.1 Загружаем спектр TiO от ExoCross (сечение поглощения)
# Предполагаем формат: столбец 1 - волновое число (см^-1), столбец 2 - сечение (см^2/молекула)

tio_spectrum_path = "/home/delta/exocross/input/YO_Mira_na.xsec"
star_spectrum_path = "/home/delta/exocross/input/tests/na/0.spec"
obs_spectrum_path = "/home/delta/exocross/input/tests/na/na_6.txt"

nu_tio, sigma_tio = np.loadtxt(tio_spectrum_path, unpack=True)
nu_tio = 10e7 / nu_tio
# rv = 34.13 km/s ?
# array[array[:, column_index].argsort()]
tio = np.column_stack((nu_tio, sigma_tio))
tio = tio[tio[:, 0].argsort()]
nu_tio, sigma_tio = tio[:, 0], tio[:, 1]
_, nu_tio = pyasl.dopplerShift(nu_tio, sigma_tio, -50, edgeHandling="firstlast")


# 1.2 Загружаем спектр звезды (поток излучения)
# Формат: столбец 1 - волновое число (см^-1), столбец 2 - поток (эрг/с/см^2/см^-1)
nu_star, _, F_star = np.loadtxt(star_spectrum_path, unpack=True)
nu_obs, F_obs = np.loadtxt(obs_spectrum_path, unpack=True)
F_obs = F_obs * np.median(F_star)


# 1.3 Задаем столбиковую концентрацию TiO (см^-2)
# Это значение нужно взять из модели атмосферы или задать вручную
N_TiO = 1e17  # Пример для холодного гиганта, измените под свою задачу

# ============================================================
# 2. ИНТЕРПОЛЯЦИЯ СПЕКТРА ЗВЕЗДЫ НА СЕТКУ TiO
# ============================================================

# Создаем интерполяционную функцию для потока звезды
F_interp_func = interp1d(
    nu_star,
    F_star,
    kind="linear",
    fill_value=0.0,  # За пределами диапазона - ноль
    bounds_error=False,
)

# Интерполируем поток на сетку TiO
F_star_interp = F_interp_func(nu_tio)

# ============================================================
# 3. РАСЧЕТ ОПТИЧЕСКОЙ ТОЛЩИ
# ============================================================

# Оптическая толща tau = N * sigma (безразмерная величина)
tau = N_TiO * sigma_tio

# ============================================================
# 4. РАСЧЕТ ПОГЛОЩЕННОГО СПЕКТРА (закон Бугера-Ламберта-Бера)
# ============================================================

# Поток, прошедший сквозь слой: F = F0 * exp(-tau)
F_transmitted = F_star_interp * np.exp(-tau)

# Поглощенный поток: F_abs = F0 - F_transmitted = F0 * (1 - exp(-tau))
F_absorbed = F_star_interp * (1.0 - np.exp(-tau))

# ============================================================
# 5. (ОПЦИОНАЛЬНО) РАСЧЕТ ИНТЕГРАЛЬНОЙ МОЩНОСТИ ПОГЛОЩЕНИЯ
# ============================================================

# Интегрируем поглощенный спектр по волновому числу, чтобы получить
# полную мощность поглощения на 1 см^2 поверхности (эрг/с/см^2)
P_abs_total = np.trapezoid(F_absorbed, nu_tio)

# Чтобы получить мощность на одну молекулу (эрг/с/молекулу),
# делим на столбиковую концентрацию
P_abs_per_molecule = P_abs_total / N_TiO

print(f"Столбиковая концентрация TiO: {N_TiO:.2e} см^-2")
print(f"Полная поглощенная мощность на 1 см^2: {P_abs_total:.3e} эрг/с/см^2")
print(f"Поглощенная мощность на 1 молекулу: {P_abs_per_molecule:.3e} эрг/с/молекулу")

# ============================================================
# 6. ПОСТРОЕНИЕ ГРАФИКОВ
# ============================================================

plt.figure(figsize=(12, 10))

# График 1: Сечение поглощения TiO
plt.subplot(3, 1, 1)
plt.plot(nu_tio, sigma_tio, "b-", linewidth=0.5)
plt.xlabel("Волновое число (см$^{-1}$)")
plt.ylabel("Сечение (см$^2$/молекула)")
plt.title("Сечение поглощения TiO")
plt.grid(True, alpha=0.3)

# График 2: Оптическая толща
plt.subplot(3, 1, 2)
plt.plot(nu_tio, tau, "r-", linewidth=0.5)
plt.xlabel("Волновое число (см$^{-1}$)")
plt.ylabel("Оптическая толща $\\tau$")
plt.title(f"Оптическая толща TiO (N = {N_TiO:.1e} см$^{{-2}}$)")
plt.grid(True, alpha=0.3)
plt.yscale("log")  # Логарифмическая шкала для лучшей визуализации

# График 3: Исходный и поглощенный спектры
plt.subplot(3, 1, 3)
plt.plot(nu_tio, F_star_interp, "k-", linewidth=1.0, label="Исходный спектр звезды")
plt.plot(
    nu_tio, F_transmitted, "g-", linewidth=0.8, label="Спектр после поглощения TiO"
)
plt.plot(nu_tio, F_absorbed, "r-", linewidth=0.8, label="Поглощенный спектр TiO")
plt.xlabel("Волновое число (см$^{-1}$)")
plt.ylabel("Поток (эрг/с/см$^2$/см$^{-1}$)")
plt.title("Спектры: исходный, прошедший и поглощенный")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(nu_tio.min(), nu_tio.max())  # Ограничиваем диапазон, чтобы видеть детали


# observation
plt.plot(nu_obs, F_obs, color="navy", ls="--", linewidth=0.8, label="Наблюдения")
plt.xlabel("Длина волны, АА")
plt.ylabel("Поток (эрг/с/см$^2$/см$^{-1}$)")
plt.title("Спектры: исходный, прошедший и поглощенный")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(nu_obs.min(), nu_obs.max())  # Ограничиваем диапазон, чтобы видеть детали


plt.tight_layout()
# plt.savefig("TiO_absorption_spectrum.png", dpi=150)
plt.show()

# ============================================================
# 7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ В ФАЙЛ
# ============================================================

# Сохраняем поглощенный спектр в файл для дальнейшего использования
save = False
if save:
    output_data = np.column_stack((nu_tio, F_absorbed))
    np.savetxt("TiO_absorbed_spectrum.txt", output_data)
    np.savetxt("normalized_obs_na.txt", np.column_stack((nu_obs, F_obs)))
    np.savetxt("TiO_pure.txt", np.column_stack((nu_tio, tau)))

    print("\nРезультаты сохранены в файлы:")
    print("  - TiO_absorbed_spectrum.dat  (поглощенный спектр)")
    print("  - TiO_absorption_spectrum.png (графики)")

# ============================================================
# 8. РАСЧЕТ СУММАРНОГО СПЕКТРА (звезда + TiO)
# ============================================================

# Если у вас есть атомарный спектр звезды с учетом других линий,
# вы можете сложить его с поглощением TiO

# F_total = F_star_atomic - F_absorbed_TiO  (если F_star_atomic уже включает
#                                              другие линии поглощения)

# Или, если F_star_interp - это континуум без линий:
# F_total = F_star_interp - F_absorbed  (но это будет спектр с линиями TiO)

print("\nГотово! Спектр поглощения TiO рассчитан с учетом столбиковой концентрации.")
