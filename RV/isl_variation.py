import numpy as np
import PyAstronomy.pyasl as pyasl
import matplotlib.pyplot as plt
import scienceplots

# Лучевые скорости (км/с)
rv = -79.5 - 1.06
rv_tio = -79 - 1.06
rv_combo = -80 - 1.06

# Загрузка данных
obs_norm_arr = []
for spec_num in range(6):
    obs_norm = np.genfromtxt(
        f"/home/delta/looks_the_same/{spec_num}/obs_norm_molecular_corrected.txt",
        skip_header=1,
    )
    obs_norm_arr.append(obs_norm)

# Словарь с межзвёздными линиями (длина волны в ангстремах, название)
ism_lines = {
    # Атомарные линии
    3933.66: "Ca II K",
    3968.47: "Ca II H",
    4226.73: "Ca I",
    5183.60: "Mg I",
    5890.00: "Na I D2",
    5895.92: "Na I D1",
    7698.97: "K I",
    7664.91: "K I",
    # Молекулярные линии
    4300.31: "CH (4300)",
    3874.61: "CN",
    3875.76: "CN",
    3883.37: "CN",
    3886.41: "CN",
    # DIB (диффузные межзвёздные полосы)
    4428.0: "DIB 4428",
    4501.0: "DIB 4501",
    5705.0: "DIB 5705",
    5780.5: "DIB 5780",
    5797.1: "DIB 5797",
    5850.0: "DIB 5850",
    6196.0: "DIB 6196",
    6203.0: "DIB 6203",
    6269.0: "DIB 6269",
    6283.8: "DIB 6283",
    6379.0: "DIB 6379",
    6613.6: "DIB 6613",
    6660.0: "DIB 6660",
    6840.0: "DIB 6840",
    7224.0: "DIB 7224",
}

# Сортировка по длине волны
ism_lines_sorted = dict(sorted(ism_lines.items()))


# Группировка линий по областям для построения
def plot_region(x_min, x_max, title, lines_dict, ax=None):
    """Построение графика для заданной области с отметками линий"""
    if ax is None:
        fig, ax = plt.subplots()

    # Отображаем все 6 спектров
    for spectra in range(6):
        ax.plot(
            obs_norm_arr[spectra][:, 0],
            obs_norm_arr[spectra][:, 1],
            label=f"obs {spectra}",
            alpha=0.7,
            linewidth=0.8,
        )

    # Отмечаем положения линий в этом диапазоне
    for wl, name in lines_dict.items():
        if x_min <= wl <= x_max:
            # Вертикальная линия в положении линии
            ax.axvline(x=wl, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
            # Подпись над линией
            y_pos = 1.3 if wl < (x_min + x_max) / 2 else 1.2
            ax.text(
                wl,
                y_pos,
                name,
                rotation=90,
                fontsize=7,
                horizontalalignment="center",
                verticalalignment="bottom",
                color="darkred",
            )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1.4)
    ax.set_title(title)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Normalized Flux")
    ax.legend(loc="upper right", fontsize=6, ncol=2)
    return ax


# Список областей для построения
regions = [
    # (x_min, x_max, title)
    (3870, 3890, "CN region"),
    (3930, 3970, "Ca II H & K region"),
    (4220, 4235, "Ca I region"),
    (4295, 4310, "CH region + DIB 4300?"),
    (4420, 4440, "DIB 4428 region"),
    (4500, 4510, "DIB 4501 region"),
    (5180, 5190, "Mg I region"),
    (5700, 5720, "DIB 5705 region"),
    (5770, 5810, "DIB 5780 region"),
    (5790, 5810, "DIB 5797 region"),
    (5840, 5860, "DIB 5850 region"),
    (5876, 5931, "Na I D1 & D2 region"),  # уже был
    (6190, 6210, "DIB 6196 & 6203 region"),
    (6260, 6280, "DIB 6269 region"),
    (6280, 6290, "DIB 6283 region"),
    (6370, 6390, "DIB 6379 region"),
    (6610, 6620, "DIB 6613 region"),
    (6650, 6670, "DIB 6660 region"),
    (6830, 6850, "DIB 6840 region"),
    (7220, 7230, "DIB 7224 region"),
    (7660, 7700, "K I region"),
]

# Построение всех графиков с единым стилем
with plt.style.context(["bmh"]):
    # Отдельный график для каждой области
    for x_min, x_max, title in regions:
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_region(x_min, x_max, title, ism_lines_sorted, ax)
        plt.tight_layout()

    # Дополнительно: обзорный график со всеми линиями (для общего понимания)
    fig, ax = plt.subplots(figsize=(14, 6))
    for spectra in range(6):
        ax.plot(
            obs_norm_arr[spectra][:, 0],
            obs_norm_arr[spectra][:, 1],
            alpha=0.3,
            linewidth=0.5,
            color="black",
        )

    # Отмечаем все линии на обзорном графике
    for wl, name in ism_lines_sorted.items():
        if 3800 < wl < 7800:  # ограничиваем видимый диапазон
            ax.axvline(x=wl, color="red", linestyle="--", alpha=0.5, linewidth=1)
            ax.text(
                wl,
                1.35,
                name,
                rotation=90,
                fontsize=6,
                horizontalalignment="center",
                verticalalignment="bottom",
                color="darkred",
                alpha=0.7,
            )

    ax.set_xlim(3800, 7800)
    ax.set_ylim(0, 1.4)
    ax.set_title("All ISM lines overview")
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Normalized Flux")
    plt.tight_layout()

    plt.show()
