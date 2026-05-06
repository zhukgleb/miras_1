from astropy.modeling import models
from astropy import units as u
from specutils import Spectrum1D
from astropy.modeling import models, fitting
from specutils.fitting import fit_lines


x = x * u.AA
y = y * u.Jy

spectrum = Spectrum1D(spectral_axis=x, flux=y)

g_init = models.Gaussian1D(amplitude=max(y), mean=6562 * u.AA, stddev=1.0 * u.AA)
g_fit = fit_lines(spectrum, g_init, window=(6560 * u.AA, 6566 * u.AA))
y_fit = g_fit(x)

with plt.style.context("science"):
    plt.plot(x, y, label="data")
    plt.plot(x, y_fit, label="gauss")
    plt.legend()
    plt.show()


g_emission = models.Gaussian1D(
    amplitude=100 * u.Unit(y.unit),
    mean=6564.6 * u.AA,  # Центр между пиками
    stddev=2.5 * u.AA,
)  # Широкий профиль

# 2. Компонент узкой абсорбции (то, что режет центр)
#    Обратите внимание: amplitude отрицательная!
g_absorption = models.Gaussian1D(
    amplitude=-50 * u.Unit(y.unit),
    mean=6563.8 * u.AA,  # Центр провала
    stddev=0.8 * u.AA,
)  # Уже, чем эмиссия

# Объединяем модели знаком "+"
# Это позволяет specutils фитировать их одновременно [citation:7]
composite_model = g_emission + g_absorption

# Задаем окно спектра, внутри которого будем фитировать
fitting_window = (6560 * u.AA, 6568 * u.AA)

# Запускаем фиттер Левенберга-Марквардта
fitter = fitting.LevMarLSQFitter()
fitted_model = fit_lines(spectrum, composite_model, window=fitting_window)

# Извлекаем подогнанные компоненты (их параметры уже оптимизированы)
fitted_emission = fitted_model[0]
fitted_absorption = fitted_model[1]

print(fitted_emission)  # Посмотрим результат
print(fitted_absorption)

x_fit = x
y_total_fit = fitted_model(x_fit)
y_emission_only = fitted_emission(x_fit)
y_absorption_only = fitted_absorption(x_fit)

plt.figure(figsize=(10, 6))
plt.plot(x, y, "k-", label="Исходный спектр", alpha=0.7)
plt.plot(
    x_fit, y_total_fit, "r--", linewidth=2, label="Sum (G_emission + G_absorption)"
)
plt.plot(x_fit, y_emission_only, "b:", label="G_emission (Широкая эмиссия)")
plt.plot(x_fit, y_absorption_only, "g:", label="G_absorption (Поглощение)")
plt.xlabel(f"Длина волны ({x.unit})")
plt.ylabel(f"Поток ({y.unit})")
plt.legend()
plt.title("Двухкомпонентный фит линии H_alpha")
plt.grid(alpha=0.3)
plt.show()
