import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button, TextBox, CheckButtons
from matplotlib.lines import Line2D
import tkinter as tk
from tkinter import filedialog, messagebox
import os


class SpectrumShifter:
    def __init__(self):
        self.spectra = []  # Список спектров [{wavelength, flux, shift, label, color}]
        self.atlas_lines = []  # Список линий атласа [{wavelength, element, elow, loggf, term, visible, color}]
        self.atlas_loaded = False

        self.fig = plt.figure(figsize=(16, 9))
        self.ax = self.fig.add_axes([0.08, 0.25, 0.9, 0.7])  # Основной график

        self.selected_spectrum = None
        self.selected_atlas_line = None
        self.dragging = False
        self.start_x = 0
        self.start_shift = 0
        self.c = 299792.458  # скорость света в км/с

        # Для выделения области
        self.rect_selector = None
        self.rect_start = None

        # Создаем элементы управления
        self.create_widgets()

        # Подключаем обработчики событий
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)

        self.plot_spectra()
        plt.show()

    def create_widgets(self):
        # Кнопка загрузки спектра
        ax_load = plt.axes([0.02, 0.16, 0.08, 0.03])
        self.btn_load = Button(ax_load, "Загрузить\nспектр")
        self.btn_load.on_clicked(self.load_spectrum)

        # Кнопка загрузки атласа
        ax_load_atlas = plt.axes([0.02, 0.12, 0.08, 0.03])
        self.btn_load_atlas = Button(ax_load_atlas, "Загрузить\nатлас")
        self.btn_load_atlas.on_clicked(self.load_atlas)

        # Кнопка сохранения атласа
        ax_save_atlas = plt.axes([0.02, 0.08, 0.08, 0.03])
        self.btn_save_atlas = Button(ax_save_atlas, "Сохранить\nатлас")
        self.btn_save_atlas.on_clicked(self.save_atlas)

        # Кнопка сброса
        ax_reset = plt.axes([0.02, 0.04, 0.08, 0.03])
        self.btn_reset = Button(ax_reset, "Сбросить\nвсё")
        self.btn_reset.on_clicked(self.reset_all)

        # Кнопка удаления спектра
        ax_delete = plt.axes([0.02, 0.0, 0.08, 0.03])
        self.btn_delete = Button(ax_delete, "Удалить\nспектр")
        self.btn_delete.on_clicked(self.delete_selected)

        # Поле для ввода красного смещения
        ax_redshift_label = plt.axes([0.12, 0.19, 0.08, 0.03])
        ax_redshift_label.axis("off")
        ax_redshift_label.text(
            0.5,
            0.5,
            "z или v(км/с):",
            transform=ax_redshift_label.transAxes,
            ha="center",
            va="center",
            fontsize=8,
        )

        ax_redshift = plt.axes([0.12, 0.16, 0.08, 0.03])
        self.text_redshift = TextBox(ax_redshift, "", initial="0")
        self.text_redshift.on_submit(self.apply_redshift)

        # Кнопка переключения режима ввода
        ax_mode = plt.axes([0.12, 0.12, 0.08, 0.03])
        self.btn_mode = Button(ax_mode, "Режим: z")
        self.btn_mode.on_clicked(self.toggle_mode)
        self.input_mode = "z"  # 'z' или 'velocity'

        # Кнопка удаления линии атласа
        ax_delete_atlas = plt.axes([0.12, 0.08, 0.08, 0.03])
        self.btn_delete_atlas = Button(ax_delete_atlas, "Удалить\nлинию")
        self.btn_delete_atlas.on_clicked(self.delete_atlas_line)

        # Кнопка восстановления всех линий атласа
        ax_restore_atlas = plt.axes([0.12, 0.04, 0.08, 0.03])
        self.btn_restore_atlas = Button(ax_restore_atlas, "Восстан.\nлинии")
        self.btn_restore_atlas.on_clicked(self.restore_atlas_lines)

        # Чекбокс для отображения/скрытия атласа
        ax_check = plt.axes([0.12, 0.0, 0.08, 0.03])
        self.check_atlas = CheckButtons(ax_check, ["Атлас"], [True])
        self.check_atlas.on_clicked(self.toggle_atlas_visibility)

        # Информационная панель
        self.info_text = self.fig.text(
            0.25,
            0.19,
            "",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        )

        # Статус бар
        self.status_text = self.fig.text(
            0.5,
            0.22,
            "",
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        )

        # Легенда для атласа
        self.atlas_info_text = self.fig.text(
            0.25,
            0.22,
            "",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
        )

    def load_spectrum(self, event=None):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Выберите файл спектра",
            filetypes=[("Text files", "*.txt *.dat *.csv"), ("All files", "*.*")],
        )
        root.destroy()

        if file_path:
            try:
                data = np.loadtxt(file_path)
                if data.shape[1] < 2:
                    self.update_status(
                        "Ошибка: файл должен содержать минимум 2 колонки"
                    )
                    return

                wavelength = data[:, 0]
                flux = data[:, 1]

                # Нормализуем поток
                flux = flux / np.median(flux)

                spectrum = {
                    "wavelength": wavelength,
                    "flux": flux,
                    "shift": 0.0,
                    "label": os.path.basename(file_path),
                    "color": plt.cm.tab10(len(self.spectra) % 10),
                }

                self.spectra.append(spectrum)
                self.selected_spectrum = len(self.spectra) - 1
                self.plot_spectra()
                self.update_status(f"Загружен спектр: {spectrum['label']}")

            except Exception as e:
                self.update_status(f"Ошибка загрузки спектра: {str(e)}")

    def load_atlas(self, event=None):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Выберите файл атласа",
            filetypes=[("Text files", "*.txt *.dat *.csv"), ("All files", "*.*")],
        )
        root.destroy()

        if file_path:
            try:
                # Читаем файл атласа
                with open(file_path, "r") as f:
                    lines = f.readlines()

                # Пропускаем заголовок
                header_found = True
                new_atlas_lines = []

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # Проверяем, является ли строка заголовком
                    if "element" in line.lower() and "wavelength" in line.lower():
                        header_found = True
                        print("header found")
                        continue

                    if not header_found:
                        # Пробуем определить, заголовок ли это
                        parts = line.split()
                        if len(parts) >= 5:
                            try:
                                float(
                                    parts[1]
                                )  # Пробуем преобразовать вторую колонку в число
                                header_found = True
                            except ValueError:
                                continue

                    # Парсим строку данных
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            element = str(parts[0]) + " " + str(parts[1])
                            wavelength = float(parts[2])
                            elow = float(parts[3])
                            loggf = float(parts[4])
                            term = " ".join(parts[5:])

                            new_atlas_lines.append(
                                {
                                    "element": element,
                                    "wavelength": wavelength,
                                    "elow": elow,
                                    "loggf": loggf,
                                    "term": term,
                                    "visible": True,
                                    "color": "red",
                                }
                            )
                        except (ValueError, IndexError):
                            continue

                if new_atlas_lines:
                    self.atlas_lines = new_atlas_lines
                    self.atlas_loaded = True
                    self.update_status(f"Загружен атлас: {len(self.atlas_lines)} линий")
                    self.plot_spectra()
                else:
                    self.update_status("Ошибка: не удалось распознать линии в файле")

            except Exception as e:
                self.update_status(f"Ошибка загрузки атласа: {str(e)}")

    def save_atlas(self, event=None):
        if not self.atlas_lines:
            self.update_status("Нет загруженного атласа для сохранения")
            return

        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(
            title="Сохранить атлас как",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        root.destroy()

        if file_path:
            try:
                # Сохраняем только видимые линии
                visible_lines = [line for line in self.atlas_lines if line["visible"]]

                with open(file_path, "w") as f:
                    f.write(
                        f"{'element':<8} {'wavelength':>10} {'elow':>8} {'loggf':>8} term\n"
                    )
                    f.write("-" * 60 + "\n")

                    for line in visible_lines:
                        f.write(
                            f"{line['element']:<8} {line['wavelength']:>10.3f} "
                            f"{line['elow']:>8.3f} {line['loggf']:>8.3f} {line['term']}\n"
                        )

                self.update_status(f"Атлас сохранен: {len(visible_lines)} линий")

            except Exception as e:
                self.update_status(f"Ошибка сохранения: {str(e)}")

    def delete_atlas_line(self, event=None):
        if self.selected_atlas_line is not None:
            # Переключаем видимость выбранной линии
            line = self.atlas_lines[self.selected_atlas_line]
            line["visible"] = not line["visible"]

            status = "скрыта" if not line["visible"] else "восстановлена"
            self.update_status(
                f"Линия {line['element']} {line['wavelength']:.3f} {status}"
            )
            self.selected_atlas_line = None
            self.plot_spectra()

    def restore_atlas_lines(self, event=None):
        if self.atlas_lines:
            for line in self.atlas_lines:
                line["visible"] = True
            self.update_status("Все линии атласа восстановлены")
            self.plot_spectra()

    def toggle_atlas_visibility(self, label):
        if hasattr(self, "_atlas_globally_visible"):
            self._atlas_globally_visible = not self._atlas_globally_visible
        else:
            self._atlas_globally_visible = False
        self.plot_spectra()

    def find_nearest_atlas_line(self, x, y):
        if not self.atlas_lines:
            return None

        min_dist = float("inf")
        nearest = None

        # Преобразуем координаты в данные
        ylim = self.ax.get_ylim()
        y_range = ylim[1] - ylim[0]

        for i, line in enumerate(self.atlas_lines):
            if not line["visible"]:
                continue

            # Проверяем расстояние по X (должны быть близко к линии)
            x_dist = abs(line["wavelength"] - x)

            # Если курсор близко к линии по X и находится в области графика
            if x_dist < 5:  # 5 ангстрем
                if y > ylim[0] and y < ylim[1]:
                    if x_dist < min_dist:
                        min_dist = x_dist
                        nearest = i

        return nearest

    def on_press(self, event):
        if event.inaxes != self.ax:
            return

        if event.button == MouseButton.LEFT:
            # Сначала проверяем, не кликнули ли по линии атласа
            if event.key == "shift":  # Shift + клик для выбора линии атласа
                clicked_atlas = self.find_nearest_atlas_line(event.xdata, event.ydata)
                if clicked_atlas is not None:
                    self.selected_atlas_line = clicked_atlas
                    line = self.atlas_lines[clicked_atlas]
                    self.update_status(
                        f"Выбрана линия: {line['element']} {line['wavelength']:.3f} Å"
                    )
                    self.plot_spectra()
                    return
            elif event.key == "control":  # Ctrl + клик для удаления линии атласа
                clicked_atlas = self.find_nearest_atlas_line(event.xdata, event.ydata)
                if clicked_atlas is not None:
                    self.atlas_lines[clicked_atlas]["visible"] = False
                    self.selected_atlas_line = None
                    self.update_status(f"Линия скрыта")
                    self.plot_spectra()
                    return

            # Иначе работаем со спектрами
            clicked_spectrum = self.find_nearest_spectrum(event.xdata, event.ydata)
            if clicked_spectrum is not None:
                self.selected_spectrum = clicked_spectrum
                self.dragging = True
                self.start_x = event.xdata
                self.start_shift = self.spectra[self.selected_spectrum]["shift"]
                self.update_status(
                    f"Выбран спектр: {self.spectra[self.selected_spectrum]['label']}"
                )
                self.plot_spectra()

    def on_release(self, event):
        if self.dragging:
            self.dragging = False
            if self.selected_spectrum is not None:
                shift = self.spectra[self.selected_spectrum]["shift"]
                z = shift / self.spectra[self.selected_spectrum]["wavelength"].mean()
                v = z * self.c
                self.update_status(f"Сдвиг: {shift:.2f} Å, z={z:.6f}, v={v:.2f} км/с")

    def on_motion(self, event):
        if (
            self.dragging
            and event.inaxes == self.ax
            and self.selected_spectrum is not None
        ):
            dx = event.xdata - self.start_x
            self.spectra[self.selected_spectrum]["shift"] = self.start_shift + dx
            self.plot_spectra()

        # Обновляем информацию о положении курсора
        if event.inaxes == self.ax:
            self.update_cursor_info(event.xdata, event.ydata)

    def on_scroll(self, event):
        # Масштабирование колесиком мыши
        if event.inaxes == self.ax:
            scale_factor = 1.1
            xlim = self.ax.get_xlim()
            x_center = event.xdata

            if event.button == "up":
                new_width = (xlim[1] - xlim[0]) / scale_factor
            else:
                new_width = (xlim[1] - xlim[0]) * scale_factor

            new_xlim = [x_center - new_width / 2, x_center + new_width / 2]
            self.ax.set_xlim(new_xlim)
            self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == "right" and self.selected_spectrum is not None:
            self.spectra[self.selected_spectrum]["shift"] += 0.1
            self.plot_spectra()
        elif event.key == "left" and self.selected_spectrum is not None:
            self.spectra[self.selected_spectrum]["shift"] -= 0.1
            self.plot_spectra()
        elif event.key == "up" and self.selected_spectrum is not None:
            self.spectra[self.selected_spectrum]["shift"] += 1.0
            self.plot_spectra()
        elif event.key == "down" and self.selected_spectrum is not None:
            self.spectra[self.selected_spectrum]["shift"] -= 1.0
            self.plot_spectra()
        elif event.key == "delete" and self.selected_spectrum is not None:
            self.delete_selected()
        elif event.key == "d" and self.selected_atlas_line is not None:
            self.delete_atlas_line()
        elif event.key == "a":
            # Автомасштабирование
            self.ax.autoscale()
            self.fig.canvas.draw()

    def update_cursor_info(self, x, y):
        if self.atlas_lines:
            # Ищем ближайшую линию атласа
            nearest = None
            min_dist = float("inf")

            for line in self.atlas_lines:
                if line["visible"]:
                    dist = abs(line["wavelength"] - x)
                    if dist < min_dist:
                        min_dist = dist
                        nearest = line

            if nearest and min_dist < 1.0:
                self.atlas_info_text.set_text(
                    f"Ближайшая линия: {nearest['element']} {nearest['wavelength']:.3f}Å\n"
                    f"E_low={nearest['elow']:.3f} loggf={nearest['loggf']:.3f} {nearest['term']}"
                )

    def find_nearest_spectrum(self, x, y):
        min_dist = float("inf")
        nearest = None

        for i, spec in enumerate(self.spectra):
            shifted_wl = spec["wavelength"] + spec["shift"]
            idx = np.argmin(np.abs(shifted_wl - x))
            if idx < len(spec["flux"]):
                dist = abs(spec["flux"][idx] - y)
                if dist < min_dist and dist < 0.5:
                    min_dist = dist
                    nearest = i

        return nearest

    def apply_redshift(self, text):
        if self.selected_spectrum is None:
            self.update_status("Сначала выберите спектр")
            return

        try:
            value = float(text)
            spec = self.spectra[self.selected_spectrum]

            if self.input_mode == "z":
                z = value
            else:
                z = value / self.c

            mean_wl = np.mean(spec["wavelength"])
            spec["shift"] = z * mean_wl

            self.plot_spectra()
            v = z * self.c
            self.update_status(
                f"Применено: z={z:.6f}, v={v:.2f} км/с, сдвиг={spec['shift']:.2f} Å"
            )

        except ValueError:
            self.update_status("Ошибка: введите число")

    def toggle_mode(self, event):
        if self.input_mode == "z":
            self.input_mode = "velocity"
            self.btn_mode.label.set_text("Режим: км/с")
        else:
            self.input_mode = "z"
            self.btn_mode.label.set_text("Режим: z")

    def reset_all(self, event=None):
        for spec in self.spectra:
            spec["shift"] = 0.0
        self.selected_spectrum = None
        self.selected_atlas_line = None
        self.plot_spectra()
        self.update_status("Все сдвиги сброшены")

    def delete_selected(self, event=None):
        if self.selected_spectrum is not None:
            label = self.spectra[self.selected_spectrum]["label"]
            self.spectra.pop(self.selected_spectrum)
            self.selected_spectrum = None if not self.spectra else 0
            self.plot_spectra()
            self.update_status(f"Удален спектр: {label}")

    def plot_spectra(self):
        self.ax.clear()

        if not self.spectra and not self.atlas_lines:
            self.ax.set_title("Загрузите спектры и/или атлас для сравнения")
            self.fig.canvas.draw()
            return

        # Рисуем спектры
        for i, spec in enumerate(self.spectra):
            shifted_wl = spec["wavelength"] + spec["shift"]
            alpha = 1.0 if i == self.selected_spectrum else 0.5
            linewidth = 2.5 if i == self.selected_spectrum else 1.5

            z = spec["shift"] / np.mean(spec["wavelength"])
            v = z * self.c
            label = f"{spec['label']} (Δλ={spec['shift']:.1f}Å, v={v:.0f} км/с)"

            self.ax.plot(
                shifted_wl,
                spec["flux"],
                color=spec["color"],
                label=label,
                alpha=alpha,
                linewidth=linewidth,
                picker=5,
            )

        # Рисуем линии атласа
        if self.atlas_lines and (
            hasattr(self, "_atlas_globally_visible")
            and self._atlas_globally_visible
            or not hasattr(self, "_atlas_globally_visible")
            or self._atlas_globally_visible
        ):
            ylim = self.ax.get_ylim() if self.spectra else (0, 1)
            y_min, y_max = ylim

            for i, line in enumerate(self.atlas_lines):
                if line["visible"]:
                    color = "green" if i == self.selected_atlas_line else "red"
                    alpha = 1.0 if i == self.selected_atlas_line else 0.7
                    linewidth = 2.0 if i == self.selected_atlas_line else 1.0

                    # Рисуем вертикальную линию
                    self.ax.axvline(
                        x=line["wavelength"],
                        color=color,
                        alpha=alpha,
                        linewidth=linewidth,
                        linestyle="--",
                    )

                    # Добавляем подпись с информацией
                    if i == self.selected_atlas_line:
                        # Для выбранной линии показываем полную информацию
                        label = f"{line['element']}\n{line['wavelength']:.3f}\nE={line['elow']:.2f}\nloggf={line['loggf']:.2f}"
                        self.ax.text(
                            line["wavelength"],
                            y_max * 0.9,
                            label,
                            rotation=90,
                            fontsize=6,
                            ha="right",
                            va="top",
                            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
                        )
                    else:
                        # Для остальных - только элемент и длину волны
                        self.ax.text(
                            line["wavelength"],
                            y_max * 0.95,
                            f"{line['element']}\n{line['wavelength']:.3f}",
                            rotation=90,
                            fontsize=5,
                            ha="right",
                            va="top",
                        )

        self.ax.set_xlabel("Длина волны (Å)")
        self.ax.set_ylabel("Нормализованный поток")

        # Заголовок с инструкциями
        title = (
            "Клик: выбрать спектр | Shift+клик: выбрать линию | Ctrl+клик: скрыть линию"
        )
        if self.spectra:
            title += "\n← →: ±0.1Å | ↑ ↓: ±1Å | Колесико: масштаб | A: автомасштаб"
        self.ax.set_title(title, fontsize=9)

        if self.spectra:
            self.ax.legend(loc="upper right", fontsize=7)
        self.ax.grid(True, alpha=0.3)

        # Обновляем информацию
        if self.selected_spectrum is not None:
            spec = self.spectra[self.selected_spectrum]
            z = spec["shift"] / np.mean(spec["wavelength"])
            v = z * self.c
            self.info_text.set_text(
                f"Спектр: {spec['label']}\nz={z:.6f}\nv={v:.1f} км/с"
            )
        else:
            self.info_text.set_text("Спектр не выбран")

        if self.atlas_lines:
            visible_count = sum(1 for line in self.atlas_lines if line["visible"])
            total_count = len(self.atlas_lines)
            self.atlas_info_text.set_text(f"Атлас: {visible_count}/{total_count} линий")

        self.fig.canvas.draw()

    def update_status(self, message):
        self.status_text.set_text(message)
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    print("=" * 70)
    print("Программа для сравнения спектров и работы со спектральным атласом")
    print("=" * 70)
    print("\nИнструкция:")
    print("─" * 70)
    print("СПЕКТРЫ:")
    print("  • 'Загрузить спектр' - загрузка спектра (формат: длина_волны поток)")
    print("  • Клик по спектру - выбор для перемещения")
    print("  • Движение мыши - сдвиг спектра по длинам волн")
    print("  • Стрелки ← → - точная настройка (±0.1 Å)")
    print("  • Стрелки ↑ ↓ - быстрая настройка (±1 Å)")
    print("  • Поле ввода - точное z или скорость (км/с)")
    print("  • Delete - удалить выбранный спектр")
    print()
    print("АТЛАС:")
    print("  • 'Загрузить атлас' - загрузка линий (формат: element wl elow loggf term)")
    print("  • Shift+клик по линии - выбрать линию атласа")
    print("  • Ctrl+клик по линии - скрыть линию атласа")
    print("  • 'Удалить линию' - скрыть/показать выбранную линию")
    print("  • 'Восстановить линии' - показать все скрытые линии")
    print("  • 'Сохранить атлас' - сохранить только видимые линии")
    print("  • Чекбокс 'Атлас' - показать/скрыть все линии")
    print()
    print("НАВИГАЦИЯ:")
    print("  • Колесико мыши - масштабирование")
    print("  • Клавиша 'A' - автомасштабирование")
    print("  • 'Сбросить всё' - обнулить сдвиги спектров")
    print("=" * 70)
    print("\nЗапуск программы...")

    app = SpectrumShifter()
