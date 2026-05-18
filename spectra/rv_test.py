import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button, TextBox
import tkinter as tk
from tkinter import filedialog
import os

class SpectrumShifter:
    def __init__(self):
        self.spectra = []  # Список спектров [{wavelength, flux, shift, label, color}]
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        self.fig.subplots_adjust(bottom=0.2)
        
        self.selected_spectrum = None
        self.dragging = False
        self.start_x = 0
        self.start_shift = 0
        self.c = 299792.458  # скорость света в км/с
        
        # Создаем элементы управления
        self.create_widgets()
        
        # Подключаем обработчики событий
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.plot_spectra()
        plt.show()
    
    def create_widgets(self):
        # Кнопка загрузки
        ax_load = plt.axes([0.1, 0.05, 0.1, 0.04])
        self.btn_load = Button(ax_load, 'Загрузить')
        self.btn_load.on_clicked(self.load_spectrum)
        
        # Кнопка сброса
        ax_reset = plt.axes([0.22, 0.05, 0.1, 0.04])
        self.btn_reset = Button(ax_reset, 'Сбросить всё')
        self.btn_reset.on_clicked(self.reset_all)
        
        # Кнопка удаления выбранного
        ax_delete = plt.axes([0.34, 0.05, 0.15, 0.04])
        self.btn_delete = Button(ax_delete, 'Удалить выбранный')
        self.btn_delete.on_clicked(self.delete_selected)
        
        # Поле для ввода красного смещения
        ax_redshift_label = plt.axes([0.52, 0.08, 0.1, 0.04])
        ax_redshift_label.axis('off')
        ax_redshift_label.text(0.5, 0.5, 'z или v(км/с):', 
                              transform=ax_redshift_label.transAxes, 
                              ha='center', va='center')
        
        ax_redshift = plt.axes([0.52, 0.05, 0.1, 0.04])
        self.text_redshift = TextBox(ax_redshift, '', initial='0')
        self.text_redshift.on_submit(self.apply_redshift)
        
        # Кнопка переключения режима ввода
        ax_mode = plt.axes([0.64, 0.05, 0.1, 0.04])
        self.btn_mode = Button(ax_mode, 'Режим: z')
        self.btn_mode.on_clicked(self.toggle_mode)
        self.input_mode = 'z'  # 'z' или 'velocity'
        
        # Информационная панель
        ax_info = plt.axes([0.76, 0.05, 0.2, 0.04])
        ax_info.axis('off')
        self.info_text = ax_info.text(0, 0.5, '', transform=ax_info.transAxes, 
                                      va='center', fontsize=9)
        
        # Статус бар
        self.status_text = self.fig.text(0.5, 0.01, '', ha='center', fontsize=10,
                                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def load_spectrum(self, event=None):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title='Выберите файл спектра',
            filetypes=[('Text files', '*.txt *.dat *.csv'), ('All files', '*.*')]
        )
        root.destroy()
        
        if file_path:
            try:
                data = np.loadtxt(file_path)
                if data.shape[1] < 2:
                    self.update_status('Ошибка: файл должен содержать минимум 2 колонки')
                    return
                
                wavelength = data[:, 0]
                flux = data[:, 1]
                
                # Нормализуем поток
                flux = flux / np.median(flux)
                
                spectrum = {
                    'wavelength': wavelength,
                    'flux': flux,
                    'shift': 0.0,
                    'label': os.path.basename(file_path),
                    'color': plt.cm.tab10(len(self.spectra) % 10)
                }
                
                self.spectra.append(spectrum)
                self.selected_spectrum = len(self.spectra) - 1
                self.plot_spectra()
                self.update_status(f'Загружен: {spectrum["label"]}')
                
            except Exception as e:
                self.update_status(f'Ошибка загрузки: {str(e)}')
    
    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        
        if event.button == MouseButton.LEFT:
            # Проверяем, кликнули ли по спектру
            clicked_spectrum = self.find_nearest_spectrum(event.xdata, event.ydata)
            if clicked_spectrum is not None:
                self.selected_spectrum = clicked_spectrum
                self.dragging = True
                self.start_x = event.xdata
                self.start_shift = self.spectra[self.selected_spectrum]['shift']
                self.update_status(f'Выбран: {self.spectra[self.selected_spectrum]["label"]}')
                self.plot_spectra()
    
    def on_release(self, event):
        if self.dragging:
            self.dragging = False
            if self.selected_spectrum is not None:
                shift = self.spectra[self.selected_spectrum]['shift']
                z = shift / self.spectra[self.selected_spectrum]['wavelength'].mean()
                v = z * self.c
                self.update_status(f'Сдвиг: {shift:.2f} Å, z={z:.6f}, v={v:.2f} км/с')
    
    def on_motion(self, event):
        if self.dragging and event.inaxes == self.ax and self.selected_spectrum is not None:
            dx = event.xdata - self.start_x
            self.spectra[self.selected_spectrum]['shift'] = self.start_shift + dx
            self.plot_spectra()
    
    def on_key(self, event):
        if event.key == 'right' and self.selected_spectrum is not None:
            self.spectra[self.selected_spectrum]['shift'] += 0.02
            self.plot_spectra()
        elif event.key == 'left' and self.selected_spectrum is not None:
            self.spectra[self.selected_spectrum]['shift'] -= 0.02
            self.plot_spectra()
        elif event.key == 'up' and self.selected_spectrum is not None:
            self.spectra[self.selected_spectrum]['shift'] += 0.5
            self.plot_spectra()
        elif event.key == 'down' and self.selected_spectrum is not None:
            self.spectra[self.selected_spectrum]['shift'] -= 0.5
            self.plot_spectra()
        elif event.key == 'delete' and self.selected_spectrum is not None:
            self.delete_selected()
    
    def find_nearest_spectrum(self, x, y):
        min_dist = float('inf')
        nearest = None
        
        for i, spec in enumerate(self.spectra):
            shifted_wl = spec['wavelength'] + spec['shift']
            # Находим ближайшую точку спектра
            idx = np.argmin(np.abs(shifted_wl - x))
            if idx < len(spec['flux']):
                dist = abs(spec['flux'][idx] - y)
                if dist < min_dist and dist < 0.5:  # Порог близости
                    min_dist = dist
                    nearest = i
        
        return nearest
    
    def apply_redshift(self, text):
        if self.selected_spectrum is None:
            self.update_status('Сначала выберите спектр')
            return
        
        try:
            value = float(text)
            spec = self.spectra[self.selected_spectrum]
            
            if self.input_mode == 'z':
                z = value
            else:  # velocity
                z = value / self.c
            
            # Применяем красное смещение
            mean_wl = np.mean(spec['wavelength'])
            spec['shift'] = z * mean_wl
            
            self.plot_spectra()
            v = z * self.c
            self.update_status(f'Применено: z={z:.6f}, v={v:.2f} км/с, сдвиг={spec["shift"]:.2f} Å')
            
        except ValueError:
            self.update_status('Ошибка: введите число')
    
    def toggle_mode(self, event):
        if self.input_mode == 'z':
            self.input_mode = 'velocity'
            self.btn_mode.label.set_text('Режим: км/с')
        else:
            self.input_mode = 'z'
            self.btn_mode.label.set_text('Режим: z')
    
    def reset_all(self, event=None):
        for spec in self.spectra:
            spec['shift'] = 0.0
        self.selected_spectrum = None
        self.plot_spectra()
        self.update_status('Все сдвиги сброшены')
    
    def delete_selected(self, event=None):
        if self.selected_spectrum is not None:
            label = self.spectra[self.selected_spectrum]['label']
            self.spectra.pop(self.selected_spectrum)
            self.selected_spectrum = None if not self.spectra else 0
            self.plot_spectra()
            self.update_status(f'Удален: {label}')
    
    def plot_spectra(self):
        self.ax.clear()
        
        if not self.spectra:
            self.ax.set_title('Загрузите спектры для сравнения')
            self.fig.canvas.draw()
            return
        
        for i, spec in enumerate(self.spectra):
            shifted_wl = spec['wavelength'] + spec['shift']
            alpha = 1.0 if i == self.selected_spectrum else 0.5
            linewidth = 2.5 if i == self.selected_spectrum else 1.5
            
            # Добавляем информацию о сдвиге в легенду
            z = spec['shift'] / np.mean(spec['wavelength'])
            v = z * self.c
            label = f'{spec["label"]} (Δλ={spec["shift"]:.1f}Å, v={v:.0f} км/с)'
            
            self.ax.plot(shifted_wl, spec['flux'], 
                        color=spec['color'], 
                        label=label,
                        alpha=alpha,
                        linewidth=linewidth,
                        picker=5)
        
        self.ax.set_xlabel('Длина волны (Å)')
        self.ax.set_ylabel('Нормализованный поток')
        self.ax.set_title('Кликните по спектру и двигайте мышью | ← → ↑ ↓ для точной настройки')
        self.ax.legend(loc='upper right', fontsize=8)
        self.ax.grid(True, alpha=0.3)
        
        # Обновляем информацию
        if self.selected_spectrum is not None:
            spec = self.spectra[self.selected_spectrum]
            z = spec['shift'] / np.mean(spec['wavelength'])
            v = z * self.c
            self.info_text.set_text(f'Выбран: {spec["label"]}\nz={z:.6f}, v={v:.1f} км/с')
        else:
            self.info_text.set_text('Спектр не выбран')
        
        self.fig.canvas.draw()
    
    def update_status(self, message):
        self.status_text.set_text(message)
        self.fig.canvas.draw_idle()

if __name__ == '__main__':
    print("=" * 60)
    print("Программа для сравнения спектров и измерения смещения")
    print("=" * 60)
    print("\nИнструкция:")
    print("1. Нажмите 'Загрузить' чтобы добавить спектры")
    print("2. Кликните по спектру для выбора")
    print("3. Двигайте мышью для сдвига по длинам волн")
    print("4. Используйте стрелки для точной настройки")
    print("5. Введите z или скорость для точного смещения")
    print("6. Delete - удалить выбранный спектр")
    print("\nФормат файла: текстовый файл с колонками:")
    print("  1 колонка - длина волны, 2 колонка - поток")
    print("=" * 60)
    
    app = SpectrumShifter()