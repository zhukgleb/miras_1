import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

from scipy.interpolate import interp1d


def spectrum_finder(folder_data: list) -> list:
    spectrum_list = []
    for i in range(len(folder_data)):
        if str(folder_data[i]).find(".spec") != -1:
            spectrum_list.append(str(folder_data[i]))
    return spectrum_list


def data_graber(path2spectrum: Path) -> np.ndarray:
    data = np.genfromtxt(path2spectrum, comments="#")
    return data


def read_spectrum_grid(path2grid: Path) -> list:
    folder_data = os.listdir(path2grid)
    spectrum_list = spectrum_finder(folder_data)

    data = []
    for i in range(len(spectrum_list)):
        data.append(data_graber(path2grid / spectrum_list[i]))
    return data


def deriv_flux(normal_flux: np.ndarray) -> np.ndarray:
    delta_arr = []
    delta = 2
    for i in range(1, len(normal_flux)):
        delta_arr.append((abs(normal_flux[i] - normal_flux[i - 1])) / delta)

    delta_arr.append(1)
    return delta_arr


def parse_header_params(filename):
    params = {}

    with open(filename, "r") as f:
        for line in f:
            if line.startswith("#"):
                clean_line = line.strip("#").strip()
                if ":" in clean_line:
                    key, value = clean_line.split(":", 1)
                    key = key.strip()
                    value = value.strip()

                    # Обработка числовых параметров
                    if key in [
                        "teff",
                        "logg",
                        "[Fe/H]",
                        "vmic",
                        "vmac",
                        "resolution",
                        "rotation",
                    ]:
                        try:
                            params[key] = float(value)
                        except ValueError:
                            params[key] = value
                    elif key in ["nlte_flag"]:
                        params[key] = value.lower() == "true"
                    else:
                        params[key] = value
            else:
                break

    return params


def load_spectrum(filename):
    params = parse_header_params(filename)

    # Загружаем данные, пропуская строки с комментариями
    try:
        data = np.loadtxt(filename, comments="#")
        if data.ndim == 1:
            # Если только одна строка данных
            wavelength = np.array([data[0]])
            flux = np.array([data[1]])
        else:
            wavelength = data[:, 0]
            flux = data[:, 1]
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        data = []
        with open(filename, "r") as f:
            for line in f:
                if not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            data.append([float(parts[0]), float(parts[1])])
                        except ValueError:
                            continue
        data = np.array(data)
        if len(data) > 0:
            wavelength = data[:, 0]
            flux = data[:, 1]
        else:
            raise ValueError(f"No data found in {filename}")

    return wavelength, flux, params


def create_model_grid():
    model_files = ["1.spec", "2.spec", "3.spec", "4.spec", "5.spec"]
    model_spectra = []
    model_wavelengths = []
    model_params_list = []

    print("Loading model spectra...")
    for i, filename in enumerate(model_files):
        if os.path.exists(filename):
            wl, flux, params = load_spectrum(filename)
            model_spectra.append(flux)
            model_wavelengths.append(wl)

            teff = params.get("teff", 5500)
            logg = params.get("logg", 4.0)
            feh = params.get("[Fe/H]", 0.0)

            model_params_list.append([teff, logg, feh])

            print(f"Loaded {filename}:")
            print(f"  Points: {len(wl)}, Teff: {teff}, logg: {logg}, [Fe/H]: {feh}")
            print(f"  Flux range: [{flux.min():.3f}, {flux.max():.3f}]")
        else:
            print(f"Warning: File {filename} not found")
            continue

    if not model_spectra:
        raise ValueError("No model spectra found!")

    # is it simular wavelenght grid?
    first_wl = model_wavelengths[0]
    for i, wl in enumerate(model_wavelengths[1:], 1):
        if len(wl) != len(first_wl) or not np.allclose(wl, first_wl, rtol=1e-6):
            print(
                f"Warning: Wavelength scales differ for model {i + 1}. Interpolating..."
            )
            interp_func = interp1d(
                wl,
                model_spectra[i],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            model_spectra[i] = interp_func(first_wl)

    common_wavelength = first_wl
    model_spectra = np.array(model_spectra)
    model_params = np.array(model_params_list)

    return common_wavelength, model_spectra, model_params


def load_observed_spectrum(filename):
    print(f"Loading observed spectrum {filename}...")

    try:
        data = np.loadtxt(filename, comments="#")
        if data.ndim == 1:
            wavelength = np.array([data[0]])
            flux = np.array([data[1]])
        else:
            wavelength = data[:, 0]
            flux = data[:, 1]
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        data = []
        with open(filename, "r") as f:
            for line in f:
                if not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            data.append([float(parts[0]), float(parts[1])])
                        except ValueError:
                            continue
        data = np.array(data)
        if len(data) > 0:
            wavelength = data[:, 0]
            flux = data[:, 1]
        else:
            raise ValueError(f"No data found in {filename}")

    print(
        f"Observed spectrum: {len(wavelength)} points, flux range [{flux.min():.3f}, {flux.max():.3f}]"
    )

    return wavelength, flux


def main(save=False):
    model_wavelength, model_spectra, model_params = (
        create_model_grid()
    )  # Re-write to the any custom grid..
    n_models = len(model_spectra)

    # data = np.genfromtxt("0.spec", comments="#")
    data = np.genfromtxt("test_distorted_simple.spec", comments="#")
    obs_wavelength = data[:, 0]
    obs_spectrum = data[:, 1]
    del data
    from fit.ContFit import AutomaticContinuumFitter

    fitter = AutomaticContinuumFitter(
        model_wavelength,
        model_spectra,
        model_params,
        continuum_degree=4,
        n_iterations=100,
    )

    print("\nStarting continuum fitting...")
    final_continuum, history = fitter.fit(obs_wavelength, obs_spectrum)

    plt.figure(figsize=(15, 12))

    plt.subplot(2, 2, 1)
    plt.plot(
        obs_wavelength, obs_spectrum, "k-", label="obs spectra", alpha=0.7, linewidth=1
    )
    plt.plot(
        obs_wavelength, final_continuum, "r-", linewidth=2, label="finded continuum"
    )
    plt.xlabel("wavelenght, angstroms")
    plt.ylabel("flux")
    plt.legend()
    plt.title("Obs spectra and continuum")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    normalized_observed = obs_spectrum / final_continuum
    best_model_idx = history["best_model_indices"][-1]
    best_model_spectrum = model_spectra[best_model_idx]
    best_model_params = model_params[best_model_idx]

    plt.plot(
        obs_wavelength,
        normalized_observed,
        "r-",
        label="normalized obs",
        alpha=0.8,
        linewidth=1,
    )
    plt.plot(
        model_wavelength,
        best_model_spectrum,
        "b--",
        label=f"best model (Teff={best_model_params[0]:.0f})",
        alpha=0.8,
    )
    plt.xlabel("wavelenght (Å)")
    plt.ylabel("normalized flux")
    plt.legend()
    plt.title("best fit comp")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(history["distances"], "bo-", markersize=4)
    plt.xlabel("iteration")
    plt.ylabel("distance to model")
    plt.title("algoritm spread")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    model_interp = interp1d(
        model_wavelength,
        best_model_spectrum,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    best_model_interp = model_interp(obs_wavelength)

    difference = normalized_observed - best_model_interp
    plt.plot(obs_wavelength, difference, "g-", alpha=0.7, linewidth=1)
    plt.axhline(0, color="k", linestyle="--", alpha=0.5)
    plt.xlabel("wavelenght, ang")
    plt.ylabel("difference (obs - model)")
    plt.title("Difference between spectra")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig("continuum_fitting_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    if save:
        print("\nSaving results...")

    normalized_data = np.column_stack([obs_wavelength, normalized_observed])
    header = f"# Normalized spectrum\n# Original: 0.spec\n# Continuum fitted automatically\n# Best matching model: Teff={best_model_params[0]:.0f}, logg={best_model_params[1]:.2f}, [Fe/H]={best_model_params[2]:.2f}"
    if save:
        np.savetxt("0_normalized.spec", normalized_data, header=header, fmt="%.6f")

    continuum_data = np.column_stack([obs_wavelength, final_continuum])
    header_cont = f"# Fitted continuum\n# Original: 0.spec\n# Best matching model: Teff={best_model_params[0]:.0f}, logg={best_model_params[1]:.2f}, [Fe/H]={best_model_params[2]:.2f}"
    if save:
        np.savetxt("0_continuum.spec", continuum_data, header=header_cont, fmt="%.6f")

    best_final_idx = history["best_model_indices"][-1]
    best_final_params = history["best_params"][-1]

    print(f"\n=== RESULTS ===")
    print(f"Best matching model: {best_final_idx + 1}.spec")
    print(
        f"Best parameters: Teff={best_final_params[0]:.1f} K, "
        f"logg={best_final_params[1]:.2f}, [Fe/H]={best_final_params[2]:.2f}"
    )
    print(f"Final distance: {history['distances'][-1]:.6f}")
    print(f"Number of iterations: {len(history['distances'])}")

    print(f"\nResults saved to:")
    print(f"- 0_normalized.spec (normalized observed spectrum)")
    print(f"- 0_continuum.spec (fitted continuum)")
    print(f"- continuum_fitting_results.png (plots)")

    return final_continuum, history


if __name__ == "__main__":
    main()
    # deriv_map(Path("data/2025-10-14-20-15-43_0.6420588332457386_NLTE_synthetic_spectra_parameters/"))