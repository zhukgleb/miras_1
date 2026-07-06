import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scienceplots


SAVE = False


def cross_section_plot(nu, xs, molecule_name="TiO"):
    with plt.style.context(["science", "ieee"]):
        fig, ax = plt.subplots(figsize=(3, 3))

        ax.plot(nu, xs, "black", linewidth=0.7)
        ax.set_xlabel(r"Wavelength, $\AA$")
        ax.set_ylabel(r"Cross section, $\text{cm}^{-2}$ / molecule")
        ax.set_title(f"Cross section of {molecule_name}")
        plt.tight_layout()
        if SAVE:
            plt.savefig("xs.pdf")
        else:
            plt.show()


def optical_depth_plot(nu_molecule, tau, column_density):
    with plt.style.context(["science", "ieee"]):
        fig, ax = plt.subplots(figsize=(3, 3))

        ax.plot(nu_molecule, tau, color="black", linewidth=0.5)
        ax.set_xlabel(r"Wavelength, $\AA$")
        ax.set_ylabel(r"Optical depth $\tau$")
        cm = "cm"
        title = f"Optical depth N = {column_density:.1e}"
        dimension = r", \text{cm}$^{-2}$"
        ax.set_title(title + dimension)
        ax.set_yscale("log")  # Логарифмическая шкала для лучшей визуализации
        plt.tight_layout()
        plt.show()


def complex_graph(nu_molecule, F_star_interp, F_transmitted, nu_obs, F_obs, nu_molecule_s=[], stick=[], fancy=True):
    if fancy:
        with plt.style.context(["science", "ieee"]):
            fig, ax = plt.subplots(figsize=(3, 3))

            ax.plot(
                nu_molecule,
                F_star_interp,
                "k-",
                linewidth=1.0,
                label="Star model spectra",
            )
            ax.plot(
                nu_molecule,
                F_transmitted,
                "g-",
                linewidth=2,
                label="Transmitted spectra",
            )
            ax.set_xlabel(r"Wavelength $\AA$")
            cm = "cm"
            dimension = r", \text{erg/s/cm}$^{-1}$"
            ax.set_title(dimension)
            # plt.plot(nu_molecule, F_absorbed, "r-", linewidth=0.8, label="Поглощенный спектр ZrO")
            ax.set_ylabel("Flux, " + dimension)
            plt.legend()
            plt.xlim(nu_molecule.min(), nu_molecule.max())

            ax.plot(
                nu_obs, F_obs, color="navy", ls="--", linewidth=0.8, label="Observation"
            )
            # plt.plot(nu_molecule_s, stick * 10e22, color="black", label="stick spectra")
            plt.legend()

            plt.tight_layout()
            # plt.savefig("TiO_absorpmoleculen_spectrum.png", dpi=150)
            plt.show()
    else:
        with plt.style.context("classic"):
            fig, ax = plt.subplots()

            ax.plot(
                nu_molecule,
                F_star_interp,
                "k-",
                linewidth=1.0,
                label="Star model spectra",
            )
            ax.plot(
                nu_molecule,
                F_transmitted,
                linewidth=2,
                label="Transmitted spectra",
            )
            ax.set_xlabel(r"Wavelength $\AA$")
            cm = "cm"
            dimension = r", \text{erg/s/cm}$^{-1}$"
            ax.set_title(dimension)
            # plt.plot(nu_molecule, F_absorbed, "r-", linewidth=0.8, label="Поглощенный спектр ZrO")
            ax.set_ylabel("Flux, " + dimension)
            plt.legend()
            plt.xlim(nu_obs.min(), nu_obs.max())

            ax.plot(
                nu_obs, F_obs, color="navy", ls="--", linewidth=0.8, label="Observation"
            )
            if len(stick) > 0:
                plt.plot(nu_molecule_s, stick * 10e22, color="black", label="stick spectra")
            plt.legend()

            plt.tight_layout()
            # plt.savefig("TiO_absorpmoleculen_spectrum.png", dpi=150)
            plt.show()



if __name__ == "__main__":
    pass
