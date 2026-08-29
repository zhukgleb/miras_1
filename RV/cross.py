from specutils import Spectrum
import astropy.units as u
from specutils.analysis import template_correlate
from specutils.analysis import template_comparison
import matplotlib.pyplot as plt
import numpy as np
from astropy.nddata import StdDevUncertainty


obs_data = np.genfromtxt(
    "/home/delta/looks_the_same/6/obs_norm_molecular_corrected.txt"
)
molecular_data = np.genfromtxt("/home/delta/looks_the_same/6/molecular_combined.txt")

for i in range(len(obs_data)):
    if obs_data[:, 1][i] > 1:
        obs_data[:, 1][i] = 0

obs_wave = obs_data[:, 0] * u.AA
obs_flux = obs_data[:, 0] * u.mJy

molecular_wave = molecular_data[:, 0] * u.AA
molecular_flux = molecular_data[:, 1] * u.mJy


obs = Spectrum(
    flux=obs_flux * np.random.randn(len(obs_flux)),
    spectral_axis=obs_wave,
    uncertainty=StdDevUncertainty(np.random.sample(len(obs_flux))),
)
mol = Spectrum(
    flux=molecular_flux * np.random.randn(len(molecular_flux)),
    spectral_axis=molecular_wave,
    uncertainty=StdDevUncertainty(np.random.sample(len(molecular_flux))),
)

corr, lag = template_correlate(mol, mol, method="direct")
# resample_method = "flux_conserving"

# tm_result = template_comparison.template_match(
#    observed_spectrum=obs,
#    spectral_templates=mol,
#    resample_method=resample_method,
# )

print(tm_result[0])
# plt.plot(obs_data[:, 0], obs_data[:, 1])
# plt.plot(molecular_data[:, 0], molecular_data[:, 1])
