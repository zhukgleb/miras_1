from signal import SIG_DFL

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import matplotlib


def import_data(path: str):
    # data = np.genfromtxt(path)
    data = pd.read_csv(path, sep='\t')

    data_vis = data.loc[data['Band'] == 'Vis.']
    data_r = data.loc[data['Band'] == 'R']
    data_b = data.loc[data['Band'] == 'B']
    data_v = data.loc[data['Band'] == 'V']

    return data_vis, data_b, data_v, data_r


def import_spectra(path: str):
    dates = np.genfromtxt(path)
    jd = dates[:, 2]
    return jd    
    

def nearest_mag(target_dates, obs_dates, obs_mags):

    obs_df = pd.DataFrame({
        'date': obs_dates,
        'mag': obs_mags
    }).sort_values('date')
    
    target_df = pd.DataFrame({'date': target_dates})
    
    result = pd.merge_asof(
        target_df.sort_values('date'),
        obs_df,
        on='date',
        direction='nearest'
    )
    
    return result['mag'].values


def make_phase(start_jd=2457616.7870, period=266.028084):
    phase_points = []
    for i in range(400):
        phase_point = start_jd - period / 10
        phase_points.append(phase_point)
        start_jd = phase_point

    return phase_points


def calculate_phase(jd):
    """
    https://simbad.cds.unistra.fr/simbad/sim-ref?bibcode=2019MNRAS.486.1907J
    """
    period = 266.028084
    epoh = 2457616.7870
    diff = abs(epoh - jd)
    raw_phase =  diff / period 
    return raw_phase % 1
    


def plot(data: pd.DataFrame):
    with plt.style.context('science'):

        font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

        matplotlib.rc('font', **font)

        fig, ax = plt.subplots(figsize=(6, 5))
        date = pd.to_numeric(data["JD"], errors="coerce")
        # m = data["Magnitude"].to_numpy(dtype=np.float16, na_value="-1", numeric_only=True)
        m = pd.to_numeric(data["Magnitude"], errors="coerce")
        spectra_jd = import_spectra("spectra_list.txt")

        nearest_m = nearest_mag(spectra_jd, date, m)

        plt.title("Observations of R Cam")

        
        ax.scatter(date, m, color="black", alpha=0.5, s=40, label="AAVSO data")    
        ax.scatter(spectra_jd,nearest_m, marker='X', s=200, color="crimson", label='NES Spectra')
        ax.set_xlim((2455580, 2457300))
        ax.set_ylim((7.2, 14.5)[::-1])
        ax.set_xlabel("JD")
        ax.set_ylabel("Visual magnitude")

        ax.set_xticks(np.linspace(2455580, 2457300, 5))


        legend = ax.legend(shadow=True)
        legend.get_frame().set_facecolor('C0')
        fig.tight_layout()  
        # plt.savefig("lightcurve.pdf", dpi=300)
        plt.show()



if __name__ == "__main__":
    d_v, d_b, d_v, d_r = import_data("lightcurve.txt")
    
    date_b = pd.to_numeric(d_b["JD"], errors="coerce")
    date_v = pd.to_numeric(d_v["JD"], errors="coerce")
    date_r = pd.to_numeric(d_r["JD"], errors="coerce")
    mb = pd.to_numeric(d_b["Magnitude"], errors="coerce")
    mv = pd.to_numeric(d_v["Magnitude"], errors="coerce")
    mr = pd.to_numeric(d_r["Magnitude"], errors="coerce")

    plt.scatter(date_b, mb)
    plt.scatter(date_v, mv)
    plt.scatter(date_r, mr)
    plt.show()

    # plot(d_v)
    
    # phase = calculate_phase(2456880.22292)    
    # print(phase)