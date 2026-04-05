import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots


def import_data(path: str):
    # data = np.genfromtxt(path)
    data = pd.read_csv(path, sep='\t')

    data_vis = data.loc[data['Band'] == 'Vis.']

    return data_vis

def try_convert(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan
    
    

def plot(data: pd.DataFrame):
    fig, ax = plt.subplots()
    date = data["JD"].to_numpy()
    # m = data["Magnitude"].to_numpy(dtype=np.float16, na_value="-1", numeric_only=True)
    m = pd.to_numeric(data["Magnitude"], errors="coerce")
    ax.plot(date, m)
    plt.show()

if __name__ == "__main__":
    d_v = import_data("lightcurve.txt")
    plot(d_v)
    
    