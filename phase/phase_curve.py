import numpy as np
import pandas as pd


def import_data(path: str):
    # data = np.genfromtxt(path)
    data = pd.read_csv(path, sep='\t')

    data_vis = data.loc[data['Band'] == 'Vis.']

    return data_vis


if __name__ == "__main__":
    d_v = import_data("lightcurve.txt")
    