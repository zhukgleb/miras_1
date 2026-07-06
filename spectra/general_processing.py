import numpy as np


def get_spectra_cut(start_wl, end_wl, data):
    delta = data[:, 0][1] - data[:, 0][0]

    cut_data = []
    start_idx = np.argmin(np.abs(data[:, 0] - start_wl))
    end_idx = np.argmin(np.abs(data[:, 0] - end_wl))

    cut_data.append(data[int(start_idx) : int(end_idx)])
    return cut_data[0]



# For future usage
def detect_orders(data):
    orders = []
    orders_gaps_idx = np.where(data[:, 1] == 0)
    import matplotlib.pyplot as plt
    plt.plot(data[:, 0], data[:, 1])
    plt.scatter(data[:, 0][orders_gaps_idx], data[:, 1][orders_gaps_idx])
    plt.show()
    return orders

def median_normalization(data):
    wavelengths = data[:, 0]
    flux = data[:, 1].copy() 
    
    zero_mask = (flux == 0)
    zero_indices = np.where(zero_mask)[0]
    
    if len(zero_indices) == 0:
        mask = (flux != 0)
        if np.any(mask):
            median_val = np.median(flux[mask])
            if median_val != 0:
                flux[mask] /= median_val
        return np.column_stack((wavelengths, flux))
    

    start_idx = 0
    for i, zero_idx in enumerate(zero_indices):
        order_slice = flux[start_idx:zero_idx]
        
        order_mask = (order_slice != 0)
        if np.any(order_mask):
            median_val = np.median(order_slice[order_mask])
            if median_val != 0:
                flux[start_idx:zero_idx][order_mask] /= median_val
        
        start_idx = zero_idx + 1
    
    if start_idx < len(flux):
        order_slice = flux[start_idx:]
        order_mask = (order_slice != 0)
        if np.any(order_mask):
            median_val = np.median(order_slice[order_mask])
            if median_val != 0:
                flux[start_idx:][order_mask] /= median_val
    
    return np.column_stack((wavelengths, flux))