"""
A small module of various functions for working with dech20/95 output files
"""

from numpy import (
    genfromtxt,
    concatenate,
    column_stack,
    savetxt,
    ndarray,
    unique,
    zeros,
    int32,
    array,
    where,
)
from astropy.io import fits
from struct import unpack, calcsize
from os import listdir, sendfile


"""
There in Dech is an option to choose to save all orders at once into one file,
but in this case all orders will not be saved in order,
but written down in a column, which creates a little confusion
"""


def tab_spectra(filepath: str, save=False) -> ndarray:
    data = genfromtxt(filepath, skip_header=1)
    all_wave = []
    all_flux = []

    for i in range(0, len(data[0]), 2):
        all_wave.append(data[:, i])

    for i in range(1, len(data[0]), 2):
        all_flux.append(data[:, i])

    conc_wave = concatenate(all_wave)
    conc_flux = concatenate(all_flux)
    output = column_stack((conc_wave, conc_flux))

    if save:
        savetxt(filepath + ".concat", output)
        return output
    else:
        return output


"""
load a dech-specif. fits file
"""


def dech_fits_loader(path2data: str):
    hdu_list = fits.open(path2data)
    # print(hdu_list[0].header)
    image_data = [hdu_list[x].data for x in range(len(hdu_list))]
    return image_data[0]


"""
Сonverts from a hex file fds to a decimal one.
Of the useful and necessary -- this is the number of orders,
which affects the final extraction. It is also important
to control the binning, for example,
for 2 by 2 the block size will already be 1024,
and the number of orders will drop by half
"""


def fds_loader(path2data: str, orders_num=40, values_per_block=2048):
    block_size = 4
    total_values = orders_num * values_per_block

    with open(path2data, "rb") as file:
        file_data = file.read()

    expected_size = total_values * block_size
    if len(file_data) < expected_size:
        raise ValueError("Expetred size is larger, than expected")

    format_string = "<" + "i" * total_values
    unpacked_data = unpack(format_string, file_data[:expected_size])

    orders = [
        unpacked_data[i : i + values_per_block]
        for i in range(0, total_values, values_per_block)
    ]
    for i in range(len(orders)):
        orders[i] = list(orders[i])  # Not good cast
        for j in range(len(orders[i])):
            orders[i][j] = orders[i][j] * 10**-4  # in angstroms

    print(f"start wavelenght: {orders[0][0]}")
    print(f"start wavelenght: {orders[-1][-1]}")

    return orders


"""
Allows stitching of spectral orders.
currently removes duplicate values
"""


def glue_spectrum(spectrum, save=True) -> ndarray:
    _, idx = unique(spectrum[:, 0], return_index=True)
    if save:
        savetxt("glued_spectrum.txt", spectrum[idx])
    return spectrum[idx]


# Thanks Nosonov Dmitry for solve for .100 file
def read_100(path2file: str, headlen=10, fmtdat="h"):
    bindat = open(path2file, "rb")
    objname = bindat.read(headlen)
    ord_num, ord_len = unpack("HH", bindat.read(4))
    fmt_str = fmtdat * ord_len
    bsize = calcsize(fmtdat)
    data = zeros((ord_num, ord_len), int32)
    for i in range(ord_num):
        data[i] = unpack(fmt_str, bindat.read(bsize * ord_len))
    return objname, ord_num, ord_len, data


def read_ccm(ccm_path, verbose=False):
    fsz = 4  # размер float (4 байта)
    with open(ccm_path, "rb") as ccm_file:
        ord_num = unpack("i", ccm_file.read(fsz))[0]
        if verbose:
            print(f"{ccm_path} has {ord_num} orders")

        data = {}
        ordls = []

        # Число точек в каждой кривой для каждого порядка
        ordls = list(unpack(f"{'f' * ord_num}", ccm_file.read(fsz * ord_num)))

        for ornum, orlen in enumerate(ordls):
            if orlen:
                data[ornum + 1] = []

        maxo = int(max(ordls))
        lastnum = ordls.index(maxo) + 1

        for _ in range(maxo - 1):
            for j in range(ord_num):
                dot = unpack("ff", ccm_file.read(fsz * 2))
                if dot[0] and len(data[j + 1]) != ordls[j]:
                    data[j + 1].append(dot)

        for j in range(lastnum):
            dot = unpack("ff", ccm_file.read(fsz * 2))
            if dot[0] and len(data[j + 1]) != ordls[j]:
                data[j + 1].append(dot)

    return data


def make_txt_from_spectra(working_folder: str, verbose=True, cutbad=True):
    fnames_in_working_folder = listdir(working_folder)
    spectra_name = ""
    disp_name = ""
    ccm_name = ""

    for name in fnames_in_working_folder:
        if name.endswith(".100") or name.endswith(".200"):
            spectra_name = name
        elif name.endswith(".fds"):
            disp_name = name
        elif name.endswith(".ccm"):
            ccm_name = name
        else:
            pass
    print(spectra_name, disp_name, ccm_name)

    o_name, o_nu, o_len, data = read_100(working_folder + spectra_name)
    if verbose:
        print(
            f"Object name is: {o_name}, have a {o_nu} orders and {o_len} lenght of each"
        )
    fds_data = fds_loader(working_folder + disp_name, o_nu, o_len)

    if cutbad:
        for i in range(len(data)):
            for j in range(len(data[i])):
                if j < int(len(data[i]) * 0.1):
                    data[i][j] = 0
                if j > int(len(data[i]) * 0.9):
                    data[i][j] = 0

    data_conc = concatenate(data)
    fds_conc = concatenate(fds_data)
    resulted_data = column_stack((fds_conc, data_conc))
    wavelenght = 0.0
    clear_wave, clear_flux = [], []
    for i in range(len(resulted_data)):
        if resulted_data[:, 0][i] >= wavelenght:
            wavelenght = resulted_data[:, 0][i]
            clear_wave.append(resulted_data[:, 0][i])
            clear_flux.append(resulted_data[:, 1][i])

    # ccm_data = read_ccm(working_folder + ccm_name)
    resulted_data = column_stack((clear_wave, clear_flux))
    return resulted_data


def make_txt_from_fits(fits_path: str, fds_path: str):
    fits_data = dech_fits_loader(fits_path)
    orders_num = fits_data.shape[0]
    blocks = fits_data.shape[1]
    o = fds_loader(fds_path, orders_num, blocks)
    print(o)
    # data = []
    data = glue_spectrum(o, True)
    return data


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    #    working_folder = "/home/lambda/stellar_chem/iras07430/"
    #    a = make_txt_from_spectra(working_folder, cutbad=True)
    #    plt.plot(a[:, 0], a[:, 1])
    #    plt.show()
    #    savetxt("iras07_unnorm.txt", a)
    # o_name, o_num, o_len, data = read_100(working_folder + "/s693012s.100")
    # print(o_name, o_num, o_len)
    # fds = fds_loader(working_folder + "/s693011s.fds", o_num, o_len)
    # plt.plot(fds[0], data[0])
    # ccm = read_ccm(working_folder + "/s693012s.ccm")

    # data = tab_spectra("data/iras05113.tab")
    # print(data)
    # plt.plot(data[:, 0], data[:, 1])
    # plt.show()

    # data = fds_loader("data/fits/s693011s.fds")
    # data = make_txt_from_spectra("data/fits/e693006s.fits.200", "data/fits/s693011s.fds")
    # print(data)
    # tab_spectra("data/irasz_1.tab", save=True)
    # spectrum = genfromtxt(
    #    "/home/lambda/TSFitPy/input_files/observed_spectra/iras2020.txt"
    # )

    # new_data = glue_spectrum(data, True)
    # plt.plot(new_data[:, 0], new_data[:, 1])
    # plt.show()
    # np.savetxt("data/iras05113_unshifted.txt", new_data)

    # f_p = "/home/alpha/miras_1/spectra/R_Cam/20111115/e577016s.fits"
    # fds_p = "/home/alpha/miras_1/spectra/R_Cam/20111115/s577015s.fds"

    # d = make_txt_from_fits(fits_path=f_p, fds_path=fds_p)
    wf = "/home/alpha/miras_1/spectra/R_Cam/20111115/"
    rd = make_txt_from_spectra(wf, True, True)
    # plt.plot(rd[:, 0], rd[:, 1])
    # plt.show()
