import os
from dech_processing import make_txt_from_spectra
from general_processing import detect_orders

dir_path = os.path.dirname(os.path.realpath(__file__))
folder_to_spectra = dir_path + "/R_Cam/"
spectra_content = os.listdir(folder_to_spectra)

spectra_content_old = [
    "20121126",
    "20140417",
    "20130529",
    "20111115",
    "20120802",
    "20140811",
    "20131009",
]

spectra_content = [
    "20120802",
    "20121126",
    "20130202",
    "20130529",
    "20131008",
    "20140417",
    "20140811",
]


spectra_path = folder_to_spectra + spectra_content[6] + "/"

data = make_txt_from_spectra(spectra_path, True, True)
orders = detect_orders(data)
print(orders)
