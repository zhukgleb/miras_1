import pandas as pd
import numpy as np


if __name__ == "__main__":
    data = pd.read_csv("agb_table.csv")
    columns = data.columns.values
    sp_type = data["main_type"]
    sp_type_other = data["other_types"]
    sp_unique = pd.unique(sp_type)
    sq_other_unique = pd.unique(sp_type_other)  # ?! need to parse data
    post_agb_data = data.loc[data["main_type"] == "post-AGB*"]
    iras_name = post_agb_data["IRAS"]

    dummy_places = "--- --- --- --- --- --- --- --- ---"
    iras_name = iras_name.to_numpy(dtype=str)
    iras_name = ["iras" + iras_name[x] for x in range(len(iras_name))]
    dummy = [dummy_places for x in range(len(iras_name))]
    dummy = np.array(dummy, dtype=str)
    np.savetxt("post-agb_catalogue.txt", np.column_stack((iras_name, dummy)), fmt="%s")
