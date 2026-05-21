import os


def read_element_data(filename):
    with open(filename, "r") as file:
        lines = file.readlines()

    i = 0
    elements_data = []
    while i < len(lines):
        line_parts = lines[i].split()
        if line_parts[0] == "'":
            atomic_num = line_parts[1]
        else:
            atomic_num = line_parts[0]
        ionization = int(line_parts[-2])
        num_lines = int(line_parts[-1])

        element_name = lines[i + 1].strip()

        for _ in range(num_lines):
            i += 1
            data_line = lines[i + 1]
            wavelength, elow, loggf, term = (
                float(data_line.split()[0]),
                float(data_line.split()[1]),
                float(data_line.split()[2]),
                str(data_line.split()[13]),
            )
            elements_data.append(
                (element_name, atomic_num, ionization, wavelength, loggf, elow, term)
            )

        i += 2

    return elements_data


def find_elements(elements_data, left_wavelength, right_wavelength, loggf_threshold):
    filtered_elements = []
    for element_data in elements_data:
        element_name, atomic_num, ionization, wavelength, loggf, elow, term = (
            element_data
        )
        if (
            left_wavelength <= wavelength <= right_wavelength
            and loggf > loggf_threshold
        ):
            filtered_elements.append(element_data)

    sorted_elements = sorted(
        filtered_elements, key=lambda x: x[3]
    )  # Sort by wavelength

    for element_data in sorted_elements:
        element_name, atomic_num, ionization, wavelength, loggf, elow, term = (
            element_data
        )
        print(
            element_name.replace("'", "").replace("NLTE", "").replace("LTE", ""),
            wavelength,
            elow,
            loggf,
            term,
        )


if __name__ == "__main__":
    linelist_path = "/home/alpha/miras_1/atlas/"
    linelist_filename = "nlte_ges_linelist_jmg04sep2023_I_II"
    print("element wavelength elow loggf term")

    left_wavelength = 5896.3  # change this to change the range of wavelengths to print
    right_wavelength = 5896.6
    loggf_threshold = -10  # change this to change the threshold for loggf

    elements_data = read_element_data(os.path.join(linelist_path, linelist_filename))
    find_elements(elements_data, left_wavelength, right_wavelength, loggf_threshold)
