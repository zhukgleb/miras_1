from collections import defaultdict


def get_atlas():
    wavelength_dict = {}
    with open("lines.txt", "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            element = f"{parts[0]} {parts[1]}"  # Ti II, Ce II и т.п.
            wavelength = float(parts[2])
            term = parts[-1]
            value = f"{element} {term}"
            wavelength_dict[wavelength] = value

    return wavelength_dict


def find_duplicate_terms(wavelength_dict, mode="full"):
    """
    Находит одинаковые термы в словаре спектральных линий.

    Параметры:
        wavelength_dict (dict): {длина_волны: "Элемент терм"}
        mode (str): 'full' — полное совпадение строки значения,
                    'term_only' — совпадение только терма (последнего слова)

    Возвращает:
        dict: {терм/значение: [список длин волн]} для записей, встречающихся >1 раза
    """
    reverse_map = defaultdict(list)

    for wavelength, value in wavelength_dict.items():
        if mode == "term_only":
            # Берём последнее слово (терм) из строки значения
            term = value.split()[-1]
            key = term
        else:  # mode == 'full'
            key = value
        reverse_map[key].append(wavelength)

    duplicates = {k: v for k, v in reverse_map.items() if len(v) > 1}
    return duplicates


if __name__ == "__main__":
    wd = get_atlas()
    dp = find_duplicate_terms(wd)
    print(dp.items())
