import re
import json

def extract_iteration_variables(data):
    """
    Extracts iteration variables and their bounds from the provided data string.

    Args:
        data (str): Multi-line string containing iteration variables and bounds.

    Returns:
        dict: Dictionary with iteration variables as keys and their bounds as values.
        list: List of unmatched variables.
    """
    pattern = re.compile(r'"(\w+)":\s*"(\d+)"|"(bound[ul]\(\d+\))":\s*"([\d.]+)"')
    matches = pattern.findall(data)

    variables = {}
    unmatched = []
    for match in matches:
        if match[0]:
            variables[match[0]] = match[1]
        elif match[2]:
            variables[match[2]] = match[3]
        else:
            unmatched.append(match)

    return variables, unmatched

def save_to_json(data, filename):
    """
    Saves the provided data to a JSON file.

    Args:
        data (dict): Data to be saved.
        filename (str): Name of the file to save the data.
    """
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def main():
    # Sample data as a multi-line string (replace this with reading from a file if necessary)
    data = """
        "bt": "2",
        "boundu(2)": "20.0",
        "rmajor": "3",
        "boundu(3)": "20",
        "te": "4",
        "boundu(4)": "150.0",
        "beta": "5",
        "dene": "6",
        "fdene": "9",
        "boundu(9)": "1.2",
        "tfcth": "13",
        "fwalld": "14",
        "ohcth": "16",
        "boundl(16)": "0.1",
        "q": "18",
        "boundl(18)": "3.8",
        "fpnetel": "25",
        "bore": "29",
        "boundl(29)": "0.1",
        "fbetatry": "36",
        "coheof": "37",
        "fjohc": "38",
        "boundu(38)": "1.0",
        "fjohc0": "39",
        "boundu(39)": "1.0",
        "fcohbop": "41",
        "fvsbrnni": "44",
        "fstrcase": "48",
        "fstrcond": "49",
        "fiooic": "50",
        "boundu(50)": "1.0",
        "fvdump": "51",
        "fjprot": "53",
        "ftmargtf": "54",
    """

    matched, unmatched = extract_iteration_variables(data)
    print(matched)
    print(unmatched)
    # save_to_json(variables, 'iteration_variables.json')

if __name__ == "__main__":
    main()
