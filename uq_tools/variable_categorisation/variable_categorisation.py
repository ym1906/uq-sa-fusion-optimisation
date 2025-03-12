import re
import json
import os

def load_json(filename):
    """
    Loads JSON data from a file.

    Args:
        filename (str): Path to the JSON file.

    Returns:
        dict: Parsed JSON data.
    """
    with open(filename, "r") as file:
        return json.load(file)


def save_json(data, filename):
    """Save JSON data to a file."""
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)
    print(f"Updated data saved to {filename}")


def extract_iteration_variables(data):
    """
    Extracts iteration variables and their bounds from the provided data string.

    Args:
        data (str): Multi-line string containing iteration variables and bounds.

    Returns:
        dict: Dictionary with iteration variables as keys and their bounds as values.
    """
    pattern = re.compile(r'"(\w+)":\s*"(\d+)"|"(bound[ul]\(\d+\))":\s*"([\d.]+)"')
    matches = pattern.findall(data)

    variables = {}
    for match in matches:
        if match[0]:
            variables[match[0]] = match[1]
        else:
            variables[match[2]] = match[3]

    return variables


def process_variable_data(data, itv_path, output_path):
    """
    Processes variable data from a given string and categorizes them into iteration variables and other variables.

    This function reads a JSON file containing variable names and their indices, then processes a multi-line string
    to extract variables based on specific patterns. The extracted variables are categorized into iteration variables
    and other variables, and the results are saved to a specified output file in JSON format.

    :param data: A multi-line string containing variable data to be processed.
    :type data: str
    :param itv_path: Path to the JSON file containing variable names and their indices.
    :type itv_path: str
    :param output_path: Path to the output file where the processed variables will be saved in JSON format.
    :type output_path: str
    :return: A tuple containing two dictionaries: iteration variables and other variables.
    :rtype: tuple(dict, dict)
    """
    ixc_pattern = r"ixc\s*=\s*(\d+)"
    boundu_pattern = r"boundu\((\d+)\)\s*=\s*([^;]*)"
    boundl_pattern = r"boundl\((\d+)\)\s*=\s*([^;]*)"

    ixc_variables = {}
    other_variables = {}

    with open(itv_path, "r") as file:
        variable_names = json.load(file)

    for line in data.strip().split("\n"):
        ixc_match = re.search(ixc_pattern, line)
        boundu_match = re.search(boundu_pattern, line)
        boundl_match = re.search(boundl_pattern, line)

        if ixc_match:
            index = ixc_match.group(1)
            var_name = next(
                (name for name, num in variable_names.items() if num == index), None
            )
            if var_name:
                ixc_variables[var_name] = index
        elif boundu_match:
            index, value = boundu_match.groups()
            ixc_variables[f"boundu({index})"] = value.strip()
        elif boundl_match:
            index, value = boundl_match.groups()
            other_variables[f"boundl({index})"] = value.strip()
        else:
            # Capture all other variables
            var_match = re.match(r"(\w+)\s*=\s*([^;]*)", line)
            if var_match:
                var_name, value = var_match.groups()
                # Convert value to float if it's a number
                try:
                    value = float(value)
                except ValueError:
                    pass
                other_variables[var_name] = value
                # other_variables[var_name] = value.strip()
   # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the processed variables to the output file
    with open(output_path, "w") as json_file:
        json.dump({"ixc_variables": ixc_variables, "other_variables": other_variables}, json_file, indent=4)

    return ixc_variables, other_variables 

def save_to_json(data, filename):
    """
    Saves the provided data to a JSON file.

    Args:
        data (dict): Data to be saved.
        filename (str): Name of the file to save the data.
    """
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)


def categorize_associated_parameters(categories, fvalue_mapping, category_map):
    """
    Categorize parameters associated with each f-value based on existing categories.
    """
    for fvalue, associated_params in fvalue_mapping.items():
        for param in associated_params:
            current_category = category_map.get(param)
            if not current_category:
                for category, vars_in_category in categories.items():
                    if param in vars_in_category:
                        categories[category].append(param)
                        category_map[param] = category
                        break
                else:
                    categories.setdefault("Unknown Category", []).append(param)
                    category_map[param] = "Unknown Category"
    return categories


def update_descriptions(categories, descriptions):
    """
    Update variable descriptions in categories with those from descriptions.
    :param categories: JSON data from variable_categories.json
    :param descriptions: JSON data from variables.json
    :return: Updated categories data
    """
    # Iterate through each category and each variable in that category
    for category, variables in categories.items():
        for var in variables:
            var_name = var["Variable"]
            # Find the corresponding description in the descriptions JSON
            for filename, vars_dict in descriptions.items():
                if var_name in vars_dict:
                    # Update description if found
                    var["Description"] = vars_dict[var_name]
                    break  # Exit once description is updated
    return categories


def swap_f_parameters(ixc_relationship_dict, ixc_indat_dict):
    new_list = []

    for key, value in ixc_indat_dict.items():
        if re.match(r"bound[ul]\(\d+\)", key):
            continue
        if key.startswith("f"):
            if key in ixc_relationship_dict:
                new_list.extend(ixc_relationship_dict[key])
            else:
                new_list.append(key)
        else:
            new_list.append(key)

    return new_list


def parse_file(file_path):
    """
    Parses a file to extract key-value pairs based on a specific pattern.

    Args:
        file_path (str): The path to the file to be parsed.

    Returns:
        dict: A dictionary containing the extracted key-value pairs.
    """
    result_dict = {}

    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            if "!!" in line:
                match = re.search(r"\(\s*(\d+)\s*\)\s+([\w\(\)]+)", line)
                if match:
                    key = match.group(2).strip()
                    value = match.group(1)
                    result_dict[key] = value

    return result_dict


def load_categories(filename):
    with open(filename, "r") as file:
        data = json.load(file)

    category_map = {}
    for category, variables in data.items():
        for variable_info in variables:
            variable_name = variable_info["Variable"]
            category_map[variable_name] = category
    return category_map


def categorize_variables(
    variable_list,
    descriptions,
    categories_filename="/home/graeme/work-notes/uncertainty_quantification/variable_categorisation/variable_categories.json",
):
    """
    Categorizes variables into predefined categories based on their descriptions and a category mapping.

    This function loads a category mapping from a JSON file, then categorizes a list of variables into predefined
    categories such as "Design Variables", "Control Variables", "Design Parameters", and "Unknown Category".
    It also associates each variable with its source and description from the provided descriptions dictionary.

    :param variable_list: List of variables to be categorized.
    :type variable_list: list
    :param descriptions: Dictionary containing variable descriptions, with filenames as keys and variable descriptions as values.
    :type descriptions: dict
    :param categories_filename: Path to the JSON file containing the category mapping. Defaults to a specific path.
    :type categories_filename: str, optional
    :return: A dictionary with categorized variables, each category containing a description and a list of variables with their sources and descriptions.
    :rtype: dict
    """
    category_map = load_categories(categories_filename)

    result = {
        "Design Variables": {"description": "Variables that define the design of the system.", "variables": {}},
        "Control Variables": {"description": "Variables that control the operation of the system.", "variables": {}},
        "Design Parameters": {"description": "Parameters that influence the design but are not directly controlled.", "variables": {}},
        "Unknown Category": {"description": "Variables that do not fit into any known category.", "variables": {}},
    }

    for var in variable_list:
        category = category_map.get(var, "Unknown Category")
        if category not in result:
            result[category] = {"description": "No description available.", "variables": {}}

        found = False
        for filename, vars_dict in descriptions.items():
            if var in vars_dict:
                source = filename.replace(".f90", "")
                if source not in result[category]["variables"]:
                    result[category]["variables"][source] = []
                if not any(v['variable'] == var for v in result[category]["variables"][source]):
                    result[category]["variables"][source].append({"variable": var, "description": vars_dict[var]})
                found = True
                break
        if not found:
            if "Unknown Source" not in result[category]["variables"]:
                result[category]["variables"]["Unknown Source"] = []
            if not any(v['variable'] == var for v in result[category]["variables"]["Unknown Source"]):
                result[category]["variables"]["Unknown Source"].append({"variable": var, "description": "No description available."})

    result = {k: v for k, v in result.items() if v["variables"]}

    return result

def convert_to_markdown_table(categorized_variables):
    markdown = "# Variable Categorization\n\n"
    
    for category, details in categorized_variables.items():
        markdown += f"## {category}\n"
        markdown += f"{details['description']}\n\n"
        
        markdown += "| **Variable** | **Description** | **Source** |\n"
        markdown += "|----------|-------------|--------|\n"
        
        for source, variables in details["variables"].items():
            for var_info in variables:
                markdown += f"| `{var_info['variable']}` | {var_info['description']} | {source} |\n"
        
        markdown += "\n"
    
    return markdown

def update_variable_names_in_dict(data_dict, obsolete_dict):
    """
    Update variable names in the provided dictionary using the obsolete variables dictionary.
    
    Args:
    data_dict (dict): The original dictionary.
    obsolete_dict (dict): Dictionary containing obsolete variable names as keys and new variable names as values.
    
    Returns:
    dict: The updated dictionary with variable names replaced.
    """
    updated_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, dict):
            updated_dict[key] = update_variable_names_in_dict(value, obsolete_dict)
        else:
            new_key = obsolete_dict.get(key, key)
            updated_dict[new_key] = value
    return updated_dict

def update_variable_names_in_json(data, obsolete_dict):
    """
    Update variable names in the provided JSON data using the obsolete variables dictionary.
    
    Args:
    data (dict): The original JSON data.
    obsolete_dict (dict): Dictionary containing obsolete variable names as keys and new variable names as values.
    
    Returns:
    dict: The updated JSON data with variable names replaced.
    """
    for item in data.get("Design Variables", []):
        old_name = item.get("Variable")
        if old_name in obsolete_dict:
            item["Variable"] = obsolete_dict[old_name]
    return data

def create_sample_space(variables, multiplier, output_file, variables_to_extract=None):
    """
    Create a dictionary of parameters with min, max, and default values for float variables.

    This function iterates through the items in the `variables` dictionary, identifies float variables,
    and calculates their min and max values based on the provided multiplier. The resulting parameters
    are saved to the specified output file in JSON format.

    :param variables: Dictionary containing variable data.
    :type variables: dict
    :param multiplier: Multiplier for calculating min and max values as a fraction.
    :type multiplier: float
    :param output_file: Path to the output file where the parameters will be saved in JSON format.
    :type output_file: str
    :param variables_to_extract: List of variables to extract from variables. If None, use all variables.
    :type variables_to_extract: list, optional
    """
    params = {}
    for key, value in variables.items():
        if variables_to_extract is None or key in variables_to_extract:
            if isinstance(value, float):
                params[key] = {
                    "type": "float",
                    "min": value * (1 - multiplier),
                    "max": value * (1 + multiplier),
                    "default": value
                }

    # Add the last entry
    params["vary_param"] = {
        "type": "string",
        "default": ""
    }

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write the new JSON to the specified output file
    with open(output_file, 'w') as f:
        json.dump(params, f, indent=4)

def extract_variables(data):
    """
    Extract all 'variable' values from the provided JSON data and put them in a list.

    :param data: Dictionary containing the JSON data.
    :type data: dict
    :return: List of all 'variable' values.
    :rtype: list
    """
    variables = []
    for category, items in data.items():
        for item in items:
            variables.append(item["variable"])
    return variables

def create_variable_template(variables_list, input_data, filename):
    """
    Create a string with the specified format from the variables list,
    append it to the input data, and save it to a file with a ".template" extension.

    :param variables_list: List of variable names.
    :type variables_list: list
    :param input_data: Initial data to append the template data to.
    :type input_data: str
    :param filename: Name of the file to save the data to.
    :type filename: str
    :return: None
    """
    variable_string = "* Uncertainties: overwrites any previous values\n"
    for variable in variables_list:
        if "(" in variable:
            formatted_variable = variable.replace("(", "_").replace(")", "")
            variable_string += f"{variable} = ${formatted_variable}\n"
        else:
            variable_string += f"{variable} = ${variable}\n"
    
    # Append the variable string to the input data
    output_data = input_data + "\n" + variable_string
    
    # Strip ".DAT" and replace with ".template"
    if filename.endswith(".DAT"):
        filename = filename[:-4] + ".template"
    elif not filename.endswith(".template"):
        filename += ".template"
    
    # Save the output data to the file
    with open(filename, 'w') as file:
        file.write(output_data)


def remove_asterisk_and_blank_lines(input_file, output_file, keep_values=True):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    with open(output_file, 'w') as file:
        for line in lines:
            stripped_line = line.strip()
            if stripped_line and not stripped_line.startswith('*'):
                if '=' in stripped_line:
                    left_part, right_part = stripped_line.split('=', 1)
                    left_part = left_part.strip()
                    if keep_values:
                        right_part = right_part.strip()
                        file.write(f"{left_part} = {right_part}\n")
                    else:
                        file.write(f"{left_part} =\n")
                else:
                    file.write(line + '\n')  # Write the line as is if there's no '='


def strip_value_from_line(input_file, output_file):
    """Prepare IN.DAT for file

    :param input_file: _description_
    :type input_file: _type_
    :param output_file: _description_
    :type output_file: _type_
    """
    with open(input_file, 'r') as file:
        lines = file.readlines()

    with open(output_file, 'w') as file:
        for line in lines:
            stripped_line = line.strip()
            if '=' in stripped_line:
                left_part = stripped_line.split('=')[0].strip()
                # Check for exceptions
                if left_part in ('ixc', 'icc'):
                    file.write(line)  # Write the line as is for ixc and icc
                else:
                    description = stripped_line.split('=')[1].strip()  # Keep the description part
                    file.write(f"{left_part} =\n")  # Write left part with '=' and no value
            else:
                file.write(line)  # Write the line as is if there's no '='

def remove_after_asterisk(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    with open(output_file, 'w') as file:
        for line in lines:
            if '*' in line:
                line = line.split('*')[0].strip() + '\n'
            file.write(line)

def write_list_to_json(data_list, filename):
    """
    Write a list to a JSON file.

    :param data_list: List of data to write to the JSON file.
    :type data_list: list
    :param filename: Name of the JSON file to save the data to.
    :type filename: str
    :return: None
    """
    with open(filename, 'w') as json_file:
        json.dump(data_list, json_file, indent=4)

def wipe_data(d, output_file=None):
    """
    Recursively sets all non-dictionary and non-list values in a dictionary or list to None.
    Optionally writes the wiped data to a new JSON file.

    Args:
        d (dict or list): The dictionary or list to be wiped.
        output_file (str, optional): The path to the file where the wiped data will be saved. Defaults to None.

    Returns:
        None: The function modifies the input dictionary or list in place. If output_file is provided, the wiped data is saved to the specified file.
    """
    if isinstance(d, dict):
        for key in d:
            if isinstance(d[key], (dict, list)):
                wipe_data(d[key])
            else:
                d[key] = None
    elif isinstance(d, list):
        for i in range(len(d)):
            if isinstance(d[i], (dict, list)):
                wipe_data(d[i])
            else:
                d[i] = None

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(d, f, indent=4)
def main():

    # Specify the paths to the dictionary files
    iteration_var_file_path = (
        "uq_tools/variable_categorisation/iteration_variables.json"
    )
    obsolete_vars = "uq_tools/variable_categorisation/obsolete_vars.json"
    output_file = "examples/eu-demo/demo_ixc_ineqs_variables.json"
    categories_filename = "uq_tools/variable_categorisation/variable_categories.json"
    fvalue_mapping_filename = "uq_tools/variable_categorisation/constraint_relationships.json"
    file_path = "examples/eu-demo/inputs/demo_newbaseline_study.txt"

    # Load both JSON files
    categories = load_json("uq_tools/variable_categorisation/variable_categories.json")
    descriptions = load_json("uq_tools/variable_descriptions.json")
    obsolete_vars = load_json(obsolete_vars)
    iteration_vars = load_json(iteration_var_file_path)
    updated_categories = update_descriptions(categories, descriptions)
    updated_descriptions = update_variable_names_in_dict(descriptions, obsolete_vars)
    iteration_vars = update_variable_names_in_dict(iteration_vars, obsolete_vars)
    # Update filenames from obsolete variables
    save_json(
        iteration_vars, iteration_var_file_path
    )
    save_json(
        updated_categories, categories_filename
    )
    save_json(updated_descriptions,"uq_tools/variable_descriptions.json")
    
    descriptions = update_variable_names_in_dict(descriptions, obsolete_vars)
    # Load the dictionaries from the specified paths
    ixc_relationship_dict = load_json(fvalue_mapping_filename)
    
        # Update the original dictionary using obsolete variables
    updated_ixc_relationship_dict = update_variable_names_in_dict(ixc_relationship_dict, obsolete_vars)
    with open(file_path, "r") as file:
        data = file.read()

    # Swap out fvalues for their associated parameters
    ixc_vars, other_vars = process_variable_data(
        data, itv_path=iteration_var_file_path, output_file=output_file
    )
    merged_dict = {**ixc_vars, **other_vars}
    merged_dict= ixc_vars
    # Generate the new list of names
    variable_list_fswapped = swap_f_parameters(ixc_relationship_dict,merged_dict)
    # Categorize variables and subcategorize by source
    categorized_variables = categorize_variables(
        variable_list_fswapped,
        descriptions,
        categories_filename,
    )

    catvar_md= convert_to_markdown_table(categorized_variables)
    print(catvar_md)
    #print(json.dumps(categorized_variables, indent=4))


if __name__ == "__main__":
    main()
