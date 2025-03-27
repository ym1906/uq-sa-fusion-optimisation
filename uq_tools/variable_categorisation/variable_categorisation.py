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
    """
    def convert_variable_names_to_num(variable_names):
        """
        Converts the keys of the variable_names dictionary from strings to integers.

        :param variable_names: A dictionary with variable names as keys and index as values in string format.
        :type variable_names: dict
        :return: A new dictionary with the variable names as keys and index as integer values.
        :rtype: dict
        """
        return {var: int(index) for var, index in variable_names.items()}
    
    ixc_pattern = re.compile(r"ixc\s*=\s*(\d+)")
    boundu_pattern = re.compile(r"boundu\((\d+)\)\s*=\s*([^;]*)")
    boundl_pattern = re.compile(r"boundl\((\d+)\)\s*=\s*([^;]*)")
    var_pattern = re.compile(r"(\w+)\s*=\s*([^;]*)")

    ixc_variables = {}
    other_variables = {}

    # Load variable names from JSON
    with open(itv_path, "r") as file:
        variable_names = json.load(file)
    variable_names = convert_variable_names_to_num(variable_names)

    def convert_to_number(value):
        """Convert value to int or float if possible."""
        try:
            return float(value) if "." in value else int(value)
        except ValueError:
            return value.strip()

    # Process each line
    for line in data.strip().split("\n"):
        ixc_match = ixc_pattern.search(line)
        boundu_match = boundu_pattern.search(line)
        boundl_match = boundl_pattern.search(line)
        var_match = var_pattern.match(line)

        if ixc_match:
            # Capture the ixc index (like 2) but we don't capture the value here
            index = int(ixc_match.group(1))
            var_name = next((name for name, num in variable_names.items() if num == index), None)
            if var_name:
                # Initialize ixc variable without a value, to be updated later
                ixc_variables[var_name] = {"index": index, "value": None, "boundu": None, "boundl": None}

        if boundu_match:
            index, value = boundu_match.groups()
            index = int(index)
            value = convert_to_number(value)
            var_name = next((name for name, num in variable_names.items() if num == index), None)
            if var_name:
                # Update the boundu for the correct variable
                if var_name in ixc_variables:
                    ixc_variables[var_name]["boundu"] = value

        if boundl_match:
            index, value = boundl_match.groups()
            index = int(index)
            value = convert_to_number(value)
            var_name = next((name for name, num in variable_names.items() if num == index), None)
            if var_name:
                # Update the boundl for the correct variable
                if var_name in ixc_variables:
                    ixc_variables[var_name]["boundl"] = value

        if var_match and not any((ixc_match, boundu_match, boundl_match)):
            var_name, value = var_match.groups()
            value = convert_to_number(value)
            other_variables[var_name] = {"value": value}  # Add value property

    # Now, link the actual variable value to its ixc reference
    for var_name, var_data in other_variables.items():
        # Check if this variable corresponds to any ixc index
        if var_name in ixc_variables:
            ixc_variables[var_name]["value"] = var_data["value"]

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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
    
    :param categories: Dictionary where categories contain dictionaries of variables.
    :param descriptions: Dictionary mapping filenames to variable descriptions.
    :return: Updated categories data.
    """
    # Iterate through each category
    for category, variables in categories.items():
        if isinstance(variables, dict):  # Ensure it's a variable dictionary
            for var_name, details in variables.items():
                # Search for the variable in descriptions
                for filename, vars_dict in descriptions.items():
                    if var_name in vars_dict:
                        # Update the description
                        details["Description"] = vars_dict[var_name]
                        break  # Stop searching once updated
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

def load_categories(categories_filename):
    """
    Load categories from the provided JSON file and create a category map.
    
    :param categories_filename: Path to the category mapping file.
    :return: category_map and description_map (if available)
    """
    with open(categories_filename, "r") as f:
        categories_data = json.load(f)
    
    # Create the category_map by iterating through categories
    category_map = {}
    description_map = {}  # You may also need to handle descriptions similarly
    
    for category, variables in categories_data.items():
        for var in variables:
            category_map[var] = category

    return category_map, description_map

# def load_categories(filename):
#     with open(filename, 'r') as file:
#         return {entry['Variable']: entry['Description'] for entry in json.load(file)}


def categorize_variables(
    variable_list,
    descriptions,
    categories_filename="/home/graeme/work-notes/uncertainty_quantification/variable_categorisation/variable_categories.json",
):
    """
    Categorizes variables into predefined categories based on their descriptions and a category mapping.
    Now includes bounds (boundl and boundu) and value properties from the variable_list.
    
    :param variable_list: Dictionary where variable names map to their attributes, including bounds and values.
    :param descriptions: Dictionary containing variable descriptions and category mapping.
    :param categories_filename: Path to the JSON file containing the category mapping.
    :return: A dictionary with categorized variables, where each category contains a dictionary of variables.
    """
    category_map, description_map = load_categories(categories_filename)
    # Initialize result with all categories
    result = {
        "Design Variables": {},
        "Control Variables": {},
        "Design Parameters": {},
        "Unknown Category": {},
    }
    
    for var_name, var_data in variable_list.items():
        # Retrieve category from category_map
        category = category_map.get(var_name)
        
        # If category is None, log the variable and categorize as "Unknown Category"
        if category is None:
            category = "Unknown Category"
        
        # Ensure the category exists in the result dictionary
        if category not in result:
            result[category] = {}
        
        # Find description in the descriptions dictionary
        description_found = None
        for filename, vars_dict in descriptions.items():
            if var_name in vars_dict:
                description_found = vars_dict[var_name]
                break
        
        # Default to description_map if not found
        description = description_found if description_found else description_map.get(var_name, "No description available.")
        
        # Assign values to the variable in the result dictionary
        result[category][var_name] = {
            "Index": var_data.get("Index", None),
            "Description": description,
            "boundl": var_data.get("boundl", None),
            "boundu": var_data.get("boundu", None),
            "value": var_data.get("value", None),
        }
    
    # Clean up the result by removing empty categories
    result = {k: v for k, v in result.items() if v}
    
    return result



def convert_to_markdown_table(categorized_variables, include_bounds=True):
    markdown = "# Variable Categorization\n\n"  # Add double newline for proper section spacing
    
    for category, details in categorized_variables.items():
        markdown += f"## {category}\n\n"  # Add double newline after category header
        
        # Add headers for the new columns: bounds and value
        if include_bounds:
            markdown += "| **Variable** | **Description** | **Lower Bound** | **Upper Bound** | **Value** |\n"
            markdown += "|--------------|-----------------|-----------------|-----------------|-----------|\n"
        else:
            markdown += "| **Variable** | **Description** | **Value** |\n"
            markdown += "|--------------|-----------------|-----------|\n"
        
        # Loop through each variable in the category
        for var_name, var_info in details.items():
            # Ensure all required keys are in the variable info, else use 'N/A'
            description = var_info.get("Description", "No description available.")
            boundl = var_info.get("boundl", "N/A")
            boundu = var_info.get("boundu", "N/A")
            value = var_info.get("value", "N/A")
            
            # Add the variable data to the markdown table
            if include_bounds:
                markdown += f"| `{var_name}` | {description} | {boundl} | {boundu} | {value} |\n"
            else:
                markdown += f"| `{var_name}` | {description} | {value} |\n"
        
        # Double newline after each category to ensure spacing between sections
        markdown += "\n\n"
    
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

def create_sample_space(variables, multiplier, output_file, bounds_dict, variables_to_extract=None):
    """
    Create a dictionary of parameters with min, max, and default values for float variables.

    This function iterates through the items in the `variables` dictionary, identifies float variables,
    and calculates their min and max values based on the provided multiplier. The resulting parameters
    are saved to the specified output file in JSON format. Parameters that do not meet the criteria are
    stored in a variable.

    :param variables: Dictionary containing variable data.
    :type variables: dict
    :param multiplier: Multiplier for calculating min and max values as a fraction.
    :type multiplier: float
    :param output_file: Path to the output file where the parameters will be saved in JSON format.
    :type output_file: str
    :param bounds_dict: Dictionary containing bounds for variables.
    :type bounds_dict: dict
    :param variables_to_extract: List of variables to extract from variables. If None, use all variables.
    :type variables_to_extract: list, optional
    :return: Dictionary of removed parameters.
    :rtype: dict
    """
    params = {}
    removed_params = {}

    for key, value in variables.items():
        if variables_to_extract is None or key in variables_to_extract:
            if isinstance(value, float):
                min_value = value * (1 - multiplier)
                max_value = value * (1 + multiplier)
                
                # Check against bounds_dict
                if key in bounds_dict:
                    min_value = max(min_value, bounds_dict[key]["lb"])
                    max_value = min(max_value, bounds_dict[key]["ub"])
                
                if max_value > min_value:
                    params[key] = {
                        "type": "float",
                        "min": min_value,
                        "max": max_value,
                        "default": value
                    }
                else:
                    removed_params[key] = {
                        "type": "float",
                        "min": min_value,
                        "max": max_value,
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

    return removed_params

def filter_variables(variable_list, removed_params):
    """
    Remove any variables from the list if they appear in the removed_params dictionary.

    :param variable_list: List of variables to be filtered.
    :type variable_list: list
    :param removed_params: Dictionary of removed parameters.
    :type removed_params: dict
    :return: Filtered list of variables.
    :rtype: list
    """
    filtered_list = [var for var in variable_list if var not in removed_params]
    return filtered_list

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
                        # Extract the value before any additional text
                        right_part = right_part.split()[0].strip()
                        file.write(f"{left_part} = {right_part}\n")
                    else:
                        file.write(f"{left_part} =\n")
                else:
                    file.write(line + '\n')  # Write the line as is if there's no '='


def swap_d_to_e(input_file, output_file):
    """
    Swap 'D' and 'd' with 'E' in the values of the input file and save to the output file.

    :param input_file: Path to the input file.
    :type input_file: str
    :param output_file: Path to the output file where the modified content will be saved.
    :type output_file: str
    """
    with open(input_file, 'r') as file:
        lines = file.readlines()

    with open(output_file, 'w') as file:
        for line in lines:
            if '=' in line:
                left_part, right_part = line.split('=', 1)
                right_part = right_part.replace('D', 'E').replace('d', 'E')
                file.write(f"{left_part}={right_part}")
            else:
                file.write(line)

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
