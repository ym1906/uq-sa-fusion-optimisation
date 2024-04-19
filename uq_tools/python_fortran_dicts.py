"""Access Python-Fortran dictionaries for Fortran source information.

This ultimately provides Process Python with the ability to access variable
information in the Process Fortran source code.

Process Python can call process.io.python_fortran_dicts.get_dicts() to load
the dicts from the JSON file created and saved at build-time and use them.
"""

import os
import json
import logging

logger = logging.getLogger(__name__)


def get_dicts():
    """Return dicts loaded from the JSON file for use in Python.

    :return: Python-Fortran dicts
    :rtype: dict
    """
    # Construct the absolute path to the uq_tools folder
    current_directory = os.path.dirname(
        __file__
    )  # Get the directory of the current script
    uq_tools_directory = os.path.abspath(
        os.path.join(current_directory, "..", "uq_tools/")
    )

    # Construct the file path to python_fortran_dicts.json
    dicts_filename = os.path.join(uq_tools_directory, "python_fortran_dicts.json")

    try:
        with open(dicts_filename, "r") as dicts_file:
            variable_dict = json.load(dicts_file)
    except FileNotFoundError as error:
        print(f"Error: {error}")
        return {}

    # Return loaded dicts
    try:
        with open(dicts_filename, "r") as dicts_file:
            return json.load(dicts_file)
    except FileNotFoundError as error:
        logger.exception("Can't find the dicts JSON file")
        raise error
