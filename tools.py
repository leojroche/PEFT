# Léo J. Roche (leoroche2@gmail.com)
# 04/12/2025

import os

def find_measurements_infos_file(dirpath):
    """
    Recursively search for the `measurement_infos.json` file containing the measurement information, starting from self.FILEPATH and moving up to the parent directories.

    :param dirpath: Directory to start the search from (string)
    :return: The path of the file if found, else None
    """
    filename = 'measurements_infos.json'
    potential_path = os.path.join(dirpath, filename)
    if os.path.isfile(potential_path):
        return potential_path

    parent_directory = os.path.abspath(os.path.join(dirpath, os.pardir))
    if dirpath == parent_directory:
        return None

    return find_measurements_infos_file(parent_directory)