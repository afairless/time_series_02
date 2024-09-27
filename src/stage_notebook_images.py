#! /usr/bin/env python3

import subprocess
from pathlib import Path


def get_git_root_path() -> Path | None:
    """
    Returns the top-level project directory where the Git repository is defined
    """

    try:
        # Run the git command to get the top-level directory
        git_root = subprocess.check_output(
            ['git', 'rev-parse', '--show-toplevel'], 
            stderr=subprocess.STDOUT)
        git_root_path = Path(git_root.decode('utf-8').strip())
        return git_root_path 

    except subprocess.CalledProcessError as e:
        print('Error while trying to find the Git root:', e.output.decode())
        return None


def identify_code_lines_with_filenames(
    code_lines: list[tuple[int, str]], extensions: list[str]
    ) -> list[tuple[int, str]]:
    """
    Identifies lines of code that contain filenames of particular file types, 
        as defined by their extensions

    code_lines: list of 2-element tuples where each first element contains the 
        line number and each second element contains the line of code
    """

    image_filename_lines = [
        (i, line) for i, line in code_lines
        # include only lines that contain an extension
        if any('.' + ext in line for ext in extensions)
        # file extension should be in a string
        #   this doesn't check whether the extension is in the string, but it 
        #   filters out lines that don't have strings
        and ('"' in line or "'" in line)
        # include only lines that with a variable assignment
        and '=' in line 
        # filter out lines that do not assign a value to a single variable name 
        #   to the left of the '=' sign
        and ' ' not in line.split('=')[0].strip()]

    return image_filename_lines


def main():

    project_root_path = get_git_root_path()
    assert isinstance(project_root_path, Path)
    notebook_py_path = project_root_path / 'src'

    notebook_py_filename = 'statsmodels_sarimax_reference.py'
    notebook_py_filepath = notebook_py_path / notebook_py_filename 

    with open(notebook_py_filepath, 'r') as f:
        code = f.readlines()

    code_lines = [(i, line) for i, line in enumerate(code)]
    image_extensions = ['png', 'jpg', 'jpeg', 'gif', 'svg', 'bmp', 'tiff']

    image_filename_lines = identify_code_lines_with_filenames(
        code_lines, image_extensions)












if __name__ == '__main__':
    main()
