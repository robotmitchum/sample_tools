# coding:utf-8
"""
    :module: jsonFile.py
    :description: Read/write json files on disk
    :author: Michel 'Mitch' Pecqueur
    :date: 2017.02
"""

import json
from pathlib import Path


def write_json(data, filepath, separators=(',', ':'), sort_keys=True, indent=None):
    """
    Write given data to disk as a json file
    Create necessary directory structure if non-existent
    :param int or None indent: Indent value
    :param bool sort_keys: Ordered keys
    :param tuple(str,str) separators:
    :param any data: Python data structure
    :param str or Path filepath: File path
    :return: If no file path supplied return json string
    :rtype: bool or str
    """
    p = Path(filepath)
    if p.is_dir():
        print('Cannot overwrite an existing directory.')
        return False
    if not p.parent.exists():
        p.parent.mkdir()
    json_data = json.dumps(data, separators=separators, indent=indent, sort_keys=sort_keys)
    if not filepath:
        return data
    else:
        with open(filepath, 'w') as f:
            f.write(json_data)
        return True


def read_json(filepath):
    """
    Read json file and return retrieved data
    :param str or Path filepath: File path
    :return: Python data structure
    :rtype: any
    """
    p = Path(filepath)
    if not p.is_file():
        print(f'File does not exist. {filepath}')
        return None
    with open(filepath, 'r') as f:
        json_data = f.read()
    return json.loads(json_data)
