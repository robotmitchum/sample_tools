# coding:utf-8
"""
    :module: jsonFile.py
    :description: Read/write json files on disk
    :author: Michel 'Mitch' Pecqueur
    :date: 2017.02
"""

import json
import os.path
from collections import OrderedDict


def write_json(data, filepath, separators=(',', ':'), sort_keys=True, indent=None):
    """
    Write given data to disk as a json file
    Create necessary folder structure if non-existent
    :param int or None indent: Indent value
    :param bool sort_keys: Ordered keys
    :param list separators:
    :param any data: Python data structure
    :param str filepath: File path
    :return: If no file path supplied return json string
    :rtype: bool or str
    """
    if os.path.isdir(filepath):
        print('Cannot overwrite an existing folder.')
        return False
    filedir = os.path.dirname(filepath)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    json_data = json.dumps(data, separators=tuple(separators), indent=indent, sort_keys=sort_keys)
    if not filepath:
        return data
    else:
        with open(filepath, 'w') as f:
            f.write(json_data)
        return True


def read_json(filepath, ordered=False):
    """
    Read json file and return retrieved data
    :param bool ordered: Ordered keys
    :param str filepath: File path
    :return: Python data structure
    :rtype: any
    """
    if not os.path.isfile(filepath):
        print(('File does not exist. {}'.format(filepath)))
        return None
    with open(filepath, 'r') as f:
        json_data = f.read()
    if ordered:
        return OrderedDict(json.loads(json_data, object_pairs_hook=OrderedDict))
    return json.loads(json_data)
