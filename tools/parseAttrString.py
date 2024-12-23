# coding:utf-8
"""
    :module: parseAttrString.py
    :description: Simple functions to parse attribute values from a string pattern
    and also re-compose a string from attribute values and a formatting pattern
    :author: Michel 'Mitch' Pecqueur
    :date: 2024.07
"""

import re


class Info:
    """
    Simple class to hold attributes
    """

    def __init__(self, info=None):
        """
        :param dict info: dict used to inform attributes
        """
        if isinstance(info, dict):
            for key, value in info.items():
                setattr(self, key, value)


def parse_string(name, pattern, obj=None):
    """
    Parse string using a pattern to extract info
    :param str name: Input string
    :param str pattern: Use curly braces {} to mark attribute names
    Example: '{group}_{note}_{vel}_{seqMode}'
    :param object obj: Optional object
    :return: Attribute name and values as a dictionary
    :rtype: dict
    """
    tokens = re.split(r'{(\w+)}', pattern)
    attrs = tokens[1::2]
    seps = list(filter(None, tokens[0::2]))

    values, splits = [], []
    tmp = name
    for sep in seps:
        splits = tmp.split(sep, 1)
        if len(splits) > 1:
            if splits[0]:
                values.append(splits[0])
            tmp = splits[-1]
        else:
            break
    values.append(tmp)

    result = dict(zip(attrs, values))
    if obj is not None:
        for attr, value in result.items():
            setattr(obj, attr, value)
    return result


def compose_string(info, pattern):
    """
    Compose a string from attribute names/values and a formatting pattern
    :param object or dict info: object or dict holding keys/attributes and their respective values
    :param str pattern: Use curly braces {} to mark attribute names
    Attributes absent from 'info' will be omitted
    :return: Resulting string
    :rtype: str
    """
    tokens = re.split(r'{(\w+)}', pattern)
    keys = tokens[1::2]

    if isinstance(info, dict):
        info = Info(info)

    values = [getattr(info, attr, None) for attr in keys]
    tokens[1::2] = values

    for i, item in enumerate(values):
        if item is None:
            tokens[i * 2], tokens[i * 2 + 1] = '', ''

    tokens = [f'{token:03d}' if isinstance(token, int) else f'{token}' for token in tokens]

    result = ''.join(tokens)
    return result
