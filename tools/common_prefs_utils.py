# coding:utf-8
"""
    :module: common_prefs_utils.py
    :description: Generic functions to get, set, save and load user widgets configuration
    For custom widgets, define get/set methods using get_prefs/set_prefs attributes
    :author: Michel 'Mitch' Pecqueur
    :date: 2024.12
"""

import os
from pathlib import Path

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog

from jsonFile import read_json, write_json


def get_settings(widget, node):
    """
    Get user widget values from a parent widget
    :param QtWidgets.QWidget widget: Given parent widget, typically a QMainWindow instance
    :param class or None node: Python object used to store settings
    :return: Preferences with widget name as key
    :rtype: dict
    """
    result = {}
    for child in widget.findChildren(QtWidgets.QWidget):
        name = child.objectName()
        if not name or name.startswith('qt_'):
            continue

        match child.__class__.__name__:
            case 'QCheckBox':
                value = child.isChecked()
            case 'QComboBox':
                value = child.currentText()
            case 'QSpinBox' | 'QDoubleSpinBox':
                value = child.value()
            case 'QLineEdit' | 'UrlPathLineEdit':
                value = child.text()
            case 'FilePathLabel':
                p = child.fullPath()
                value = Path(p).as_posix() if p else ''
            case _:
                if hasattr(child, 'get_prefs'):
                    value = child.get_prefs()
                else:
                    value = None

        if value is not None:
            if node:
                setattr(node, name, value)
            result[name] = value

    return result


def set_settings(widget, node):
    """
    Set user widget values
    :param QtWidgets.QWidget widget: Given parent widget, typically a QMainWindow instance
    :param class or None node: Python object used to store attributes
    :return: Preferences with widget name as key
    :rtype: dict
    """
    for child in widget.findChildren(QtWidgets.QWidget):
        name = child.objectName()
        if not name or name.startswith('qt_') or not hasattr(node, name):
            continue

        value = getattr(node, name)
        match child.__class__.__name__:
            case 'QCheckBox':
                child.setChecked(value)
            case 'QComboBox':
                values = [child.itemText(i) for i in range(child.count())]
                if value in values:
                    child.setCurrentText(value)
                else:
                    print(f'{name} QComboxBox skipped : {value} not found in items')
            case 'QSpinBox' | 'QDoubleSpinBox':
                child.setValue(value)
            case 'QLineEdit' | 'UrlPathLineEdit':
                child.setText(value)
            case 'FilePathLabel':
                child.setFullPath(value)
            case _:
                if hasattr(child, 'set_prefs'):
                    child.set_prefs(value)

    return True


def write_settings(widget, filepath=None, startdir=None, ext='json'):
    """
    Write settings to a json file on disk
    :param QtWidgets.QWidget widget:
    :param str or Path or None filepath:
    :param str or Path or None startdir:
    :param str or None ext: Customize extension, ignored when explicitly providing filepath
    :return:
    """
    try:
        if filepath is None:
            startdir = startdir or Path(os.getcwd()) / 'settings.json'

            options = QFileDialog.Options()
            filepath, _ = QFileDialog.getSaveFileName(widget, caption='Save current settings', directory=str(startdir),
                                                      filter=f'{ext} files (*.{ext})', options=options)
        if filepath:
            filepath = Path(filepath)
            prefs = get_settings(widget, None)
            write_json(data=prefs, filepath=str(filepath), indent=2, sort_keys=False)
            return filepath
        else:
            return False
    except Exception as e:
        print(e)
        pass


def read_settings(widget, filepath=None, startdir=None, ext='json'):
    """
    Write settings to a json file on disk
    :param QtWidgets.QWidget widget:
    :param str or Path or None filepath:
    :param str or Path or None startdir:
    :param str or None ext: Customize extension, ignored when explicitly providing filepath
    :return:
    """
    try:
        if filepath is None:
            startdir = startdir or os.getcwd()

            options = QFileDialog.Options()
            filepath, _ = QFileDialog.getOpenFileName(widget, caption='Load settings', directory=str(startdir),
                                                      filter=f'{ext} files (*.{ext})', options=options)
        if filepath:
            filepath = Path(filepath)
            prefs = read_json(filepath=str(filepath))
            node = Node()
            for key, value in prefs.items():
                setattr(node, key, value)
            set_settings(widget, node)
            return filepath
        else:
            return False
    except Exception as e:
        print(e)
        pass


class Node:
    """
    Simple class to hold attributes
    """

    def __init__(self):
        pass
