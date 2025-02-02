# coding:utf-8
"""
    :module: common_ui_utils.py
    :description:
    :author: Michel 'Mitch' Pecqueur
    :date: 2024.08
"""
import os
import re
import sys
import traceback
from pathlib import Path

from PyQt5 import QtCore, Qt, QtWidgets, QtGui

from sample_utils import Sample


def get_user_directory(subdir='Documents'):
    if sys.platform == 'win32':
        import winreg

        folders = {
            'Desktop': 'Desktop',
            'Documents': 'Personal',
            'Downloads': '{374DE290-123F-4565-9164-39C4925E467B}',
            'Music': 'My Music',
            'Pictures': 'My Pictures',
            'Videos': 'My Video',
        }
        folders_lowercase = {k.lower(): v for k, v in folders.items()}

        p = Path(subdir)
        subdir_root = p.parts[0].lower()
        key_name = folders_lowercase.get(subdir_root, 'desktop')

        # This works even if user profile has been moved to some other drive
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders') as key:
            result, _ = winreg.QueryValueEx(key, key_name)
            result = Path(result)

        if subdir_root in folders_lowercase:
            result = result / '/'.join(p.parts[1:])
        else:
            result = result.parent / subdir
    else:
        result = Path.home() / subdir

    return result


def replace_widget(old_widget, new_widget):
    """
    Replace a placeholder widget with another widget (typically a customised version of this widget)
    :param QtWidgets.QWidget old_widget:
    :param QtWidgets.QWidget new_widget:
    """

    attrs = ['objectName', 'parent', 'toolTip']
    values = [getattr(old_widget, attr)() for attr in attrs]

    lyt = old_widget.parent().layout()
    lyt.replaceWidget(old_widget, new_widget)
    old_widget.close()

    set_attrs = [f'set{a[0].upper()}{a[1:]}' for a in attrs]
    for a, v in zip(set_attrs, values):
        getattr(new_widget, a)(v)

    return new_widget


def resource_path(relative_path, as_str=True):
    """
    Get absolute path to resource, works for dev and for PyInstaller
    Modified from :
    https://stackoverflow.com/questions/31836104/pyinstaller-and-onefile-how-to-include-an-image-in-the-exe-file
    :param str or WindowsPath relative_path:
    :param bool as_str: Return result as a string
    :return:
    :rtype: str or Path
    """
    if hasattr(sys, '_MEIPASS'):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path().resolve()
    result = base_path / relative_path
    return (result, str(result))[bool(as_str)]


def resource_path_alt(relative_path, parent_dir=None, as_str=True):
    """
    Alternate version with preliminary local check
    Allow to override resources embedded in PyInstaller executable
    :param str or WindowsPath relative_path:
    :param str or WindowsPath parent_dir: override parent dir for local check
    :param bool as_str:
    :return:
    :rtype: str or WindowsPath
    """
    path = (Path(parent_dir) / Path(relative_path).name, Path(relative_path))[bool(parent_dir is None)]
    if path.exists():
        return (path, str(path))[bool(as_str)]
    else:
        return resource_path(relative_path, as_str=as_str)


# Utility classes

class FilePathLabel(QtWidgets.QLabel):
    def __init__(self, text='', file_mode=False, parent=None):
        super().__init__(text, parent)
        self._full_path = ''
        self._default_path = ''

        self._file_mode = file_mode
        self.display_length = 40
        self.start_dir = ''
        self.current_dir = os.path.dirname(sys.modules['__main__'].__file__)
        self.file_types = ['.wav', '.flac', '.aif']

        # self.setStyleSheet('QLabel{color: #808080}')

        self.setAcceptDrops(True)

    def fullPath(self):
        return self._full_path

    def setFullPath(self, path):
        """
        Set full path while updating display name
        :param path:
        :return:
        """
        if path:
            path = os.path.normpath(path)
            p = Path(path)
            if p.is_relative_to(self.current_dir):
                path = str(p.relative_to(self.current_dir))
            self.start_dir = (path, str(Path(path).parent))[self._file_mode]
        self._full_path = path
        self.setText(self.shorten_path(path))

    def shorten_path(self, path):
        if len(path) > self.display_length:
            return f"...{path[-self.display_length:]}"
        return path

    def browse_path(self):
        if not self.start_dir or not Path(self.start_dir).is_dir():
            self.start_dir = os.getcwd()

        if self._file_mode:
            fmts = [f'*{fmt}' for fmt in self.file_types]
            fltr = 'Audio File ({})'.format(' '.join(fmts))
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select File", self.start_dir, fltr)
        else:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory", self.start_dir)

        if path:
            path = os.path.normpath(path)
            p = Path(path)
            if p.is_relative_to(self.current_dir):
                path = str(p.relative_to(self.current_dir))
            self.setFullPath(path)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            items = event.mimeData().urls()
            items = [item.toLocalFile() for item in items]
            if self._file_mode:
                items = [item for item in items if Path(item).is_file()]
            else:
                items = [item for item in items if Path(item).is_dir()]
            if items:
                path = os.path.normpath(items[0])
                p = Path(path)
                if p.is_relative_to(self.current_dir):
                    path = str(p.relative_to(self.current_dir))
                self.setFullPath(path)
        else:
            event.ignore()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()


class Node:
    """
    Simple class to hold attributes
    """

    def __init__(self):
        pass


class KeyPressHandler(QtCore.QObject):
    def __init__(self, parent=None):
        super().__init__(parent)

    def eventFilter(self, obj, event):
        if event.type() == event.KeyPress:
            focus_wid = QtWidgets.QApplication.focusWidget()
            allow_play = not isinstance(focus_wid, QtWidgets.QLineEdit)
            if event.key() == Qt.Qt.Key_Space and allow_play:
                self.parent().play_stop_toggle()
                return True
            else:
                return False
        # Pass the event on to the base class
        return super().eventFilter(obj, event)


# Context Menu

def add_ctx(widget, values=(), names=None, default_idx=None, trigger=None):
    """
    Add a simple context menu setting provided values to the given widget

    :param QtWidgets.QWidget widget: The widget to which the context menu will be added
    :param list values: A list of values to be added as actions in the context menu
    :param int default_idx:
    :param list or None names: A list of strings or values to be added as action names
    must match values length
    :param QWidget or None trigger: Optional widget triggering the context menu
    typically a QPushButton or QToolButton
    """
    if not names:
        names = list(values)
        if default_idx is not None:
            names[default_idx] = f'{names[default_idx]} (Default)'

    def show_context_menu(event):
        menu = QtWidgets.QMenu(widget)
        for name, value in zip(names, values):
            if value == '---':
                menu.addSeparator()
            else:
                action = QtWidgets.QAction(f'{name}', widget)
                if hasattr(widget, 'setValue'):
                    action.triggered.connect(lambda checked, v=value: widget.setValue(v))
                elif hasattr(widget, 'setFullPath'):
                    action.triggered.connect(lambda checked, v=value: widget.setFullPath(v))
                elif hasattr(widget, 'setText'):
                    action.triggered.connect(lambda checked, v=value: widget.setText(v))
                menu.addAction(action)
        pos = widget.mapToGlobal(widget.contentsRect().bottomLeft())
        menu.setMinimumWidth(widget.width())
        menu.exec_(pos)

    widget.setContextMenuPolicy(3)
    if trigger is None:
        widget.customContextMenuRequested.connect(show_context_menu)
    else:
        trigger.clicked.connect(show_context_menu)


def add_insert_ctx(widget, values=(), names=None):
    """
    Add context menu inserting text in a line edit
    :param QWidget widget:
    :param list values:
    :param lit or None names:
    :return:
    """
    if not names:
        names = values

    def show_context_menu(event):
        menu = QtWidgets.QMenu(widget)
        for name, value in zip(names, values):
            action = QtWidgets.QAction(f"{name}", widget)
            action.triggered.connect(lambda _, v=value: widget.insert(v))
            menu.addAction(action)
        menu.exec_(QtGui.QCursor.pos())

    widget.setContextMenuPolicy(3)
    widget.customContextMenuRequested.connect(show_context_menu)


def popup_menu(content, parent=None):
    """
    Create a popup menu

    :param list content: a list of dictionary containing action names (str) execution commands (python)

    Example :
    [
        { 'type': 'cmds', 'name': 'Hello', 'cmd': 'print("Hello world !")' },
        { 'type': '---' },
        { 'type': 'cmds', 'name': 'foo', 'cmd': 'print("bar")' }
    ]

    type : separator as '---' or command as 'cmds'
    name : Name of the command displayed in popup_menu

    :param QWidget parent:

    """
    menu = QtWidgets.QMenu(parent)
    for item in content:
        if 'type' in list(item.keys()):
            if item['type'] == '---':
                menu.addSeparator()
            elif item['type'] == 'cmds':
                action = menu.addAction(item['name'])
                action.triggered.connect(item['cmd'])
    cur_pos = QtGui.QCursor.pos()
    pos = parent.mapToGlobal(parent.rect().bottomLeft())
    pos.setX(cur_pos.x() - menu.width() // 4)
    menu.exec_(pos)


# Name formatting
def shorten_path(file_path, max_length=30):
    if len(file_path) <= max_length:
        return file_path
    return '...' + file_path[-max_length:]


def sample_to_name(file_path):
    """
    Format name to display in sample list widget
    :param file_path:
    :return:
    """
    info = Sample(file_path)

    path_len = 30
    sn = shorten_path(file_path, path_len)
    spc = ' ' * 2

    try:
        notename = f'({info.noteName})'
        note = (f'{str(info.note) : >3} {notename : <6}', f'{str(None) : <9}')[info.note is None]

        pitch_fraction = info.pitchFraction
        if pitch_fraction is not None:
            if pitch_fraction > 0:
                sign = '+'
            elif pitch_fraction < 0:
                sign = ''
            else:
                sign = ' '
            pitch_fraction = f'{sign}{round(pitch_fraction, 3)}'

        loop = f'{str(info.loopStart) : >7}-{str(info.loopEnd) : <7}'

        name = (
            f'{sn : <{path_len}}{spc}Chn: {info.params.nchannels : <2}{spc}Note: {note} {str(pitch_fraction) : <7}{spc}Loop: {loop}'
            f'{spc}len: {info.params.nframes : >7}')
    except Exception as e:
        traceback.print_exc()
        name = sn

    return name


# String formatting functions


def shorten_str(name, sep='_'):
    """
    Transform a camelCased string or with separators / spaces into a shortened string
    Example : 'thisIs1Example' -> 'TI1E'

    :param str name: String to shorten
    :param str sep: Separator to remove

    :return: Shortened string
    :rtype: str
    """
    res = name.replace(sep, ' ')
    res = re.sub(r'([A-Z])', r' \1', res)  # Insert spaces before caps
    res = re.sub(r'(\d)', r' \1', res)  # Insert spaces before digits
    res = [r[0].upper() for r in res.split()]
    result = ''.join(res)
    return result


def beautify_str(name, sep='_'):
    """
    Transform a camelCased string or a string with separators / spaces into a 'beautified' string
    Example : 'thisIs1Example' -> 'This Is 1 Example'

    :param str name: String to beautify
    :param str sep: Separator to remove

    :return: beautified string
    :rtype: str
    """
    res = name.replace(sep, ' ')
    res = re.sub(r'(?<![A-Z])(?=[A-Z])', ' ', res)  # Insert spaces at caps not preceded by another cap
    res = re.sub(r'(?<!\d)(?=\d)', ' ', res)  # Insert spaces at digits not preceded by another digit
    res = [r[0].upper() + r[1:] for r in res.split()]
    result = ' '.join(res)
    return result
