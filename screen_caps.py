# coding:utf-8
"""
    :module: screen_caps.py
    :description: Quick and dirty batch screen capture for all the tools to update the documentation more easily
    :author: Michel 'Mitch' Pecqueur
    :date: 2025.04
"""

import importlib
import os
import sys
from functools import partial
from pathlib import Path

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from dark_fusion_style import apply_dark_theme


class ScreenCap(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Capture UI')
        self.setGeometry(100, 100, 300, 200)

        self.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))

        self.current_dir = Path(__file__).parent
        self.tools_path = self.current_dir / 'tools'

        self.caps_path = self.current_dir / 'screencaps'
        if not self.caps_path.exists():
            os.makedirs(self.caps_path, exist_ok=True)

        if self.tools_path.exists():
            print(f'tools sub-directory successfully found: {str(self.tools_path)}')
            sys.path.append(str(self.tools_path))

        self.scripts = ['sample_tools_UI.py', 'smp2ds_UI.py', 'drds_UI.py', 'split_tool_UI.py', 'rename_tool_UI.py',
                        'loop_tool_UI.py', 'st_tool_UI.py', 'upsample_tool_UI.py']

        self.script_modules = {}

        self.batch_capture()

    def batch_capture(self):
        for script in self.scripts:
            module_name = str(Path(script).stem)
            try:
                self.script_modules[module_name] = importlib.import_module(module_name)
                print(f'{module_name}: loaded')
                mod = self.script_modules[module_name]
                wid = mod.run(parent=self)
                filename = module_name.lower()
                QTimer.singleShot(100, partial(self.capture, wid, filename))
            except Exception as e:
                print(f'Failed to import {module_name}: {e}')

    def capture(self, widget, filename):
        pixmap = widget.grab()
        filepath = self.caps_path / f'{filename.lower()}.png'
        filepath = str(filepath.resolve())
        pixmap.save(filepath, 'PNG')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    apply_dark_theme(app)

    font = app.font()
    font.setPointSize(11)
    app.setFont(font)

    window = ScreenCap()
    window.show()

    sys.exit(app.exec_())
