# coding:utf-8
"""
    :module: sample_tools_UI.pyw
    :description: Launcher Hub for Sample Tools
    :author: Michel 'Mitch' Pecqueur
    :date: 2024.10
"""

# import atexit
import ctypes
import importlib
import platform
import sys
from functools import partial
from pathlib import Path

from dark_fusion_style import apply_dark_theme

from PyQt5 import QtWidgets, QtCore, QtGui, Qt

from __init__ import __version__  # noqa

from tools.simple_logger import SimpleLogger

if getattr(sys, 'frozen', False):
    import pyi_splash  # noqa

    pyi_splash.close()


def resource_path(relative_path, as_str=True):
    """
    Get absolute path to resource, works for dev and for PyInstaller
    Modified from :
    https://stackoverflow.com/questions/31836104/pyinstaller-and-onefile-how-to-include-an-image-in-the-exe-file
    :param str or WindowsPath relative_path:
    :param bool as_str: Return result as a string
    :return:
    :rtype: str or WindowsPath
    """
    if hasattr(sys, '_MEIPASS'):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path().resolve()
    result = base_path / relative_path
    return (result, str(result))[bool(as_str)]


class SampleToolsUi(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName('sample_tools_ui')
        self.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.confirm_close = False

        self.current_dir = Path(__file__).parent
        self.tools_path = self.current_dir / 'tools'
        self.icons_path = 'tools/UI/icons/'

        self.tool_name = 'Sample Tools'
        self.tool_version = __version__
        self.icon_file = resource_path(Path(self.icons_path).joinpath('sample_tools_64.png'))
        self.setWindowTitle(f'{self.tool_name} v{self.tool_version}')

        self.running = {}

        if self.tools_path.exists():
            print(f'tools sub-directory successfully found: {str(self.tools_path)}')
            sys.path.append(str(self.tools_path))
            # Check append
            if str(self.tools_path) in sys.path:
                print('tools sub-directory successfully added to sys.path')
            else:
                print('tools sub-directory failed to be added to sys.path')
        else:
            print(f'tools sub-directory not found: {str(self.tools_path)}')

        self.tools = {
            'SMP2ds': 'smp2ds_UI.py',
            'DR-DS': 'drds_UI.py',
            'Split Audio Tool': 'split_tool_UI.py',
            'Rename Sample Tool': 'rename_tool_UI.py',
            'Loop Tool': 'loop_tool_UI.py',
            'Stereo Tool': 'st_tool_UI.py',
            'Mutate Tool': 'mutate_tool_UI.py',
            'Upsample Tool': 'upsample_tool_UI.py',
        }

        self.tool_modules = {}
        self.import_tools()

        self.icons = {
            'SMP2ds': 'smp2ds_64.png',
            'DR-DS': 'drds_64.png',
            'Split Audio Tool': 'split_tool_64.png',
            'Rename Sample Tool': 'rename_tool_64.png',
            'Loop Tool': 'loop_tool_64.png',
            'Stereo Tool': 'st_tool_64.png',
            'Mutate Tool': 'mutate_tool_64.png',
            'Upsample Tool': 'upsample_tool_64.png',
        }

        self.status_tips = {
            'SMP2ds': 'Create Decent Sampler presets from samples',
            'DR-DS': 'Create Decent Sampler drum presets from samples',
            'Split Audio Tool': 'Split and trim audio file(s) by detecting silences',
            'Rename Sample Tool': "Rename audio files and update their 'smpl' chunk/metadata",
            'Loop Tool': 'Detect loop points or modify audio files to make them loop',
            'Stereo Tool': 'Apply pseudo-stereo/stereo imaging effect to audio file(s)',
            'Mutate Tool': 'Generate randomized mutants/variants from single samples',
            'Upsample Tool': 'Up-sample audio file(s) using spectral band replication',
        }

        self.setupUi()

    def import_tools(self):
        for name in self.tools:
            module_name = self.tools[name].split('.')[0]
            try:
                self.tool_modules[name] = importlib.import_module(module_name)
                print(f'{module_name}: loaded')
            except Exception as e:
                print(f'Failed to import {module_name}: {e}')

    def setupUi(self):
        self.centralwidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralwidget)
        self.lyt = QtWidgets.QHBoxLayout(self.centralwidget)

        self.status_bar = QtWidgets.QStatusBar(parent=self)
        plt = self.palette()
        self.bgc = plt.base().color().getRgb()[:3]
        self.setStyleSheet(f'QStatusBar{{background-color:rgb{self.bgc};}}')
        self.setStatusBar(self.status_bar)

        for tool, cmd in self.tools.items():
            img_file = resource_path(Path(self.icons_path).joinpath(self.icons[tool]))
            btn = IconButton(image=img_file, parent=self)

            name = Path(cmd).stem.removesuffix('_UI')
            btn.setObjectName(name)
            btn.setToolTip(tool)
            btn.setStatusTip(self.status_tips[tool])
            btn.clicked.connect(partial(self.launch_tool, name=tool))
            self.lyt.addWidget(btn)

        img_file = resource_path(Path(self.icons_path).joinpath('quit_64.png'))
        close_btn = IconButton(image=img_file, parent=self)

        close_btn.clicked.connect(self.close)
        close_btn.setToolTip('Quit')
        close_btn.setStatusTip('Close all tools and quit')

        self.lyt.addWidget(close_btn)

        self.centralwidget.setLayout(self.lyt)

        app_icon = QtGui.QIcon()
        app_icon.addFile(self.icon_file, QtCore.QSize(64, 64))
        self.setWindowIcon(app_icon)

        self.setFixedSize(640, 112)

    def launch_tool(self, name):
        """
        Launch given tool if it is not already running
        :param str name:
        :return:
        """
        if name in self.running:
            print(f'{name} already running')
            try:
                self.running[name].show()
                self.running[name].showNormal()
                self.running[name].raise_()
                self.running[name].activateWindow()
            except Exception as e:
                print(e)
            return

        mod = self.tool_modules[name]
        tool = mod.run(parent=self)
        tool.destroyed.connect(partial(self.running.pop, name))
        self.running[name] = tool
        print(f'{name} launched')

    def run(self):
        # Center on screen not on its parent
        parent = self.parent()
        if parent:
            screen = self.parent().screen()
        else:
            screen = self.screen()

        self.show()

        screen_geo = screen.geometry()
        # Move window up in the screen
        x = screen_geo.x() + (screen_geo.width() - self.width()) // 2
        y = screen_geo.y() + int(screen_geo.height() * .1 - self.height() / 2)
        self.move(x, y)

        return self

    def closeEvent(self, event):
        if self.confirm_close:
            confirm_dlg = QtWidgets.QMessageBox.question(self, 'Confirmation', 'Are you sure you want to quit?',
                                                         Qt.QMessageBox.Yes | Qt.QMessageBox.No, Qt.QMessageBox.No)
            if confirm_dlg == Qt.QMessageBox.Yes:
                print(f'{self.objectName()} closed')
                event.accept()
            else:
                event.ignore()


def run(mw=SampleToolsUi, parent=None):
    window = mw(parent=parent)
    return window.run()


class IconButton(QtWidgets.QPushButton):
    def __init__(self, parent=None, size=64, image=''):
        super().__init__()
        self.setParent(parent)
        self.setGeometry(0, 0, size, size)
        icon = QtGui.QIcon()
        icon.addFile(image)
        self.setIcon(icon)
        self.setIconSize(QtCore.QSize(size, size))
        size_policy = Qt.QSizePolicy(Qt.QSizePolicy.Fixed, Qt.QSizePolicy.Fixed)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        size_policy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(size_policy)
        self.setFlat(True)


# def global_exception_handler(exctype, value, tb):
#     logger.log_exception(value)


# logger = SimpleLogger('sample_tools_log.txt')
# sys.excepthook = global_exception_handler
# atexit.register(lambda: logger.logger.info("Application is shutting down."))

if __name__ == '__main__':
    app_id = f'mitch.sampleTools.{__version__}'

    if platform.system() == 'Windows':
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)

    app = QtWidgets.QApplication(sys.argv)
    apply_dark_theme(app)

    font = app.font()
    font.setPointSize(11)
    app.setFont(font)

    current_screen = app.primaryScreen()
    for screen in QtWidgets.QApplication.screens():
        if screen.geometry().contains(Qt.QCursor.pos()):
            current_screen = screen
            break

    screen_geo = current_screen.geometry()

    try:
        win = run()
        win.confirm_close = True

    except Exception as e:
        print(e)
        # logger.log_exception(e)

    sys.exit(app.exec_())
