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

import qdarkstyle
from PyQt5 import QtWidgets, QtCore, QtGui, Qt

from __init__ import __version__  # noqa

# from tools.simple_logger import SimpleLogger

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
    def __init__(self):
        super().__init__()
        self.setObjectName('sample_tools_ui')
        self.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.setWindowTitle(f'Sample Tools v{__version__}')

        self.running = {}

        self.current_dir = Path(__file__).parent
        self.tools_path = self.current_dir / 'tools'
        sys.path.append(resource_path(self.tools_path, as_str=True))

        self.tools = {
            'SMP2ds': 'smp2ds_UI.pyw',
            'Split Audio Tool': 'split_tool_UI.pyw',
            'Rename Sample Tool': 'rename_tool_UI.pyw',
            'Loop Tool': 'loop_tool_UI.pyw',
            'Stereo Tool': 'st_tool_UI.pyw',
            'Upsample Tool': 'upsample_tool_UI.pyw',
        }

        # self.tool_modules = {k: eval(v.split('.')[0]) for k, v in self.tools.items()}
        self.tool_modules = {}
        self.import_tools()

        self.icons_path = 'tools/UI/icons/'

        self.icons = {
            'SMP2ds': 'smp2ds_64.png',
            'Split Audio Tool': 'split_tool_64.png',
            'Rename Sample Tool': 'rename_tool_64.png',
            'Loop Tool': 'loop_tool_64.png',
            'Stereo Tool': 'st_tool_64.png',
            'Upsample Tool': 'upsample_tool_64.png',
        }

        self.status_tips = {
            'SMP2ds': 'Create Decent Sampler presets from samples',
            'Split Audio Tool': 'Split and trim audio file(s) by detecting silences',
            'Rename Sample Tool': "Rename audio files and update their 'smpl' chunk/metadata",
            'Loop Tool': 'Detect loop points or modify audio files to make them loop',
            'Stereo Tool': 'Apply pseudo-stereo/stereo imaging effect to audio file(s)',
            'Upsample Tool': 'Up-sample audio file(s) using spectral band replication',
        }

        self.setupUi()

    def import_tools(self):
        for name in self.tools:
            module_name = self.tools[name].split('.')[0]
            self.tool_modules[name] = importlib.import_module(module_name)

    def setupUi(self):
        centralwidget = QtWidgets.QWidget(self)
        centralwidget.setObjectName('centralwidget')
        lyt = QtWidgets.QHBoxLayout(centralwidget)
        lyt.setObjectName('tool_btn_lyt')

        self.setCentralWidget(centralwidget)

        status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(status_bar)

        for tool, cmd in self.tools.items():
            img_file = resource_path(Path(self.icons_path).joinpath(self.icons[tool]))
            btn = IconButton(image=img_file, parent=self)

            name = Path(cmd).stem.removesuffix('_UI')
            btn.setObjectName(name)
            btn.setToolTip(tool)
            btn.setStatusTip(self.status_tips[tool])
            btn.clicked.connect(partial(self.launch_tool, name=tool))
            lyt.addWidget(btn)

        img_file = resource_path(Path(self.icons_path).joinpath('quit_64.png'))
        close_btn = IconButton(image=img_file, parent=self)

        close_btn.clicked.connect(self.close)
        close_btn.setToolTip('Quit')
        close_btn.setStatusTip('Close all tools and quit')

        lyt.addWidget(close_btn)

        self.setLayout(lyt)

        app_icon = QtGui.QIcon()
        img_file = resource_path(Path(self.icons_path).joinpath('sample_tools_64.png'))
        app_icon.addFile(img_file, QtCore.QSize(64, 64))
        self.setWindowIcon(app_icon)

        self.setFixedSize(576, 112)

    def launch_tool(self, name):
        """
        Launch given tool if it is not already running
        :param str name:
        :return:
        """
        # QtWidgets.QMainWindow()

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

    def closeEvent(self, event):
        confirm_dlg = QtWidgets.QMessageBox.question(self, 'Confirmation', 'Are you sure you want to quit?',
                                                     Qt.QMessageBox.Yes | Qt.QMessageBox.No, Qt.QMessageBox.No)
        if confirm_dlg == Qt.QMessageBox.Yes:
            print(f'{self.objectName()} closed')
            event.accept()
        else:
            event.ignore()


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
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))

    font = app.font()
    font.setPointSize(11)
    app.setFont(font)

    screen_geo = app.primaryScreen().geometry()

    try:
        window = SampleToolsUi()
        # Move window up in the screen
        x = screen_geo.x() + (screen_geo.width() - window.width()) // 2
        y = screen_geo.y() + int(screen_geo.height() * .1 - window.height() / 2)
        window.move(x, y)
        window.show()
        sys.exit(app.exec_())

    except Exception as e:
        print(e)
        # logger.log_exception(e)
