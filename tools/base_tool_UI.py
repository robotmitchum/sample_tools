# coding:utf-8
"""
    :module: base_tool_UI.py
    :description: Base class for audio batch tool
    :author: Michel 'Mitch' Pecqueur
    :date: 2024.06

    NOTE : Under windows, when experiencing pauses with video players in a web browser (YT for example)
    or any other interference with any software currently running
    replace the portaudio dll used by sounddevice found in
    .../Lib/site-packages/_sounddevice_data/portaudio-binaries
    by a non-asio dll from this repository :
    https://github.com/spatialaudio/portaudio-binaries

    Sound device is used to preview sounds by double-clicking on the item
"""

import ctypes
import os
import platform
import sys
import threading
from pathlib import Path
from typing import cast

import qdarkstyle
import soundfile as sf
from PyQt5 import QtWidgets, QtGui, Qt, QtCore

from audio_player import AudioPlayer
from common_ui_utils import FilePathLabel, replace_widget, resource_path
from common_ui_utils import Worker, Node, KeyPressHandler, sample_to_name
from sample_utils import Sample
from waveform_widgets import WaveformDialog, LoopPointDialog

from common_prefs_utils import get_settings, set_settings, read_settings, write_settings


class BaseToolUi(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setupUi(self)
        self.setAttribute(Qt.Qt.WA_DeleteOnClose)

        self.output_path_l = replace_widget(self.output_path_l, FilePathLabel(parent=self))
        self.output_path_l = cast(FilePathLabel, self.output_path_l)  # For auto-completion

        self.options = Node()

        self.player = AudioPlayer()
        self.temp_audio = Node()
        self.playback_thread = None

        self.threadpool = QtCore.QThreadPool(parent=self)
        self.worker = None

        self.current_dir = Path(__file__).parent
        self.base_dir = self.current_dir.parent

        self.file_types = ['.wav', '.flac', '.aif']
        self.file_types.extend([ext.upper() for ext in self.file_types])  # Also add uppercase variant

        self.output_path_l.setText('')
        self.last_file = ''

        self.setup_menu_bar()
        self.default_settings = Node()
        self.settings_ext = self.objectName().removesuffix('_mw').replace('_', '')
        self.settings_path = None
        self.set_settings_path()

        self.setup_connections()

        self.suffix_le.setText('')
        self.progress_pb.setTextVisible(True)

        # Set a mono font to the list widget to simplify white spaces handling
        font_path = resource_path(self.current_dir / 'RobotoMono-Medium.ttf')
        font_id = QtGui.QFontDatabase.addApplicationFont(font_path)
        font_family = QtGui.QFontDatabase.applicationFontFamilies(font_id)[0]

        custom_font = Qt.QFont(font_family, 11)
        self.files_lw.setFont(custom_font)
        self.files_lw.setUniformItemSizes(True)

        custom_font = Qt.QFont(font_family)
        self.progress_pb.setFont(custom_font)

    def setup_connections(self):
        # Files widgets
        self.set_files_tb.clicked.connect(self.browse_files)
        self.files_lw.setContextMenuPolicy(3)
        self.files_lw.customContextMenuRequested.connect(self.files_lw_ctx)
        self.files_lw.doubleClicked.connect(self.play_lw_item)

        # Add Suffix widget
        try:
            self.add_suffix_cb.stateChanged.connect(lambda state: self.suffix_le.setEnabled(state == 2))
        except:
            pass

        # Custom events
        self.files_lw.keyPressEvent = self.key_lw_items_event

        self.files_lw.setAcceptDrops(True)
        self.files_lw.dragEnterEvent = self.drag_enter_event
        self.files_lw.dragMoveEvent = self.drag_move_event
        self.files_lw.dropEvent = self.lw_drop_event

        # Settings
        self.set_settings_path()
        self.load_settings_a.triggered.connect(self.load_settings)
        self.save_settings_a.triggered.connect(self.save_settings)
        self.restore_defaults_a.triggered.connect(self.restore_defaults)
        # self.get_defaults()

        self.disable_focus()

    def get_options(self):
        self.options.no_overwriting = self.no_overwriting_cb.isChecked()

        # Get format settings
        self.options.bit_depth = self.bitdepth_cmb.currentText()
        self.options.ext = self.format_cmb.currentText()

        if self.add_suffix_cb.isChecked():
            self.options.suffix = self.suffix_le.text() or ''
        else:
            self.options.suffix = ''

    def as_worker(self, cmd):
        if self.worker:
            if self.worker.running:
                return False
        self.worker = Worker(cmd)
        self.threadpool.start(self.worker)

    def browse_files(self):
        self.refresh_lw_items()
        if not self.last_file:
            items = self.files_lw.selectedItems() or self.get_lw_items()
            items = [s.data(Qt.Qt.UserRole) for s in items]
            if items:
                self.last_file = items[-1]

        if self.last_file:
            startdir = str(Path(self.last_file).parent)
        else:
            startdir = os.getcwd()

        fmts = [f'*{fmt}' for fmt in self.file_types]
        fltr = 'Audio Files ({});;All Files (*)'.format(' '.join(fmts))
        new_files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select audio files", startdir, fltr)

        if new_files:
            files = self.get_lw_items()
            files.extend(new_files)
            self.add_lw_items(files)

    def files_lw_ctx(self):
        menu = QtWidgets.QMenu(self)
        names = [menu.addAction(item) for item in
                 ['Display Waveform\tW', 'Display Loop Point\tL', 'Remove item(s) from list\tDel', 'Clear list']]
        menu.insertSeparator(menu.actions()[2])
        cmds = [self.waveform_dlg, self.loop_point_dlg, self.del_lw_items, self.files_lw.clear]
        action = menu.exec_(QtGui.QCursor.pos())
        for name, cmd in zip(names, cmds):
            if action == name:
                cmd()

    def waveform_dlg(self):
        items = self.get_sel_lw_items()
        if items:
            item = items[0]
            data, sr = sf.read(items[0])
            info = Sample(item)
            dialog = WaveformDialog(data=data, loop_start=info.loopStart, loop_end=info.loopEnd, title=info.name,
                                    cues=info.cues, parent=self)
            dialog.exec_()

    def loop_point_dlg(self):
        items = self.get_sel_lw_items()
        if items:
            item = items[0]
            info = Sample(item)
            if info.loopStart is not None:
                data, sr = sf.read(items[0])
                dialog = LoopPointDialog(data=data, loop_start=info.loopStart, loop_end=info.loopEnd, title=info.name,
                                         disp_len=200, parent=self)
                dialog.exec_()

    def get_lw_items(self):
        return [self.files_lw.item(i).data(Qt.Qt.UserRole) for i in range(self.files_lw.count())]

    def get_sel_lw_items(self):
        return [item.data(Qt.Qt.UserRole) for item in self.files_lw.selectedItems()]

    def del_lw_items(self):
        for item in self.files_lw.selectedItems():
            self.files_lw.takeItem(self.files_lw.row(item))

    def add_lw_items(self, files):
        files = [os.path.normpath(f) for f in files]
        files = list(dict.fromkeys(files))
        names = [sample_to_name(f) for f in files]

        self.files_lw.clear()
        self.files_lw.addItems(names)

        for i, file_path in enumerate(files):
            self.files_lw.item(i).setData(Qt.Qt.UserRole, file_path)

        if files:
            self.last_file = files[-1]

    def refresh_lw_items(self):
        lw_items = [self.files_lw.item(i) for i in range(self.files_lw.count())]
        for item in lw_items:
            f = item.data(Qt.Qt.UserRole)
            if Path(f).is_file():
                item.setText(sample_to_name(f))
            else:
                self.files_lw.takeItem(self.files_lw.row(item))
        self.files_lw.update()

    def play_lw_item(self, *args):
        self.player.stop()
        audio_file = args[0].data(Qt.Qt.UserRole)
        if os.path.isfile(audio_file):
            data, sr = sf.read(audio_file)
            self.playback_thread = threading.Thread(target=self.player.play,
                                                    args=(data, sr, None, None, self.progress_pb.setFormat),
                                                    daemon=True)
            self.playback_thread.start()

    def key_lw_items_event(self, event):
        if event.key() == Qt.Qt.Key_Delete:
            items = self.files_lw.selectedItems()
            for item in items:
                self.files_lw.takeItem(self.files_lw.row(item))

        if event.key() == Qt.Qt.Key_W:
            self.waveform_dlg()
        if event.key() == Qt.Qt.Key_L:
            self.loop_point_dlg()

        if event.key() == Qt.Qt.Key_Down:
            mx = self.files_lw.count() - 1
            sel_indices = [a.row() + 1 if a.row() < mx else mx for a in self.files_lw.selectedIndexes()]
            self.files_lw.clearSelection()
            for idx in sel_indices:
                self.files_lw.item(idx).setSelected(True)
        elif event.key() == Qt.Qt.Key_Up:
            sel_indices = [a.row() - 1 if a.row() > 0 else 0 for a in self.files_lw.selectedIndexes()]
            self.files_lw.clearSelection()
            for idx in sel_indices:
                self.files_lw.item(idx).setSelected(True)

        elif event.modifiers() & Qt.Qt.ControlModifier:
            if event.key() == Qt.Qt.Key_A:
                self.files_lw.selectAll()
            elif event.key() == Qt.Qt.Key_I:
                items = self.files_lw.selectedItems()
                self.files_lw.selectAll()
                for item in items:
                    item.setSelected(False)
        else:
            super().keyPressEvent(event)

    def play_stop_toggle(self):
        if self.player.is_playing.is_set():
            self.player.stop(msg=self.progress_pb.setFormat)
        else:
            items = self.files_lw.selectedItems()
            if items:
                audio_file = items[0].data(Qt.Qt.UserRole)
                if os.path.isfile(audio_file):
                    data, sr = sf.read(audio_file)
                    info = Sample(audio_file)
                    self.playback_thread = threading.Thread(target=self.player.play,
                                                            args=(data, sr, info.loopStart, info.loopEnd,
                                                                  self.progress_pb.setFormat), daemon=True)
                    self.playback_thread.start()

    @staticmethod
    def drag_enter_event(event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    @staticmethod
    def drag_move_event(event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def lw_drop_event(self, event):
        if event.mimeData().hasUrls():
            self.refresh_lw_items()
            items = event.mimeData().urls()
            items = [Path(item.toLocalFile()) for item in items]

            files = self.get_lw_items()
            files.extend([item for item in items if Path(item).suffix in self.file_types])

            dirs = [item for item in items if item.is_dir()]
            for d in dirs:
                for ext in self.file_types:
                    files.extend(Path(d).glob(f'*{ext}'))

            self.add_lw_items(files)
        else:
            event.ignore()

    def closeEvent(self, event):
        print(f'{self.objectName()} closed')
        self.removeEventFilter(self)
        self.player.stop()
        event.accept()

    def disable_focus(self):
        for widget in self.findChildren(QtWidgets.QWidget):
            if type(widget) in [QtWidgets.QPushButton, QtWidgets.QCheckBox, QtWidgets.QComboBox]:
                widget.setFocusPolicy(Qt.Qt.NoFocus)

    def run(self):
        key_press_handler = KeyPressHandler(self)
        self.centralWidget().installEventFilter(key_press_handler)

        # Center on screen not on its parent
        parent = self.parent()
        if parent:
            screen = self.parent().screen()
        else:
            screen = self.screen()

        screen_geo = screen.geometry()
        x = screen_geo.x() + (screen_geo.width() - self.width()) // 2
        y = screen_geo.y() + (screen_geo.height() - self.height()) // 2
        self.move(x, y)

        self.show()
        return self

    def setup_menu_bar(self):
        self.menu_bar = QtWidgets.QMenuBar(self)

        self.settings_menu = QtWidgets.QMenu(self.menu_bar)
        self.settings_menu.setTitle('Settings')
        self.setMenuBar(self.menu_bar)

        self.save_settings_a = QtWidgets.QAction(self)
        self.save_settings_a.setText('Save settings')
        self.load_settings_a = QtWidgets.QAction(self)
        self.load_settings_a.setText('Load settings')
        self.restore_defaults_a = QtWidgets.QAction(self)
        self.restore_defaults_a.setText('Restore defaults')

        self.settings_menu.addAction(self.load_settings_a)
        self.settings_menu.addAction(self.save_settings_a)
        self.settings_menu.addSeparator()
        self.settings_menu.addAction(self.restore_defaults_a)
        self.menu_bar.addAction(self.settings_menu.menuAction())

    def set_settings_path(self):
        self.settings_path = self.current_dir / f'settings.{self.settings_ext}'

    def load_settings(self):
        p = Path(self.settings_path)
        if p.suffix == f'.{self.settings_ext}':
            p = p.parent
        read_settings(widget=self, filepath=None, startdir=p, ext=self.settings_ext)

    def save_settings(self):
        write_settings(widget=self, filepath=None, startdir=self.settings_path, ext=self.settings_ext)

    def get_defaults(self):
        get_settings(self, self.default_settings)

    def restore_defaults(self):
        set_settings(widget=self, node=self.default_settings)


def launch(mw, app_id=''):
    """
    Launch UI
    To be called under top-level code environment

    Example:
    if __name__ == "__main__":
        launch(MyToolUI,app_id='mitch.mytool.1.00')

    :param Class mw: Main window class to launch
    :param str app_id: Unique app identifier
    :return:
    """
    if platform.system() == 'Windows':
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
    font = app.font()
    font.setPointSize(11)
    app.setFont(font)

    window = mw()
    key_press_handler = KeyPressHandler(window)
    app.installEventFilter(key_press_handler)
    window.show()
    sys.exit(app.exec_())
