# coding:utf-8
"""
    :module: waveform_widgets.py
    :description: Dialog to display waveform and  loop point
    :author: Michel 'Mitch' Pecqueur
    :date: 2024.08
"""
import sys

import numpy as np
from PyQt5 import QtWidgets, QtGui, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter


class WaveformDialog(QtWidgets.QDialog):
    """
    Display an audio waveform with optional loop points
    """

    def __init__(self, data=None, loop_start=None, loop_end=None, cues=None, title=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Waveform')

        pos = QtGui.QCursor.pos()
        w, h = 800, 400
        self.setGeometry(pos.x() - w // 2, pos.y() - h // 2, w, h)

        self.layout = QtWidgets.QVBoxLayout(self)

        self.info_dialog_bb = QtWidgets.QDialogButtonBox(self)
        self.info_dialog_bb.setOrientation(Qt.Qt.Horizontal)
        self.info_dialog_bb.setStandardButtons(Qt.QDialogButtonBox.Close)
        self.info_dialog_bb.setCenterButtons(True)

        close_pb = self.info_dialog_bb.button(Qt.QDialogButtonBox.Close)
        close_pb.setDefault(True)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.info_dialog_bb)
        self.setLayout(self.layout)

        close_pb.clicked.connect(self.close)
        self.info_dialog_bb.rejected.connect(self.close)

        self.plot(data, loop_start, loop_end, cues, title)

    def plot(self, data=None, loop_start=None, loop_end=None, cues=None, title=None):
        subplt = self.figure.add_subplot(111)
        subplt.clear()
        subplt.set_ylim(-1, 1)

        # "Dark Theme"
        bg_color = '#19232d'
        mid_color = '#455364'

        self.figure.patch.set_facecolor(bg_color)
        subplt.set_facecolor(bg_color)
        subplt.tick_params(axis='x', colors=mid_color)
        subplt.tick_params(axis='y', colors=mid_color)
        subplt.spines['bottom'].set_color(mid_color)
        subplt.spines['top'].set_color(mid_color)
        subplt.spines['left'].set_color(mid_color)
        subplt.spines['right'].set_color(mid_color)
        subplt.title.set_color('#FFFFFF')
        subplt.xaxis.label.set_color(mid_color)
        subplt.yaxis.label.set_color(mid_color)

        if data is None:
            # Generate some example data
            t = np.linspace(0, 1, 1000)
            data = np.sin(2 * np.pi * 10 * t)
            loop_start = 700
            loop_end = 800

        subplt.set_xlim(0, len(data))

        if data.ndim > 1:
            data = np.mean(data, axis=-1)

        if title:
            subplt.set_title(title)

        # Custom formatter for y-axis ticks to display dB
        def db_formatter(x, pos):
            return f'{20 * np.log10(np.abs(x) + 1e-12):.1f} dB'

        def percent_formatter(x, pos):
            return f'{x / (len(data) - 1):.1f}'

        subplt.xaxis.set_major_formatter(FuncFormatter(percent_formatter))
        subplt.yaxis.set_major_formatter(FuncFormatter(db_formatter))

        subplt.plot(data, label='Audio', color='#4080A0')
        subplt.axhline(y=0, xmin=0, xmax=len(data) - 1, color='#606060')
        if loop_start is not None:
            subplt.axvline(x=loop_start, ymin=-1, ymax=1, label='Loop Start', color='#00FF80')
        if loop_start is not None:
            subplt.axvline(x=loop_end, ymin=-1, ymax=1, label='Loop End', color='#FF0080')
        if cues is not None:
            for c, cue in enumerate(cues):
                subplt.axvline(x=cue, ymin=-1, ymax=1, label=f'Cue {c}', color='#8080FF')

        self.canvas.draw()

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        self.close()


class LoopPointDialog(QtWidgets.QDialog):
    """
    Display overlaid audio around loop start and loop end providing info about the quality of a loop
    """

    def __init__(self, data=None, loop_start=None, loop_end=None, title=None, parent=None, disp_len=None):
        super().__init__(parent)
        self.setWindowTitle('Loop Point')

        pos = QtGui.QCursor.pos()
        w, h = 800, 400
        self.setGeometry(pos.x() - w // 2, pos.y() - h // 2, w, h)

        self.layout = QtWidgets.QVBoxLayout(self)

        self.info_dialog_bb = QtWidgets.QDialogButtonBox(self)
        self.info_dialog_bb.setOrientation(Qt.Qt.Horizontal)
        self.info_dialog_bb.setStandardButtons(Qt.QDialogButtonBox.Close)
        self.info_dialog_bb.setCenterButtons(True)

        close_pb = self.info_dialog_bb.button(Qt.QDialogButtonBox.Close)
        close_pb.setDefault(True)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.info_dialog_bb)
        self.setLayout(self.layout)

        close_pb.clicked.connect(self.close)
        self.info_dialog_bb.rejected.connect(self.close)

        self.plot(data, loop_start, loop_end, title, disp_len)

    def plot(self, data=None, loop_start=None, loop_end=None, title=None, disp_len=None):
        subplt = self.figure.add_subplot(111)
        subplt.clear()

        disp_len = disp_len or 200

        # "Dark Theme"
        bg_color = '#19232d'
        mid_color = '#455364'

        self.figure.patch.set_facecolor(bg_color)
        subplt.set_facecolor(bg_color)
        subplt.tick_params(axis='x', colors=mid_color)
        subplt.tick_params(axis='y', colors=mid_color)
        subplt.spines['bottom'].set_color(mid_color)
        subplt.spines['top'].set_color(mid_color)
        subplt.spines['left'].set_color(mid_color)
        subplt.spines['right'].set_color(mid_color)
        subplt.title.set_color('#FFFFFF')
        subplt.xaxis.label.set_color(mid_color)
        subplt.yaxis.label.set_color(mid_color)

        if data is None:
            # Generate some example data
            t = np.linspace(0, 1, 1000)
            data = np.sin(2 * np.pi * 10 * t)
            loop_start = 750
            loop_end = 849

        subplt.set_xlim(0, len(data))

        # - Prepare data -
        if data.ndim > 1:
            data = np.mean(data, axis=-1)

        loop_len = loop_end - loop_start + 1
        disp_len = min(disp_len, loop_len)

        start_data = data[max(loop_start - disp_len // 2, 0):loop_start + disp_len // 2]
        pad = (disp_len // 2) * 2 - len(start_data)
        if pad:
            start_data = np.pad(start_data, pad_width=(pad, 0), mode='constant', constant_values=(0, 0))

        end_data = data[loop_end + 1 - disp_len // 2:min(loop_end + 1 + disp_len // 2, len(data))]
        pad = (disp_len // 2) * 2 - len(end_data)
        if pad:
            end_data = np.pad(end_data, pad_width=(0, pad), mode='constant', constant_values=(0, 0))
        # ---

        if title:
            subplt.set_title(title)

        def x_formatter(x, pos):
            return f'{x - disp_len // 2:.0f}'

        subplt.xaxis.set_major_formatter(FuncFormatter(x_formatter))
        subplt.set_xlim(0, disp_len)

        mx = np.max(np.abs(np.append(start_data, end_data))) * 1.05
        subplt.set_ylim(-mx, mx)

        subplt.plot(start_data, label='Loop Start', color='#4080A0')
        subplt.plot(end_data, label='Loop End', color='#A08040')

        subplt.axhline(y=0, xmin=0, xmax=disp_len, color='#606060')
        subplt.axvline(x=disp_len // 2, ymin=-1, ymax=1, label='Loop Point', color='#00FF80')

        self.canvas.draw()

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        self.close()


class TestWidgets(QtWidgets.QMainWindow):
    def __init__(self):
        super(TestWidgets, self).__init__()
        self.setWindowTitle('Audio Waveform Display')
        self.setGeometry(960, 540, 400, 400)
        self.centralWidget = QtWidgets.QWidget(self)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralWidget)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.setCentralWidget(self.centralWidget)

        waveform_pb = QtWidgets.QPushButton('Display Waveform', parent=self)
        waveform_pb.clicked.connect(self.waveform_dialog)
        looppoint_pb = QtWidgets.QPushButton('Display Loop Point', parent=self)
        looppoint_pb.clicked.connect(self.looppoint_dialog)

        self.horizontalLayout.addWidget(waveform_pb)
        self.horizontalLayout.addWidget(looppoint_pb)

    def waveform_dialog(self):
        dialog = WaveformDialog(parent=self)
        dialog.exec_()

    def looppoint_dialog(self):
        dialog = LoopPointDialog(parent=self)
        dialog.exec_()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = TestWidgets()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
