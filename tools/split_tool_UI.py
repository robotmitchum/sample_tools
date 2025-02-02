# coding:utf-8
"""
    :module: split_tool_UI.pyw
    :description: Tool to split audio files in into several files based on silence detection
    Additional options such as pitch detection for sampled/virtual instruments
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

import importlib
import traceback
from functools import partial
from pathlib import Path

from PyQt5 import QtGui, QtCore

import split_audio as sa
from UI import split_tool as gui
from audio_player import play_notification
from base_tool_UI import BaseToolUi, launch
from common_ui_utils import add_ctx, get_user_directory, resource_path
from sample_utils import Sample

try:
    import crepe

    has_crepe = True
except Exception as e:
    has_crepe = False
    pass

__version__ = '1.1.0'


class SplitToolUi(gui.Ui_split_tool_mw, BaseToolUi):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle(f'Split Audio Tool v{__version__}')

        app_icon = QtGui.QIcon()
        img_file = resource_path(self.current_dir / 'UI/icons/split_tool_64.png')
        app_icon.addFile(img_file, QtCore.QSize(64, 64))
        self.setWindowIcon(app_icon)

        self.setup_pitch_mode_cmb(use_crepe=has_crepe)

        self.use_pitch_fraction_cb.setChecked(True)

        self.get_defaults()

        self.update_message('Split and trim audio file(s) by detecting silences')

    def setup_connections(self):
        super().setup_connections()

        # Output directory
        default_dir = get_user_directory()
        desktop_dir = get_user_directory('Desktop')
        self.output_path_l.setFullPath(default_dir)
        add_ctx(self.output_path_l, values=['', default_dir, desktop_dir],
                names=['Clear', 'Default directory', 'Desktop'])
        self.set_output_path_tb.clicked.connect(self.output_path_l.browse_path)

        # Base name widget
        add_ctx(self.basename_le, ['', 'Split', 'Sample'])

        self.subdir_cb.stateChanged.connect(lambda state: self.subdir_le.setEnabled(state == 2))
        add_ctx(self.subdir_le, ['', 'Splits', 'Instrument/Samples'])

        # Min duration widget
        add_ctx(self.min_dur_dsb, [.1, .2, .5, 1, 2])

        # Split dB widget
        add_ctx(self.split_db_dsb, [-40, -60, -80, -100, -120])

        # Fade dB widget
        add_ctx(self.fade_db_dsb, [-20, -40, -60, -80, -100])

        # Suffix widget
        self.suffix_mode_cmb.currentTextChanged.connect(lambda state: self.suffix_le.setEnabled(state == 'suffix'))
        add_ctx(self.suffix_le, ['', 'attack release'])

        # Extra suffix widget
        self.extra_suffix_cb.stateChanged.connect(lambda state: self.extra_suffix_le.setEnabled(state))
        add_ctx(self.extra_suffix_le, ['', '_v127', '_v063', '_f', '_p'])

        # Pitch widgets
        self.suffix_mode_cmb.currentTextChanged.connect(
            lambda state: self.pitch_mode_wid.setEnabled(state.startswith('note')))

        # Preview / Process buttons
        # Execute "as worker" to prevent multiple execution
        self.process_pb.clicked.connect(partial(self.as_worker, partial(self.do_process, mode='batch')))
        self.process_sel_pb.clicked.connect(partial(self.as_worker, partial(self.do_process, mode='sel')))

    def setup_pitch_mode_cmb(self, use_crepe=False):
        """
        Modify Pitch mode combo box depending on the presence of 'crepe'
        :param bool use_crepe:
        :return:
        """
        values = ['corr', 'pyin']
        if use_crepe:
            values.append('crepe')

        self.pitch_mode_cmb.clear()
        self.pitch_mode_cmb.addItems(values)

        tooltip = self.pitch_mode_cmb.toolTip()
        lines = tooltip.split('\n')
        tooltip = ''
        for line in lines:
            if 'crepe' not in line or use_crepe:
                tooltip += line + '\n'

        self.pitch_mode_cmb.setToolTip(tooltip[:-1])

    def get_options(self):
        suffix_mode = self.suffix_mode_cmb.currentText()
        self.options.suffix = None

        if suffix_mode == 'suffix':
            self.options.suffix = list(filter(None, self.suffix_le.text().split()))

        self.options.extra_suffix = ('', self.extra_suffix_le.text())[self.extra_suffix_cb.isChecked()]

        self.options.detect_pitch = 0
        if suffix_mode == 'note':
            self.options.detect_pitch = 1
        elif suffix_mode == 'noteName':
            self.options.detect_pitch = 2

        self.options.pitch_mode = self.pitch_mode_cmb.currentText()
        self.options.use_pitch_fraction = self.use_pitch_fraction_cb.isChecked()

        self.options.min_duration = self.min_dur_dsb.value()
        self.options.split_db = self.split_db_dsb.value()
        self.options.fade_db = self.fade_db_dsb.value()
        self.options.write_cue = self.write_cue_cb.isChecked()
        self.options.dc_offset = self.dc_offset_cb.isChecked()
        self.options.dither = self.dither_cb.isChecked()
        self.options.subdir = ('', self.subdir_le.text())[self.subdir_cb.isChecked()] or ''
        self.options.bd_cmb = self.bitdepth_cmb.currentText()
        self.options.ext_cmb = self.format_cmb.currentText()

    def do_process(self, worker, progress_callback, message_callback, mode):
        if mode == 'batch':
            files = self.get_lw_items()
        else:
            files = self.get_sel_lw_items()

        if not files:
            return False

        range_callback = worker.signals.progress_range

        importlib.reload(sa)

        count = len(files)

        # Options
        self.get_options()
        options = vars(self.options)

        # Progress bar init
        range_callback.emit(0, count)
        progress_callback.emit(0)
        message_callback.emit('%p%')

        done = 0
        try:
            for i, f in enumerate(files):
                if worker.is_stopped():
                    return False

                info = Sample(f)
                prm = info.params

                p = Path(f)
                parent = self.output_path_l.fullPath() or p.parent
                stem = p.stem

                if options['subdir']:
                    parent = Path.joinpath(Path(self.output_path_l.fullPath()), options['subdir'])

                basename = self.basename_le.text() or stem

                if options['ext_cmb'] == 'same':
                    ext = p.suffix.strip('.')
                else:
                    ext = options['ext_cmb']

                if options['bd_cmb'] == 'same':
                    bit_depth = prm.sampwidth * 8
                else:
                    bit_depth = int(options['bd_cmb'])

                filepath = Path.joinpath(Path(parent), f'{basename}.{ext}')
                result = sa.split_audio(input_file=f, output_file=str(filepath), bit_depth=bit_depth,
                                        suffix=options['suffix'], extra_suffix=options['extra_suffix'],
                                        use_note=options['detect_pitch'],
                                        pitch_mode=options['pitch_mode'],
                                        use_pitch_fraction=options['use_pitch_fraction'],
                                        min_duration=options['min_duration'],
                                        split_db=options['split_db'], fade_db=options['fade_db'],
                                        dc_offset=options['dc_offset'], dither=options['dither'],
                                        write_cue_file=options['write_cue'],
                                        dry_run=False, progress_bar=self.progress_pb,
                                        worker=worker, progress_callback=progress_callback,
                                        message_callback=message_callback,
                                        range_callback=range_callback)
                print(result)

                done += 1
                progress_callback.emit(i + 1)

        except Exception as e:
            traceback.print_exc()

        message_callback.emit(f'{done} of {count} file(s) processed.')
        if done < count:
            progress_callback.emit(0)
            message_callback.emit('Error while processing, Please check settings')
            play_notification(audio_file=self.current_dir / 'process_error.flac')
        else:
            progress_callback.emit(count)
            message_callback.emit(f'{done} of {count} file(s) processed.')
            play_notification(audio_file=self.current_dir / 'process_complete.flac')

        return True


def run(mw=SplitToolUi, parent=None):
    window = mw(parent=parent)
    return window.run()


if __name__ == "__main__":
    launch(mw=SplitToolUi, app_id=f'mitch.SplitTool.{__version__}')
