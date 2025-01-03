# coding:utf-8
"""
    :module: loop_tool_UI.py
    :description: Loop sample tool
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
import inspect
import threading
import traceback
from functools import partial
from pathlib import Path

from PyQt5 import QtGui, QtCore

import loop_sample as ls
from audio_player import play_notification
from base_tool_UI import BaseToolUi, launch
from common_ui_utils import add_ctx, resource_path
from sample_utils import Sample
# import UI.loop_tool as gui
from tools.UI import loop_tool as gui

# from simple_logger import SimpleLogger

__version__ = '1.1.0'


class LoopToolUi(gui.Ui_loop_tool_mw, BaseToolUi):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle(f'Loop Tool v{__version__}')

        # log_path = self.base_dir / 'loop_tool_log.txt'
        # self.logger = SimpleLogger(log_path)

        self.suffix_le.setText('_looped')

        app_icon = QtGui.QIcon()
        img_file = resource_path(self.current_dir / 'UI/icons/loop_tool_64.png')
        app_icon.addFile(img_file, QtCore.QSize(64, 64))
        self.setWindowIcon(app_icon)

        self.loop_min_length_dsb.setValue(100)
        self.loop_window_dsb.setValue(10)

        self.fade_in_dsb.setValue(.05)
        self.fade_out_dsb.setValue(.05)

        self.get_defaults()

        self.progress_pb.setFormat('Detect loop points or modify audio files to make them loop')

    def setup_connections(self):
        super().setup_connections()

        # Envelope widgets
        self.env_shaping_cb.stateChanged.connect(lambda state: self.env_shaping_wid.setEnabled(state == 2))
        add_ctx(self.env_th_dsb, [0, -6, -12, -18, -24])
        add_ctx(self.env_min_dsb, [-12, -24, -30, -48, -60])
        add_ctx(self.env_window_dsb, [5, 10, 25, 50, 100])
        self.env_shaping_cb.stateChanged.connect(lambda state: self.env_window_wid.setEnabled(state == 2))

        # Detect loop widgets
        self.detect_loop_cb.stateChanged.connect(lambda state: self.detect_loop_wid.setEnabled(state == 2))
        self.detect_loop_cb.stateChanged.connect(lambda state: self.detect_extra_wid.setEnabled(state == 2))
        add_ctx(self.cues_sb, [90000, 300000, 640000, 1000000, 4000000])
        add_ctx(self.loop_min_length_dsb, [50, 100, 250, 500, 1000, 2000])
        add_ctx(self.loop_window_dsb, [5, 10, 25, 50, 100])
        add_ctx(self.window_offset_dsb, [0, .5, 1])

        self.range_limits_cb.stateChanged.connect(lambda state: self.range_limits_wid.setEnabled(state == 2))
        add_ctx(self.start_range_dsb, [.2, .333, .5, .667, .8, 1], default_idx=-1)
        add_ctx(self.end_range_dsb, [0, .333, .5, .667, .8, .95, .98], default_idx=0)

        # Cross-fade widgets
        self.crossfade_cb.stateChanged.connect(lambda state: self.crossfade_wid.setEnabled(state == 2))
        add_ctx(self.fade_in_dsb, [0, .01, .05, .25, .5, 1, 2])
        add_ctx(self.fade_out_dsb, [0, .01, .05, .25, .5, 1])

        # FFT Re-Synth widgets
        self.resynth_cb.stateChanged.connect(lambda state: self.resynth_wid.setEnabled(state == 2))
        self.resynth_cb.stateChanged.connect(lambda state: self.resynth_tone_wid.setEnabled(state == 2))
        self.resynth_cb.stateChanged.connect(lambda state: self.resynth_mix_wid.setEnabled(state == 2))

        self.fft_range_cmb.currentTextChanged.connect(lambda state: self.fft_start_dsb.setEnabled(state == 'custom'))
        self.fft_range_cmb.currentTextChanged.connect(lambda state: self.fft_start_l.setEnabled(state == 'custom'))
        self.fft_range_cmb.currentTextChanged.connect(lambda state: self.fft_size_dsb.setEnabled(state == 'custom'))
        self.fft_range_cmb.currentTextChanged.connect(lambda state: self.fft_size_l.setEnabled(state == 'custom'))

        self.resynth_mix_cmb.currentTextChanged.connect(lambda state: self.st_width_dsb.setEnabled(state == 'all'))
        self.resynth_mix_cmb.currentTextChanged.connect(lambda state: self.st_width_l.setEnabled(state == 'all'))

        self.resynth_mix_cmb.currentTextChanged.connect(lambda state: self.resynth_fade_in_l.setEnabled(state != 'all'))
        self.resynth_mix_cmb.currentTextChanged.connect(
            lambda state: self.resynth_fade_in_dsb.setEnabled(state != 'all'))
        self.resynth_mix_cmb.currentTextChanged.connect(
            lambda state: self.resynth_fade_out_l.setEnabled(state == 'loop'))
        self.resynth_mix_cmb.currentTextChanged.connect(
            lambda state: self.resynth_fade_out_dsb.setEnabled(state == 'loop'))
        self.resynth_mix_cmb.currentTextChanged.emit(self.resynth_mix_cmb.currentText())

        add_ctx(self.fft_start_dsb, [0, .125, .25, .5, .75])
        add_ctx(self.fft_size_dsb, [.125, .25, .5, 1])
        add_ctx(self.resynth_duration_dsb, [.5, 1, 2, 4, 8])

        add_ctx(self.resynth_fade_in_dsb, [.125, .25, .5, 1])
        add_ctx(self.resynth_fade_out_dsb, [.125, .25, .5, 1])

        add_ctx(self.atonal_mix_dsb, [0, .05, .1, .2, .5, 1])
        add_ctx(self.freqs_le, ['', '110', '440', '110,440'])
        add_ctx(self.st_width_dsb, [0, .25, .5, .75, 1])

        # Output path widget
        self.set_output_path_tb.clicked.connect(self.output_path_l.browse_path)

        # Add Suffix widget
        add_ctx(self.suffix_le, ['_result', '_looped'])

        # Preview / Process buttons
        # Execute "as worker" to prevent multiple execution
        self.process_pb.clicked.connect(partial(self.as_worker, partial(self.do_process, 'batch')))
        self.process_sel_pb.clicked.connect(partial(self.as_worker, partial(self.do_process, 'sel')))
        self.preview_pb.clicked.connect(partial(self.as_worker, partial(self.do_process, 'preview')))

        # Custom events

    def get_options(self):
        self.options.no_overwriting = self.no_overwriting_cb.isChecked()

        # - UI settings -
        self.options.env_window = self.env_window_dsb.value()

        # Envelope shaping parameters
        env_threshold = self.env_th_dsb.value()
        env_min = self.env_min_dsb.value()
        env_mode = self.env_mode_cmb.currentText()
        self.options.shape_env = (None, {'env_threshold': env_threshold, 'env_min': env_min, 'env_mode': env_mode})[
            self.env_shaping_cb.isChecked()]

        # Detect loop settings
        n_cues = self.cues_sb.value()
        loop_min_len = self.loop_min_length_dsb.value()
        loop_window = self.loop_window_dsb.value()
        window_offset = self.window_offset_dsb.value()
        range_limits = self.range_limits_cb.isChecked()
        start_range = [None, self.start_range_dsb.value()][range_limits]
        end_range = [None, self.end_range_dsb.value()][range_limits]
        hash_search = self.hash_search_cb.isChecked()
        self.options.detect_loop = (None, {'n_cues': n_cues, 'min_len': loop_min_len,
                                           'window_size': loop_window, 'window_offset': window_offset,
                                           'start_range': start_range, 'end_range': end_range,
                                           'hash_search': hash_search})[self.detect_loop_cb.isChecked()]

        # Cross-fade parameters
        self.options.fd_preview_only = self.fd_preview_only_cb.isChecked()
        fade_mode = self.fade_mode_cmb.currentText()
        fade_in = self.fade_in_dsb.value()
        fade_out = self.fade_out_dsb.value()
        self.options.crossfade = (None, {'fade_in': fade_in, 'fade_out': fade_out, 'mode': fade_mode})[
            self.crossfade_cb.isChecked()]

        # Re-synthesis parameters
        fft_range = self.fft_range_cmb.currentText()
        fft_start = self.fft_start_dsb.value()
        fft_size = self.fft_size_dsb.value()
        duration = self.resynth_duration_dsb.value()
        atonal_mix = self.atonal_mix_dsb.value()
        freq_mode = self.resynth_freq_mode_cmb.currentText()

        freqs = self.freqs_le.text() or None
        if freqs:
            freqs = [float(f) for f in freqs.replace(',', ' ').split(' ')]
        resynth_mix = self.resynth_mix_cmb.currentText()
        fade_in = self.resynth_fade_in_dsb.value()
        fade_out = self.resynth_fade_out_dsb.value()
        width = self.st_width_dsb.value()

        self.options.resynth = \
            (None, {'fft_range': fft_range, 'fft_start': fft_start, 'fft_size': fft_size, 'duration': duration,
                    'atonal_mix': atonal_mix, 'freq_mode': freq_mode, 'freqs': freqs,
                    'resynth_mix': resynth_mix, 'fade_in': fade_in, 'fade_out': fade_out, 'width': width})[
                self.resynth_cb.isChecked()]

        # Trim
        self.options.trim_after = self.trim_after_cb.isChecked()

        # Get format settings
        self.options.bit_depth = self.bitdepth_cmb.currentText()
        self.options.ext = self.format_cmb.currentText()

        if self.add_suffix_cb.isChecked():
            self.options.suffix = self.suffix_le.text() or ''
        else:
            self.options.suffix = ''

    def do_process(self, mode='batch'):
        if mode == 'batch':
            files = self.get_lw_items()
        else:
            files = self.get_sel_lw_items()

        if not files:
            return False

        input_file = None

        if mode == 'preview':
            files = files[0:1]

        per_file_progress = (None, self.progress_pb)[mode == 'preview']

        count = len(files)

        importlib.reload(ls)

        # Options
        self.get_options()

        if mode != 'preview' and self.options.fd_preview_only is True:
            self.options.crossfade = None

        options = vars(self.options)
        func_args = inspect.getfullargspec(ls.loop_sample)[0]
        kwargs = {k: v for k, v in options.items() if k in func_args}
        kwargs.pop('bit_depth')

        # Progress bar init
        self.progress_pb.setMaximum(count)
        self.progress_pb.setValue(0)
        self.progress_pb.setTextVisible(True)
        self.progress_pb.setFormat('Searching loops... %p%')

        done = 0
        self.player.stop()

        try:
            for i, f in enumerate(files):
                p = Path(f)
                parent = self.output_path_l.fullPath() or p.parent
                basename = p.stem

                ext = options['ext']
                if ext == 'same':
                    ext = p.suffix[1:]

                # aif support for reading only
                if ext not in ['wav', 'flac']:
                    ext = 'wav'

                if mode != 'preview':
                    bit_depth = options['bit_depth']
                    if bit_depth == 'same':
                        bit_depth = Sample(f).params.sampwidth * 8
                    elif isinstance(bit_depth, str):
                        bit_depth = eval(bit_depth)
                    filepath = Path.joinpath(Path(parent), f'{basename}{options["suffix"]}.{ext}').__str__()
                else:
                    filepath = None
                    bit_depth = None

                # QUICK FIX : updating the progress bar during loop search makes the app unstable even with throttling
                # Try to thread the progress bar in a future update
                self.temp_audio = ls.loop_sample(input_file=f, output_file=filepath, bit_depth=bit_depth,
                                                 progress_pb=per_file_progress, **kwargs)
                # self.temp_audio = ls.loop_sample(input_file=f, output_file=filepath, bit_depth=bit_depth,
                #                                  progress_pb=self.progress_pb, **kwargs)
                done += 1
                self.progress_pb.setMaximum(count)
                self.progress_pb.setValue(i + 1)

        except Exception as e:
            traceback.print_exc()
            # self.logger.log_exception(f'An error occurred: {e}')

        self.progress_pb.setMaximum(1)
        if done < count:
            self.progress_pb.setValue(0)
            self.progress_pb.setFormat('Error while processing, Please check settings')
            play_notification(audio_file=self.current_dir / 'process_error.flac')
        elif mode == 'preview':
            if self.temp_audio is not None:
                data = self.temp_audio.audio
                info = self.temp_audio.info
                self.temp_audio.info.input_file = input_file
                sr = info.params.framerate
                self.playback_thread = threading.Thread(target=self.player.play,
                                                        args=(data, sr, info.loopStart, info.loopEnd,
                                                              self.progress_pb.setFormat), daemon=True)
                self.playback_thread.start()

            self.progress_pb.setValue(1)
            self.progress_pb.setFormat(f'Preview completed')
        else:
            self.progress_pb.setValue(1)
            self.progress_pb.setFormat(f'{done} of {count} file(s) processed.')
            play_notification(audio_file=self.current_dir / 'process_complete.flac')

        self.refresh_lw_items()

        return True


def run(mw=LoopToolUi, parent=None):
    window = mw(parent=parent)
    return window.run()


if __name__ == "__main__":
    launch(mw=LoopToolUi, app_id=f'mitch.loopTool.{__version__}')
