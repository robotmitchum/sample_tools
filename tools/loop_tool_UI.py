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
from UI import loop_tool as gui
from audio_player import play_notification
from base_tool_UI import BaseToolUi, launch
from common_ui_utils import add_ctx, resource_path, get_user_directory, style_widget
from sample_utils import Sample
from utils import is_note_name, name_to_note, note_to_hz

# from simple_logger import SimpleLogger

__version__ = '1.2.0'


class LoopToolUi(gui.Ui_loop_tool_mw, BaseToolUi):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.tool_name = 'Loop Tool'
        self.tool_version = __version__
        self.icon_file = resource_path(self.current_dir / 'UI/icons/loop_tool_64.png')
        self.setWindowTitle(f'{self.tool_name} v{self.tool_version}')

        # log_path = self.base_dir / 'loop_tool_log.txt'
        # self.logger = SimpleLogger(log_path)

        self.suffix_le.setText('_looped')

        app_icon = QtGui.QIcon()
        app_icon.addFile(self.icon_file, QtCore.QSize(64, 64))
        self.setWindowIcon(app_icon)

        self.loop_min_length_dsb.setValue(100)
        self.loop_window_dsb.setValue(10)

        self.fade_in_dsb.setValue(.05)
        self.fade_out_dsb.setValue(.05)

        self.get_defaults()

        self.update_message('Detect loop points or modify audio files to make them loop')

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
        self.fft_range_cmb.currentTextChanged.connect(lambda state: self.fft_end_dsb.setEnabled(state == 'custom'))
        self.fft_range_cmb.currentTextChanged.connect(lambda state: self.fft_end_l.setEnabled(state == 'custom'))

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
        add_ctx(self.fft_end_dsb, [.125, .25, .5, .75, 1])

        min_fft_size = .001
        self.fft_start_dsb.setMaximum(self.fft_start_dsb.maximum() - min_fft_size)
        self.fft_start_dsb.valueChanged.connect(lambda value: self.fft_end_dsb.setMinimum(value + min_fft_size))

        add_ctx(self.resynth_duration_dsb, [.5, 1, 2, 4, 8])

        add_ctx(self.resynth_fade_in_dsb, [.125, .25, .5, 1])
        add_ctx(self.resynth_fade_out_dsb, [.125, .25, .5, 1])

        add_ctx(self.atonal_mix_dsb, [0, .05, .1, .2, .5, 1])
        add_ctx(self.freqs_le, ['', 'A3', '440', '110 440'])
        add_ctx(self.st_width_dsb, [0, .25, .5, .75, 1])

        # Output directory
        default_dir = get_user_directory()
        desktop_dir = get_user_directory('Desktop')
        add_ctx(self.output_path_l, values=['', default_dir, desktop_dir],
                names=['Clear', 'Default directory', 'Desktop'])
        self.set_output_path_tb.clicked.connect(self.output_path_l.browse_path)

        # Add Suffix widget
        add_ctx(self.suffix_le, ['_result', '_looped'])

        # Preview / Process buttons
        # Execute "as worker" to prevent multiple execution
        self.process_pb.clicked.connect(partial(self.as_worker, partial(self.do_process, mode='batch')))
        self.process_pb.setFixedHeight(24)
        style_widget(self.process_pb, properties={'border-radius': 8})

        self.process_sel_pb.clicked.connect(partial(self.as_worker, partial(self.do_process, mode='sel')))
        self.process_sel_pb.setFixedHeight(24)
        style_widget(self.process_sel_pb, properties={'background-color': 'rgb(95,95,95)', 'border-radius': 8})

        self.preview_pb.clicked.connect(partial(self.as_worker, partial(self.do_process, mode='preview')))
        self.preview_pb.setFixedHeight(24)
        style_widget(self.preview_pb, properties={'background-color': 'rgb(95,95,95)', 'border-radius': 8})

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
        fft_end = self.fft_end_dsb.value()
        duration = self.resynth_duration_dsb.value()
        atonal_mix = self.atonal_mix_dsb.value()
        freq_mode = self.resynth_freq_mode_cmb.currentText()

        freqs = []
        freq_tokens = self.freqs_le.text()
        if freq_tokens:
            for item in freq_tokens.replace(',', ' ').split(' '):
                if is_note_name(item):
                    freqs.append(note_to_hz(name_to_note(item)))
                else:
                    try:
                        freqs.append(float(item))
                    except Exception as e:
                        print(f'Freqs: failed to interpret {item}')

        resynth_mix = self.resynth_mix_cmb.currentText()
        fade_in = self.resynth_fade_in_dsb.value()
        fade_out = self.resynth_fade_out_dsb.value()
        width = self.st_width_dsb.value()

        self.options.resynth = \
            (None, {'fft_range': fft_range, 'fft_start': fft_start, 'fft_end': fft_end, 'duration': duration,
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

    def do_process(self, worker, progress_callback, message_callback, mode):
        self.player.stop()

        if mode == 'batch':
            files = self.get_lw_items()
        else:
            files = self.get_sel_lw_items()

        if not files:
            return False

        range_callback = worker.signals.progress_range

        input_file = None

        if mode == 'preview':
            files = files[0:1]

        count = len(files)
        per_file_progress = (None, self.progress_pb)[count <= 10]
        # per_file_progress = self.progress_pb

        importlib.reload(ls)

        # Options
        self.get_options()

        if mode != 'preview' and self.options.fd_preview_only is True:
            self.options.crossfade = None

        options = vars(self.options)
        func_args = inspect.getfullargspec(ls.loop_sample)[0]
        kwargs = {k: v for k, v in options.items() if k in func_args}
        kwargs.pop('bit_depth')

        done = 0

        # Progress bar init
        self.update_range(0, count * 100)
        if progress_callback is not None:
            range_callback.emit(0, count * 100)
            progress_callback.emit(0)
            message_callback.emit('Searching loops... %p%')

        try:
            for i, f in enumerate(files):
                if worker.is_stopped():
                    return False

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

                self.temp_audio = ls.loop_sample(input_file=f, output_file=filepath, bit_depth=bit_depth,
                                                 progress_bar=per_file_progress,
                                                 worker=worker, progress_callback=progress_callback,
                                                 message_callback=message_callback, **kwargs)

                if self.temp_audio is None:
                    return False

                done += 1
                if progress_callback is not None:
                    pb_value = (i + 1) * 100
                    progress_callback.emit(pb_value)

        except Exception as e:
            traceback.print_exc()
            # self.logger.log_exception(f'An error occurred: {e}')

        if done < count:
            progress_callback.emit(0)
            message_callback.emit('Error while processing, Please check settings')
            play_notification(audio_file=self.current_dir / 'process_error.flac')
        elif mode == 'preview':
            if self.temp_audio is not None:
                data = self.temp_audio.audio
                info = self.temp_audio.info
                self.temp_audio.info.input_file = input_file
                sr = info.params.framerate
                self.playback_thread = threading.Thread(target=self.player.play,
                                                        args=(data, sr, info.loopStart, info.loopEnd), daemon=True)
                self.playback_thread.start()

            progress_callback.emit(count * 100)
            message_callback.emit('Preview completed')
        else:
            progress_callback.emit(count * 100)
            message_callback.emit(f'{done} of {count} file(s) processed.')
            play_notification(audio_file=self.current_dir / 'process_complete.flac')

        self.refresh_lw_items()

        return True


def run(mw=LoopToolUi, parent=None):
    window = mw(parent=parent)
    return window.run()


if __name__ == "__main__":
    launch(mw=LoopToolUi, app_id=f'mitch.loopTool.{__version__}')
