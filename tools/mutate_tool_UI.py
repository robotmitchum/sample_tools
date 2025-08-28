# coding:utf-8
"""
    :module: mutate_UI.py
    :description:
    :author: Michel 'Mitch' Pecqueur
    :date: 2025.07
"""

import inspect
import threading
import traceback
from functools import partial
from pathlib import Path

import numpy as np
from PyQt5 import QtGui, QtCore

from UI import mutate_tool as gui
from audio_player import play_notification
from base_tool_UI import BaseToolUi, launch
from common_ui_utils import add_ctx, resource_path, get_user_directory, style_widget, \
    popup_menu
from mutate import mutate
from sample_utils import Sample
from subprocess_utils import DisableShellWindows

__version__ = '1.0.0'


class MutateToolUi(gui.Ui_mutate_tool_mw, BaseToolUi):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.tool_name = 'Mutate Tool'
        self.tool_version = __version__
        self.icon_file = resource_path(self.current_dir / 'UI/icons/mutate_tool_64.png')
        self.setWindowTitle(f'{self.tool_name} v{self.tool_version}')

        self.suffix_le.setText('_rr')

        app_icon = QtGui.QIcon()
        app_icon.addFile(self.icon_file, QtCore.QSize(64, 64))
        self.setWindowIcon(app_icon)

        self.get_defaults()

        self.update_message('Generate randomized mutants/variants from single samples')

    def setup_connections(self):
        super().setup_connections()

        # Variations widgets
        add_ctx(self.count_sb, [1, 3, 5])
        self.seed_mode_cmb.currentTextChanged.connect(lambda state: self.seed_sb.setEnabled(state == 'value'))
        add_ctx(self.seed_sb, [1, 123, 12345])

        # Input widgets
        add_ctx(self.count_sb, [1, 3, 5])
        add_ctx(self.min_dur_dsb, [.05, .1, .2, .5, 1])
        add_ctx(self.trim_db_dsb, [-40, -60, -80, -100, -120])
        add_ctx(self.fade_db_dsb, [-20, -40, -60, -80, -100])

        # Noise|Tonal splitting
        add_ctx(self.split_stft_sb, [512, 1024, 2048, 4096, 8192])
        add_ctx(self.iterations_sb, [1, 3, 5])

        # Params widgets
        self.params_pb.clicked.connect(self.params_ctx)

        add_ctx(self.noise_amp_dsb, [1, .5, 0])
        add_ctx(self.noise_rate_dsb, [.5, 1, 2])
        add_ctx(self.noise_pitch_dsb, [-6, -3, 0, 3, 6])

        add_ctx(self.tonal_amp_dsb, [1, .5, 0])
        add_ctx(self.tonal_rate_dsb, [.5, 1, 2])
        add_ctx(self.tonal_pitch_dsb, [-6, -3, 0, 3, 6])

        # Random widgets
        self.random_pb.clicked.connect(self.random_ctx)

        add_ctx(self.noise_rand_amp_dsb, [1, .5, 0])
        add_ctx(self.noise_rand_rate_dsb, [.25, .5, 1, 2])
        add_ctx(self.noise_rand_pitch_dsb, [0, 1, 2])

        add_ctx(self.tonal_rand_amp_dsb, [1, .5, 0])
        add_ctx(self.tonal_rand_rate_dsb, [.25, .5, 1, 2])
        add_ctx(self.tonal_rand_pitch_dsb, [0, 0.05, 0.1])

        # Output directory
        default_dir = get_user_directory()
        desktop_dir = get_user_directory('Desktop')
        add_ctx(self.output_path_l, values=['', default_dir, desktop_dir],
                names=['Clear', 'Default directory', 'Desktop'])
        self.set_output_path_tb.clicked.connect(self.output_path_l.browse_path)

        # Sampling widgets
        self.oversample_cb.stateChanged.connect(lambda state: self.oversample_sb.setEnabled(state == 2))
        self.target_sr_cmb.currentTextChanged.connect(lambda state: self.target_sr_sb.setEnabled(state == 'custom'))
        add_ctx(self.oversample_sb, [2, 3, 4])
        add_ctx(self.target_sr_sb, [44100, 48000, 88200, 96000])
        add_ctx(self.suffix_le, ['', '_rr', '_seq', '_variant'])

        # Process buttons
        self.process_pb.clicked.connect(partial(self.as_worker, partial(self.do_process, mode='batch')))
        self.process_pb.setFixedHeight(24)
        style_widget(self.process_pb, properties={'border-radius': 8})

        self.process_sel_pb.clicked.connect(partial(self.as_worker, partial(self.do_process, mode='sel')))
        self.process_sel_pb.setFixedHeight(24)
        style_widget(self.process_sel_pb, properties={'background-color': 'rgb(95,95,95)', 'border-radius': 8})

        self.preview_pb.clicked.connect(partial(self.as_worker, partial(self.do_process, mode='preview')))
        self.preview_pb.setFixedHeight(24)
        style_widget(self.preview_pb, properties={'background-color': 'rgb(95,95,95)', 'border-radius': 8})

    def get_options(self):
        super().get_options()

        self.options.count = self.count_sb.value()
        self.options.seed_mode = self.seed_mode_cmb.currentText()
        self.options.seed = self.seed_sb.value()

        self.options.multi_chn = self.multi_chn_cb.isChecked()
        self.options.trim_db = self.trim_db_dsb.value()
        self.options.min_duration = self.min_dur_dsb.value()

        self.options.stft_size = self.split_stft_sb.value()
        self.options.iterations = self.iterations_sb.value()

        self.options.params = (
            (self.noise_amp_dsb.value(), self.noise_rate_dsb.value(), self.noise_pitch_dsb.value()),
            (self.tonal_amp_dsb.value(), self.tonal_rate_dsb.value(), self.tonal_pitch_dsb.value()))

        self.options.random_params = (
            (self.noise_rand_amp_dsb.value(), self.noise_rand_rate_dsb.value(), self.noise_rand_pitch_dsb.value()),
            (self.tonal_rand_amp_dsb.value(), self.tonal_rand_rate_dsb.value(), self.tonal_rand_pitch_dsb.value()))

        self.options.match_cues = self.match_cues_cb.isChecked()
        self.options.interp = self.amp_interp_cmb.currentText()

        self.options.bit_depth = None
        if self.bitdepth_cmb.currentText() != 'same':
            self.options.bit_depth = int(self.bitdepth_cmb.currentText())

        self.options.os_factor = (1, self.oversample_sb.value())[self.oversample_cb.isChecked()]

        self.options.fade_db = self.fade_db_dsb.value()

        self.options.target_sr = (None, self.target_sr_sb.value())[self.target_sr_cmb.currentText() == 'custom']

        self.options.output_dir = self.output_path_l.fullPath()
        self.options.ext = self.format_cmb.currentText()

    def do_process(self, worker, progress_callback, message_callback, mode='batch'):
        """
        :param int mode: 'batch' 'sel' or 'preview'
        :return:
        """
        if mode == 'batch':
            files = self.get_lw_items()
        else:
            files = self.get_sel_lw_items()

        if not files:
            return False

        self.player.stop()

        input_file = None

        if mode == 'preview':
            files = files[0:1]

        num_files = len(files)

        # Options
        self.get_options()
        options = vars(self.options)
        func_args = inspect.getfullargspec(mutate)[0]
        kwargs = {k: v for k, v in options.items() if k in func_args}

        done = 0

        # Progress bar init
        count = options['count']
        self.progress_pb.setMaximum(num_files * count)
        progress_callback.emit(0)
        message_callback.emit('Work in progress %p%')

        try:
            for i, input_file in enumerate(files):
                p = Path(input_file)
                info = Sample(input_file)
                prm = info.params

                if mode != 'preview':
                    output_ext = (options['ext'], p.suffix[1:])[options['ext'] == 'same']
                    output_dir = (p.parent, Path(options['output_dir']))[bool(options['output_dir'])]
                    kwargs['output_file'] = output_dir / f'{p.stem}{options['suffix']}.{output_ext}'

                # - Process -
                with DisableShellWindows():
                    result = mutate(input_file=input_file, **kwargs, worker=worker, message_callback=message_callback,
                                    progress_callback=progress_callback, progress_bar=self.progress_pb)
                    if mode == 'preview':
                        self.temp_audio.audio = np.concatenate(result, axis=0)
                        self.temp_audio.info = info

                done += 1
                progress_callback.emit((i + 1) * count)

        except Exception as e:
            traceback.print_exc()

        message_callback.emit(f'{done} of {num_files} file(s) processed.')

        if done < num_files:
            progress_callback.emit(0)
            message_callback.emit('Error while processing, Please check settings')
        elif mode == 'preview':
            if self.temp_audio.audio is not None:
                data = self.temp_audio.audio
                info = self.temp_audio.info
                self.temp_audio.info.input_file = input_file
                sr = options['target_sr'] or info.params.framerate
                self.playback_thread = threading.Thread(target=self.player.play, args=(data, sr, None, None),
                                                        daemon=True)
                self.playback_thread.start()

            progress_callback.emit(num_files * count)
            message_callback.emit('Preview completed')
        else:
            play_notification(audio_file=self.current_dir / 'process_complete.flac')

        self.refresh_lw_items()

        return True

    def params_ctx(self):
        names = ['Reset params']
        values = [[1, 1, 0, 1, 1, 0]]
        content = [{'type': 'cmds', 'name': name, 'cmd': partial(self.set_params, value)}
                   for name, value in zip(names, values)]
        popup_menu(content=content, parent=self.params_pb)

    def set_params(self, value):
        widgets = [self.noise_amp_dsb, self.noise_rate_dsb, self.noise_pitch_dsb,
                   self.tonal_amp_dsb, self.tonal_rate_dsb, self.tonal_pitch_dsb]
        for wid, val in zip(widgets, value):
            wid.setValue(val)

    def random_ctx(self):
        names = ['Reset random']
        values = [[.5, .333, 2, .5, .333, .05]]
        content = [{'type': 'cmds', 'name': name, 'cmd': partial(self.set_random, value)}
                   for name, value in zip(names, values)]
        popup_menu(content=content, parent=self.random_pb)

    def set_random(self, value):
        widgets = [self.noise_rand_amp_dsb, self.noise_rand_rate_dsb, self.noise_rand_pitch_dsb,
                   self.tonal_rand_amp_dsb, self.tonal_rand_rate_dsb, self.tonal_rand_pitch_dsb]
        for wid, val in zip(widgets, value):
            wid.setValue(val)


def run(mw=MutateToolUi, parent=None):
    window = mw(parent=parent)
    return window.run()


if __name__ == "__main__":
    launch(mw=MutateToolUi, app_id=f'mitch.MutateToolUi.{__version__}')
