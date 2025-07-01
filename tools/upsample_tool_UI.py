# coding:utf-8
"""
    :module: upsample_tool_UI.py
    :description:
    :author: Michel 'Mitch' Pecqueur
    :date: 2024.08
"""

import importlib
import os
import threading
import traceback
from functools import partial
from pathlib import Path

import numpy as np
import soundfile as sf
from PyQt5 import QtWidgets, QtGui, QtCore
from scipy import interpolate

import noise_reduction as nr
from UI import upsample_tool as gui
from audio_player import play_notification
from base_tool_UI import BaseToolUi, launch
from common_audio_utils import pad_audio
from common_ui_utils import add_ctx, FilePathLabel, replace_widget, resource_path, get_user_directory, style_widget
from file_utils import resolve_overwriting
from loop_sample import db_to_lin
from sample_utils import Sample
from split_audio import envelope_transform
from subprocess_utils import DisableShellWindows
from upsample import audio_upsample
from utils import append_metadata, set_md_tags

__version__ = '1.1.1'


class UpsampleToolUi(gui.Ui_upsample_tool_mw, BaseToolUi):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.tool_name = 'Upsample Tool'
        self.tool_version = __version__
        self.icon_file = resource_path(self.current_dir / 'UI/icons/upsample_tool_64.png')
        self.setWindowTitle(f'{self.tool_name} v{self.tool_version}')

        self.suffix_le.setText('_up')
        self.denoise_mix_dsb.setValue(1.0)

        app_icon = QtGui.QIcon()
        app_icon.addFile(self.icon_file, QtCore.QSize(64, 64))
        self.setWindowIcon(app_icon)

        self.get_defaults()

        self.update_message('Up-sample audio file(s) using spectral band replication')

    def setup_connections(self):
        super().setup_connections()

        # Declip widgets
        self.declip_cb.stateChanged.connect(lambda state: self.declip_wid.setEnabled(state == 2))
        add_ctx(self.declip_th_dsb, [-.1, -.5, -1, -3])
        add_ctx(self.declip_mix_dsb, [.125, .25, .5, 1])

        # Denoise widgets
        self.denoise_cb.stateChanged.connect(lambda state: self.denoise_wid.setEnabled(state == 2))
        add_ctx(self.denoise_mix_dsb, [1, 1.5, 2])
        self.denoise_mode_cmb.currentTextChanged.connect(
            lambda state: self.noise_path_wid.setEnabled(state == 'custom'))

        self.noise_path_l = replace_widget(self.noise_path_l, FilePathLabel(file_mode=True, parent=self))
        add_ctx(self.noise_path_l, values=[''], names=['Clear'])
        self.set_noise_path_tb.clicked.connect(self.noise_path_l.browse_path)

        # Output directory
        default_dir = get_user_directory()
        desktop_dir = get_user_directory('Desktop')
        add_ctx(self.output_path_l, values=['', default_dir, desktop_dir],
                names=['Clear', 'Default directory', 'Desktop'])
        self.set_output_path_tb.clicked.connect(self.output_path_l.browse_path)

        # Upsample widgets
        self.upsample_cb.stateChanged.connect(lambda state: self.upsample_wid.setEnabled(state == 2))
        self.f_max_mode_cmb.currentTextChanged.connect(lambda state: self.f_max_sb.setEnabled(state == 'custom'))
        add_ctx(self.f_max_sb, [5512, 8000, 10000, 11025, 16000, 20000, 22050])
        add_ctx(self.target_sr_sb, [44100, 48000, 88200, 96000])
        add_ctx(self.upsample_mix_dsb, [.25, .5, 1, 2])

        add_ctx(self.suffix_le, ['_up', '_result'])

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

        # - UI settings -
        self.options.declip = self.declip_cb.isChecked()
        self.options.declip_th = self.declip_th_dsb.value()
        self.options.declip_mix = self.declip_mix_dsb.value()

        self.options.denoise = self.denoise_cb.isChecked()
        self.options.denoise_mode = self.denoise_mode_cmb.currentText()
        self.options.noise_path = self.noise_path_l.fullPath()
        self.options.denoise_mix = self.denoise_mix_dsb.value()

        self.options.upsample = self.upsample_cb.isChecked()
        if self.f_max_mode_cmb.currentText() == 'custom':
            self.options.f_max = self.f_max_sb.value()
        else:
            self.options.f_max = None
        self.options.target_sr = self.target_sr_sb.value()
        self.options.upsample_mix = self.upsample_mix_dsb.value()

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

        importlib.reload(nr)

        input_file = None

        if mode == 'preview':
            files = files[0:1]

        count = len(files)

        # Options
        self.get_options()
        options = vars(self.options)
        subtypes = {16: 'PCM_16', 24: 'PCM_24'}

        done = 0

        # Progress bar init
        self.progress_pb.setMaximum(count)
        progress_callback.emit(0)
        message_callback.emit('Work in progress %p%')

        try:
            for i, input_file in enumerate(files):
                info = Sample(input_file)
                prm = info.params

                data, sr = sf.read(input_file)

                # Get 'smpl' metadata
                tags = ['note', 'pitchFraction', 'loopStart', 'loopEnd', 'loops', 'cues']

                if sr != options['target_sr']:
                    cue_scl = options['target_sr'] / sr
                    if info.loopStart is not None:
                        info.loopStart = int(info.loopStart * cue_scl)
                        info.loopEnd = int(info.loopEnd * cue_scl)
                        info.loops[0] = [info.loopStart, info.loopEnd]

                md = {}
                for tag in tags:
                    md[tag] = getattr(info, tag, None)
                md = {k: v for k, v in md.items() if v}

                p = Path(input_file)
                parent = self.output_path_l.fullPath() or p.parent
                stem = p.stem

                if options['ext'] == 'same':
                    ext = p.suffix.strip('.')
                else:
                    ext = options['ext']
                if ext == 'flac':
                    subtypes[32] = 'PCM_24'
                else:
                    subtypes[32] = 'FLOAT'

                if options['bit_depth'] == 'same':
                    bit_depth = prm.sampwidth * 8
                else:
                    bit_depth = int(options['bit_depth'])

                suffix = options['suffix']
                filepath = Path(parent) / f'{stem}{suffix}.{ext}'

                # - Process -

                result = data
                nch = result.ndim

                if options['declip']:
                    result = nr.declip(audio=result, db=options['declip_th'], threshold=None, mix=options['declip_mix'],
                                       normalize=False)

                if options['denoise']:
                    if options['denoise_mode'] == 'custom':
                        if options['noise_path']:
                            noise_data, _ = sf.read(options['noise_path'])
                        else:
                            noise_data = None
                    else:
                        noise_data = nr.generate_quantize_noise(output_file=None, sr=sr, length=None, bd=8,
                                                                seed=0)

                    if noise_data is not None:
                        noise_amp = max(1, options['denoise_mix'])
                        result = nr.denoise(audio=result, noise_profile=noise_data * noise_amp, sr=sr,
                                            mix=min(1, options['denoise_mix']), normalize=True)

                if options['upsample']:
                    env = None
                    env_transform = False

                    if env_transform:
                        env = envelope_transform(data=result, w=1024, mode='max', interp='linear')
                        env_threshold = db_to_lin(-.1)
                        env_min = db_to_lin(-48)
                        env = np.clip(env, a_min=env_min, a_max=env_threshold) / env_threshold
                        if nch > 1:
                            env = np.tile(env, (nch, 1)).T
                        result /= env

                    pad = 256
                    smp_len = len(result)
                    result = pad_audio(result, before=pad, after=pad, mode='reflect')

                    with DisableShellWindows():
                        result = audio_upsample(input_file=None, output_file=None, audio=result, sr=sr,
                                                f_max=options['f_max'], target_sr=options['target_sr'],
                                                mix=options['upsample_mix'])

                    m = options['target_sr'] / sr
                    result = result[int(pad * m):int((smp_len + pad) * m)]

                    if env_transform:
                        x = np.arange(len(env))
                        x_new = np.linspace(0, len(env) - 1, num=len(result), endpoint=False)
                        env = interpolate.interp1d(x, env, kind='linear')(x_new)
                        result *= env

                    sr = options['target_sr']

                # - - -

                self.temp_audio.audio = result
                self.temp_audio.info = info

                # Write file

                if mode != 'preview':
                    if self.no_overwriting_cb.isChecked() and Path(filepath).resolve() == Path(input_file).resolve():
                        resolve_overwriting(input_file, mode='dir', dir_name='backup_', test_run=False)

                    # Soundfile only recognizes aiff and not aif when writing
                    sf_path = (filepath, f'{filepath}f')[ext == 'aif']
                    sf.write(str(sf_path), self.temp_audio.audio, sr, subtype=subtypes[bit_depth])
                    if sf_path != filepath:
                        os.rename(sf_path, filepath)

                    if md:
                        # Add 'smpl' metadata
                        if ext == 'wav':
                            append_metadata(str(filepath), note=info.note, pitch_fraction=info.pitchFraction,
                                            loop_start=info.loopStart, loop_end=info.loopStart)
                        # Add metadata as tags for flac
                        elif ext == 'flac':
                            set_md_tags(str(filepath), md=md)

                done += 1
                progress_callback.emit(i + 1)

        except Exception as e:
            traceback.print_exc()

        message_callback.emit(f'{done} of {count} file(s) processed.')

        if done < count:
            progress_callback.emit(0)
            message_callback.emit('Error while processing, Please check settings')
        elif mode == 'preview':
            if self.temp_audio.audio is not None:
                data = self.temp_audio.audio
                info = self.temp_audio.info
                self.temp_audio.info.input_file = input_file
                if options['upsample']:
                    sr = options['target_sr']
                else:
                    sr = info.params.framerate
                self.playback_thread = threading.Thread(target=self.player.play,
                                                        args=(data, sr, info.loopStart, info.loopEnd), daemon=True)
                self.playback_thread.start()

            progress_callback.emit(count * 100)
            message_callback.emit('Preview completed')
        else:
            play_notification(audio_file=self.current_dir / 'process_complete.flac')

        self.refresh_lw_items()

        return True

    def set_noise_path(self):
        if self.last_file:
            startdir = str(Path(self.last_file).parent)
        else:
            startdir = os.getcwd()
        fmts = [f'*{fmt}' for fmt in self.file_types]
        fltr = 'Audio File ({})'.format(' '.join(fmts))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select noise profile", startdir, fltr)
        if path:
            path = os.path.normpath(path)
            p = Path(path)
            if p.is_relative_to(self.current_dir):
                path = str(p.relative_to(self.current_dir))
            self.noise_path_l.setFullPath(path)


def run(mw=UpsampleToolUi, parent=None):
    window = mw(parent=parent)
    return window.run()


if __name__ == "__main__":
    launch(mw=UpsampleToolUi, app_id=f'mitch.UpsampleToolUi.{__version__}')
