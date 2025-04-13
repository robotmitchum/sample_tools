# coding:utf-8
"""
    :module: st_tool_UI.py
    :description: Convert mono samples to stereo using pseudo-stereo techniques or Impulse responses
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
import os
import threading
import traceback
from functools import partial
from pathlib import Path
from typing import cast

import soundfile as sf
from PyQt5 import QtWidgets, QtGui, QtCore

import pseudo_stereo as ps
from UI import st_tool as gui
from audio_player import play_notification
from base_tool_UI import BaseToolUi, launch
from common_audio_utils import rms
from common_ui_utils import add_ctx, replace_widget, FilePathLabel, style_widget
from common_ui_utils import resource_path, resource_path_alt, get_user_directory
from file_utils import resolve_overwriting
from sample_utils import Sample
from utils import append_metadata, set_md_tags

__version__ = '1.1.1'


class StToolUi(gui.Ui_st_tool_mw, BaseToolUi):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle(f'Stereo Tool v{__version__}')

        self.ir_path_l = cast(FilePathLabel, self.ir_path_l)  # For auto-completion

        self.file_types = ['.wav', '.flac', '.aif']
        self.output_path = ''

        self.ir_samples = []
        self.get_ir_samples()

        self.width_dsb.setValue(1.0)

        self.cutoff_dsb.setValue(500)
        self.band_dsb.setValue(500)

        self.suffix_le.setText('_st')

        app_icon = QtGui.QIcon()
        img_file = resource_path(self.current_dir / 'UI/icons/st_tool_64.png')
        app_icon.addFile(img_file, QtCore.QSize(64, 64))
        self.setWindowIcon(app_icon)

        self.get_defaults()

        self.update_message('Apply pseudo-stereo/stereo imaging effect to audio file(s)')

    def setup_connections(self):
        super().setup_connections()

        # IR path widgets
        self.ir_path_l = replace_widget(self.ir_path_l, FilePathLabel(file_mode=True, parent=self))
        self.set_ir_path_tb.clicked.connect(self.ir_path_l.browse_path)
        self.ir_path_l.setContextMenuPolicy(3)
        self.ir_path_l.customContextMenuRequested.connect(self.ir_path_l_ctx)
        self.ir_path_l.setEnabled(False)

        # Output path widgets
        default_dir = get_user_directory()
        desktop_dir = get_user_directory('Desktop')
        add_ctx(self.output_path_l, values=['', default_dir, desktop_dir],
                names=['Clear', 'Default directory', 'Desktop'])
        self.set_output_path_tb.clicked.connect(self.output_path_l.browse_path)

        # Delay widget
        add_ctx(self.delay_dsb, [3, 6, 12, 18, 24, 30])

        # Stereo mode widgets
        self.st_mode_cmb.currentTextChanged.connect(lambda state: self.delay_l.setEnabled(not state.startswith('conv')))
        self.st_mode_cmb.currentTextChanged.connect(
            lambda state: self.delay_dsb.setEnabled(not state.startswith('conv')))
        self.st_mode_cmb.currentTextChanged.connect(
            lambda state: self.set_ir_path_tb.setEnabled(state.startswith('conv')))

        self.st_mode_cmb.currentTextChanged.connect(lambda state: self.seed_l.setEnabled(state == 'velvet'))
        self.st_mode_cmb.currentTextChanged.connect(lambda state: self.seed_sb.setEnabled(state == 'velvet'))
        add_ctx(self.seed_sb, [-1, 0, 123])

        self.st_mode_cmb.currentTextChanged.connect(lambda state: self.mxlen_cb.setEnabled(state.startswith('conv')))
        self.st_mode_cmb.currentTextChanged.connect(lambda state: self.ir_path_l.setEnabled(state.startswith('conv')))
        self.st_mode_cmb.currentTextChanged.connect(lambda state: self.ir_path_l.setEnabled(state.startswith('conv')))
        self.st_mode_cmb.currentTextChanged.connect(lambda state: self.wet_dsb.setEnabled(state == 'convolve'))
        self.st_mode_cmb.currentTextChanged.connect(lambda state: self.wet_l.setEnabled(state == 'convolve'))
        add_ctx(self.wet_dsb, [0, .25, .5, .75, 1])

        self.st_mode_cmb.currentTextChanged.connect(
            lambda state: self.filter_side_wid.setDisabled(state.startswith('conv')))
        self.st_mode_cmb.currentTextChanged.connect(
            lambda state: self.filter_side_cb.setDisabled(state.startswith('conv')))

        self.st_mode_cmb.currentTextChanged.connect(
            lambda state: self.balance_cb.setDisabled(state.startswith('conv')))

        # Refresh widgets related to combo box
        self.st_mode_cmb.currentTextChanged.emit(self.st_mode_cmb.currentText())

        # LP/HP widgets
        self.filter_side_cb.stateChanged.connect(lambda state: self.filter_side_wid.setEnabled(state == 2))
        self.st_mode_cmb.currentTextChanged.connect(
            lambda state: self.filter_side_wid.setEnabled(not state.startswith('conv')))
        add_ctx(self.cutoff_dsb, [100, 250, 500, 750, 1000, 2000, 4000])
        add_ctx(self.band_dsb, [-500, -250, -100, 100, 250, 500, 1000])

        # Width widget
        add_ctx(self.width_dsb, [0, .25, .5, .75, 1])

        # Suffix widget
        add_ctx(self.suffix_le, ['_st', '_result'])

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

        # Custom events

    def get_options(self):
        self.options.no_overwriting = self.no_overwriting_cb.isChecked()

        # - UI settings -
        self.options.mode = self.st_mode_cmb.currentText()
        self.options.delay = self.delay_dsb.value()
        self.options.width = self.width_dsb.value()
        self.options.seed = self.seed_sb.value()
        self.options.balance = (0, 4)[self.balance_cb.isChecked()]
        self.options.mx_len = self.mxlen_cb.isChecked()
        self.options.wet = self.wet_dsb.value()

        self.options.cutoff = (None, self.cutoff_dsb.value())[self.filter_side_cb.isChecked()]
        self.options.band = self.band_dsb.value()

        self.options.ir_path = (None, self.ir_path_l.fullPath())[self.options.mode.startswith('conv')]

        # Get format settings
        self.options.bit_depth = self.bitdepth_cmb.currentText()
        self.options.ext = self.format_cmb.currentText()

        if self.add_suffix_cb.isChecked():
            self.options.suffix = self.suffix_le.text() or ''
        else:
            self.options.suffix = ''

    def do_process(self, worker, progress_callback, message_callback, mode='batch'):
        """
        :param Worker or None worker:
        :param function or None progress_callback:
        :param function or None message_callback:

        :param int mode: 'batch' 'sel' or 'preview'
        :return:
        """
        if mode == 'batch':
            files = self.get_lw_items()
        else:
            files = self.get_sel_lw_items()

        if not files:
            return False

        input_file = None

        if mode == 'preview':
            files = files[0:1]

        count = len(files)

        # Options
        self.get_options()
        options = vars(self.options)
        subtypes = {16: 'PCM_16', 24: 'PCM_24'}

        if options['mode'].startswith('conv') and not Path(options['ir_path']).is_file():
            print('No IR specified')
            return False

        importlib.reload(ps)

        done = 0
        self.player.stop()

        # Progress bar init
        self.progress_pb.setMaximum(count)
        if progress_callback is not None:
            progress_callback.emit(0)
            message_callback.emit('Work in progress %p%')

        try:
            for i, input_file in enumerate(files):
                if worker.is_stopped():
                    return False

                info = Sample(input_file)
                prm = info.params

                data, sr = sf.read(input_file)
                if data.ndim > 1:
                    mx = rms(data)
                    data = data.mean(axis=-1)
                    data *= mx / rms(data)  # Match volume

                # Get 'smpl' metadata
                tags = ['note', 'pitchFraction', 'loopStart', 'loopEnd', 'loops', 'cues']
                md = {}
                for tag in tags:
                    md[tag] = getattr(info, tag, None)
                md = {k: v for k, v in md.items() if v}

                p = Path(input_file)
                parent = self.output_path or p.parent
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
                filepath = Path.joinpath(Path(parent), f'{stem}{suffix}.{ext}')

                self.temp_audio.audio = ps.pseudo_stereo(data, sr, delay=options['delay'], mode=options['mode'],
                                                         seed=options['seed'], ir_file=options['ir_path'],
                                                         mx_len=options['mx_len'], wet=options['wet'],
                                                         cutoff=options['cutoff'], band=options['band'],
                                                         balance=options['balance'],
                                                         width=options['width'])
                self.temp_audio.info = info

                if mode != 'preview':
                    if self.no_overwriting_cb.isChecked() and str(filepath) == input_file:
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
                                            loop_start=info.loopStart, loop_end=info.loopEnd)
                        # Add metadata as tags for flac
                        elif ext == 'flac':
                            # Add short info about conversion
                            comment = f"pseudo_stereo: {options['mode']}"
                            if options['mode'].startswith('conv'):
                                ir_name = Path(options['ir_path']).stem
                                comment += f" {ir_name}"
                            md['comment'] = comment
                            set_md_tags(str(filepath), md=md)

                done += 1
                if progress_callback is not None:
                    progress_callback.emit(i + 1)

        except Exception as e:
            traceback.print_exc()

        if progress_callback is not None:
            message_callback.emit(f'{done} of {count} file(s) processed.')

        if done < count:
            self.progress_pb.setMaximum(1)
            if progress_callback is not None:
                progress_callback.emit(0)
                message_callback.emit('Error while processing, Please check settings')
        elif mode == 'preview':
            if self.temp_audio.audio is not None:
                data = self.temp_audio.audio
                info = self.temp_audio.info
                self.temp_audio.info.input_file = input_file
                sr = info.params.framerate
                self.playback_thread = threading.Thread(target=self.player.play,
                                                        args=(data, sr, info.loopStart, info.loopEnd), daemon=True)
                self.playback_thread.start()

            self.progress_pb.setMaximum(1)
            if progress_callback is not None:
                progress_callback.emit(1)
                message_callback.emit('Preview completed')
        else:
            play_notification(audio_file=self.current_dir / 'process_complete.flac')

        self.refresh_lw_items()

        return True

    def get_ir_samples(self, ir_subdir='dh_ir'):
        self.ir_samples = []
        ir_path = resource_path_alt(self.base_dir / ir_subdir, parent_dir=self.app_dir, as_str=False)
        if ir_path.is_dir():
            for ext in self.file_types:
                ir_smp = ir_path.glob(f'*{ext}')
                self.ir_samples.extend(ir_smp)
        if self.ir_samples:
            self.ir_path_l.setFullPath(self.ir_samples[0])

    def ir_path_l_ctx(self):
        values = self.ir_samples
        names = [str(Path(ir_smp).stem) for ir_smp in self.ir_samples]
        menu = QtWidgets.QMenu(self)
        names = [menu.addAction(name) for name in names]
        cmds = [partial(self.ir_path_l.setFullPath, val) for val in values]
        action = menu.exec_(QtGui.QCursor.pos())
        for name, cmd in zip(names, cmds):
            if action == name:
                cmd()


def run(mw=StToolUi, parent=None):
    window = mw(parent=parent)
    return window.run()


if __name__ == "__main__":
    launch(mw=StToolUi, app_id=f'mitch.StTool.{__version__}')
