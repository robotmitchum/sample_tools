# coding:utf-8
"""
    :module: Smp2Ds_UI.py
    :description: Create a Decent Sampler preset (.dspreset) file from a collection of samples

    Samples must be in flac or wav

    They have to be correctly named and located in a directory following this pattern 'Instrument/Samples'

    - ID3 tags are supported for 'flac' format
    - Note/Loop metadata from 'smpl' chunk are supported for wav format
    - limited features for aif format (no support for embedded tags or metadata)

    :author: Michel 'Mitch' Pecqueur
    :date: 2024.05
"""

import ctypes
import importlib
import os
import platform
import re
import sys
import traceback
from functools import partial
from pathlib import Path
import inspect

from PyQt5 import QtGui, QtCore, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtWidgets import QMenu, QMenuBar, QAction

import smp_to_dspreset as smp2ds
from UI import smp_to_ds as gui
from audio_player import play_notification
from common_prefs_utils import Node, get_settings, set_settings, read_settings, write_settings
from common_ui_utils import add_ctx, add_insert_ctx, popup_menu, get_custom_font, style_widget
from common_ui_utils import beautify_str, resource_path, resource_path_alt, shorten_path, AboutDialog
from jsonFile import read_json
from smp_to_dspreset import __version__, __ds_version__
from tools.worker import Worker
from dark_fusion_style import apply_dark_theme


class Smp2dsUi(gui.Ui_smp_to_ds_ui, QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setupUi(self)
        self.setAttribute(Qt.Qt.WA_DeleteOnClose)

        self.current_dir = Path(__file__).parent
        self.base_dir = self.current_dir.parent

        self.tool_name = 'SMP2ds'
        self.tool_version = __version__
        self.icon_file = resource_path(self.current_dir / 'UI/icons/smp2ds_64.png')
        self.setWindowTitle(f'{self.tool_name} v{self.tool_version}')

        self.options = Node()

        self.root_dir = None  # Instrument root directory
        self.ir_subdir = None

        self.setAcceptDrops(True)

        self.threadpool = QtCore.QThreadPool(parent=self)
        self.worker = None
        self.active_workers = []
        self.worker_result = None
        self.event_loop = QtCore.QEventLoop()

        self.current_dir = Path(__file__).parent
        self.base_dir = self.current_dir.parent

        if getattr(sys, 'frozen', False):
            self.app_dir = Path(sys.executable).parent
        else:
            self.app_dir = self.base_dir
        os.chdir(self.app_dir)

        self.smp_attrib_cfg = resource_path_alt(self.base_dir / 'smp_attrib_cfg.json', parent_dir=self.app_dir)

        self.pattern_le.setText('{group}_{note}_{vel}')
        self.limit_le.setText('')
        self.override_cb.setChecked(False)

        self.instr_range_cfg = {}
        self.instr_range_cfg_path = resource_path_alt(self.base_dir / 'instr_range_cfg.json', parent_dir=self.app_dir)
        self.set_instr_range_cfg(cfg_file=self.instr_range_cfg_path)

        self.limit_le.setText('autox2')
        self.set_adsr('.001 .25 1.0 .375')
        self.set_adr('-100 100 100')
        self.max_adsr_dsb.setValue(10)

        self.attenuation_dsb.setValue(-9)
        self.fk_volume_dsb.setValue(-15)

        self.plt_cfg_dir = resource_path_alt(self.base_dir / 'plt_cfg', parent_dir=self.app_dir, as_str=False)
        text_font_path = resource_path(self.current_dir / 'HelveticaNeueThin.otf')
        self.text_font = (text_font_path, 24)

        self.plt_cfg_suffix = '_plt_cfg'
        self.default_plt = 'Dark'
        self.palette_cfg = {}
        self.populate_palette_cmb()

        self.setup_menu_bar()
        self.default_settings = Node()
        self.settings_ext = 'smp2ds'
        self.settings_path = None
        self.set_settings_path()

        self.setup_connections()

        self.crossfade_cmb.setCurrentText('linear')

        app_icon = QtGui.QIcon()
        app_icon.addFile(self.icon_file, QtCore.QSize(64, 64))
        self.setWindowIcon(app_icon)

        custom_font = get_custom_font(self.current_dir / 'RobotoMono-Medium.ttf')

        self.progress_pb.setFont(custom_font)
        self.progress_pb.setStyleSheet('QProgressBar{border: none;}')
        self.progress_pb.setAlignment(QtCore.Qt.AlignCenter)
        self.progress_pb.setTextVisible(True)
        self.progress_pb.setFont(custom_font)
        self.progress_pb.setValue(0)
        self.update_message('Create a Decent Sampler preset from samples')

    def setup_connections(self):
        self.setpath_tb.clicked.connect(self.set_rootdir)

        # Pattern widgets
        add_ctx(self.pattern_le,
                values=['{group}_{note}', '{group}_{note}_{vel}', '{group}_{note}_{seqPosition}',
                        '{group}_{note}_{trigger}'],
                trigger=self.pattern_pb)
        smp_attrib = ['group', 'note', 'pitchFraction', 'vel', 'trigger', 'seqPosition', '_']
        src_fields = [f'{{{at}}}' for at in smp_attrib]
        add_insert_ctx(self.pattern_le, values=src_fields)

        # Mapping widgets
        self.limit_ctx()
        add_ctx(self.transpose_sb, values=[-12, 0, 12])
        values = ['', 'x3', 'x5', '-1 0 1', '-2 -1 0 1 2']
        names = ['None', 'x3', 'x5', '-1 0 1\t(same as x3)', '-2 -1 0 1 2\t(same as x5)']
        add_ctx(self.rrofs_le, values=values, names=names, trigger=self.fake_rr_pb)
        self.pf_mode_cmb.currentTextChanged.connect(lambda state: self.pf_th_dsb.setEnabled(state not in ['on', 'off']))
        add_ctx(self.pf_th_dsb, values=[0, 2.5, 5, 10, 50, 100])

        # Envelope widgets
        self.adsr_enable_cb.stateChanged.connect(lambda state: self.set_adsr_wid_style(value=state))

        self.ADSR_pb.clicked.connect(self.adsr_ctx)
        add_ctx(self.A_dsb, values=[0, .001, .25, .5, 1])
        add_ctx(self.D_dsb, values=[.25, .375, .5, 1, 2])
        add_ctx(self.S_dsb, values=[0, .2, .5, .8, 1])
        add_ctx(self.R_dsb, values=[.25, .375, .5, 1, 2])

        self.ADRr_pb.clicked.connect(self.adr_ctx)
        add_ctx(self.Ar_dsb, values=[-100, 0, 100], names=['log\t-100', 'lin\t0', 'exp\t100'])
        add_ctx(self.Dr_dsb, values=[-100, 0, 100], names=['log\t-100', 'lin\t0', 'exp\t100'])
        add_ctx(self.Rr_dsb, values=[-100, 0, 100], names=['log\t-100', 'lin\t0', 'exp\t100'])

        self.crossfade_cmb.currentTextChanged.connect(lambda state: self.crossfade_dsb.setEnabled(state != 'off'))

        # Fake Legato Widgets
        self.fake_leg_cb.stateChanged.connect(lambda state: self.fake_leg_wid.setEnabled(state))
        add_ctx(self.fk_leg_start_dsb, values=[.1, .25, .5, 1])
        add_ctx(self.fk_leg_a_dsb, values=[.01, .1, .2, .4])
        add_ctx(self.fk_leg_a_curve_dsb, values=[-100, 0, 100], names=['log\t-100', 'lin\t0', 'exp\t100'])

        # Fake Release Widgets
        self.fake_rls_cb.stateChanged.connect(lambda state: self.fake_options_wid.setEnabled(state))
        self.fake_rls_cb.stateChanged.connect(lambda state: self.fk_adsr_wid.setEnabled(state))
        self.fk_rls_mode_cmb.currentTextChanged.connect(
            lambda state: self.fk_rls_tweaks_wid.setEnabled(state == 'start'))

        add_ctx(self.fk_volume_dsb, values=[0, -6, -12, -15, -18])
        add_ctx(self.fk_tuning_dsb, values=[-12, -6, 0, 6, 12])
        add_ctx(self.fk_cutoff_dsb, values=[10000, 5000, 2500, 1000, 500])

        self.fk_ADSR_pb.clicked.connect(self.fk_adsr_ctx)
        add_ctx(self.fk_A_dsb, values=[.001, .01, .02, .05, .1])
        add_ctx(self.fk_D_dsb, values=[.05, .1, .25, .375, .5])
        self.fk_S_dsb.setHidden(True)
        add_ctx(self.fk_R_dsb, values=[.05, .1, .25, .375, .5])

        self.fk_ADRr_pb.clicked.connect(self.fk_adr_ctx)
        add_ctx(self.fk_Ar_dsb, values=[-100, 0, 100], names=['log\t-100', 'lin\t0', 'exp\t100'])
        add_ctx(self.fk_Dr_dsb, values=[-100, 0, 100], names=['log\t-100', 'lin\t0', 'exp\t100'])
        add_ctx(self.fk_Rr_dsb, values=[-100, 0, 100], names=['log\t-100', 'lin\t0', 'exp\t100'])

        # Other options
        add_ctx(self.crossfade_dsb, values=[0, .05, .125, .25, .5, .75, 1])
        add_ctx(self.attenuation_dsb, values=[0, -3, -6, -9],
                names=['None\t0', 'Drums\t-3', 'Average\t-6', 'Polyphonic\t-9'])

        values = [0, .2, .5, .8, 1]
        names = list(values)
        names[0] = '0\tOrgan, Harpsichord'
        add_ctx(self.ampveltrk_dsb, values=values, names=names)
        add_ctx(self.notepan_dsb, values=[0, .2, .5, .8, 1])

        # UI cosmetics
        self.bg_text_cmb.currentTextChanged.connect(lambda state: self.bg_text_le.setEnabled(state == 'custom'))
        add_ctx(self.bg_text_le, values=[''], names=['Clear'])

        self.hsv_pb.clicked.connect(self.hsv_adjust_ctx)
        add_ctx(self.reverb_wet_dsb, values=[0, .2, .5, .8, 1])
        add_ctx(self.max_adsr_dsb, values=[0, 5, 10, 20])

        self.use_reverb_cb.stateChanged.connect(lambda state: self.reverb_wet_dsb.setEnabled(state))
        self.use_reverb_cb.stateChanged.connect(lambda state: self.use_ir_cb.setEnabled(state))

        # Process buttons
        self.add_suffix_cb.stateChanged.connect(lambda state: self.suffix_le.setEnabled(state))
        add_ctx(self.suffix_le, values=['', '_release', '_legato'])

        self.create_dsp_pb.clicked.connect(self.create_dspreset)
        self.create_dsp_pb.setFixedHeight(32)
        style_widget(self.create_dsp_pb, properties={'border-radius': 12})

        self.create_dslib_pb.clicked.connect(partial(self.as_worker, self.create_dslibrary))
        self.create_dslib_pb.setFixedHeight(32)
        style_widget(self.create_dslib_pb, properties={'background-color': 'rgb(95,95,95)', 'border-radius': 12})

        # Settings
        self.load_settings_a.triggered.connect(self.load_settings)
        self.save_settings_a.triggered.connect(self.save_settings)
        self.restore_defaults_a.triggered.connect(self.restore_defaults)
        self.get_defaults()

        # Help
        self.visit_github_a.triggered.connect(self.visit_github)
        self.about_a.triggered.connect(self.about_dialog)

    def as_worker(self, cmd):
        if not any(worker.running for worker in self.active_workers):
            self.worker = Worker(cmd)

            # Worker signals
            self.worker.signals.progress.connect(self.update_progress)
            self.worker.signals.progress_range.connect(self.update_range)
            self.worker.signals.message.connect(self.update_message)
            self.worker.signals.result.connect(self.handle_result)

            self.worker.signals.finished.connect(lambda: self.cleanup_worker(self.worker))

            self.active_workers.append(self.worker)
            self.threadpool.start(self.worker)
        else:
            print('Task is already running!')

    def setup_menu_bar(self):
        self.menu_bar = QMenuBar(self)
        self.menu_bar.setNativeMenuBar(False)

        plt = self.menu_bar.palette()
        plt.setColor(QtGui.QPalette.Background, QtGui.QColor(39, 39, 39))
        self.menu_bar.setPalette(plt)

        # Settings Menu
        self.settings_menu = QMenu(self.menu_bar)
        self.settings_menu.setTitle('Settings')

        self.save_settings_a = QAction(self)
        self.save_settings_a.setText('Save settings')
        self.load_settings_a = QAction(self)
        self.load_settings_a.setText('Load settings')
        self.restore_defaults_a = QAction(self)
        self.restore_defaults_a.setText('Restore defaults')

        self.settings_menu.addAction(self.load_settings_a)
        self.settings_menu.addAction(self.save_settings_a)
        self.settings_menu.addSeparator()
        self.settings_menu.addAction(self.restore_defaults_a)

        self.menu_bar.addAction(self.settings_menu.menuAction())

        # Help Menu
        self.help_menu = QMenu(self.menu_bar)
        self.help_menu.setTitle('?')
        self.visit_github_a = QAction(self)
        self.visit_github_a.setText('Visit repository on github')
        self.about_a = QAction(self)
        self.about_a.setText('About')

        self.help_menu.addAction(self.visit_github_a)
        self.help_menu.addAction(self.about_a)

        self.menu_bar.addAction(self.help_menu.menuAction())

        # Add menu bar
        self.setMenuBar(self.menu_bar)

    def get_options(self):
        self.options.root_dir = self.root_dir
        self.options.smp_attrib_cfg = self.smp_attrib_cfg

        self.options.add_suffix = (None, self.suffix_le.text())[self.add_suffix_cb.isChecked()]
        self.options.auto_increment = self.auto_incr_cb.isChecked()

        self.options.ir_subdir = (None, 'IR')[self.use_ir_cb.isChecked()]

        self.options.pattern = self.pattern_le.text()
        self.options.group_naming = self.groupnaming_cmb.currentText()

        self.options.override = self.override_cb.isChecked()
        self.options.loop = self.loop_cb.isChecked()

        self.options.amp_env_enabled = self.adsr_enable_cb.isChecked()
        self.options.adsr = self.A_dsb.value(), self.D_dsb.value(), self.S_dsb.value(), self.R_dsb.value()
        self.options.adr_curve = self.Ar_dsb.value(), self.Dr_dsb.value(), self.Rr_dsb.value()

        self.options.fake_legato = self.fake_leg_cb.isChecked()
        self.options.fk_leg_start = self.fk_leg_start_dsb.value()
        self.options.fk_leg_a = self.fk_leg_a_dsb.value()
        self.options.fk_leg_a_curve = self.fk_leg_a_curve_dsb.value()

        self.options.fake_release = self.fake_rls_cb.isChecked()
        self.options.fk_rls_mode = self.fk_rls_mode_cmb.currentText()
        self.options.fk_rls_volume = self.fk_volume_dsb.value()
        self.options.fk_rls_cutoff = self.fk_cutoff_dsb.value()
        self.options.fk_rls_tuning = self.fk_tuning_dsb.value()
        self.options.fk_rls_adsr = self.fk_A_dsb.value(), self.fk_D_dsb.value(), 0, self.fk_R_dsb.value()
        self.options.fk_rls_adr_curve = self.fk_Ar_dsb.value(), self.fk_Dr_dsb.value(), self.fk_Rr_dsb.value()

        self.options.transpose = self.transpose_sb.value()
        self.options.pf_mode = self.pf_mode_cmb.currentText()
        self.options.pf_th = self.pf_th_dsb.value()
        self.options.tuning = self.tuning_dsb.value()

        self.options.pad_vel = self.pad_vel_cb.isChecked()

        self.options.seq_mode = self.seq_mode_cmb.currentText()

        limit = self.limit_le.text()
        if limit.lower().startswith('auto'):
            limit = limit
        elif '-' in limit or '+' in limit:
            limit = [val for val in limit.split()]
        else:
            limit = [int(val) for val in limit.split()] or True
        self.options.limit = limit

        self.options.note_limit_mode = self.note_limit_cmb.currentText()

        rr_ofs = self.rrofs_le.text()
        if rr_ofs:
            rr_ofs = rr_ofs.strip(' ').lower()
            if rr_ofs[0] == 'x' and rr_ofs[1:].isdigit():
                rr_offset = int(rr_ofs[1:])
            else:
                rr_offset = [int(val) for val in self.rrofs_le.text().split()] or [0]
        else:
            rr_offset = None
        self.options.rr_offset = rr_offset

        self.options.rr_bounds = self.rr_bounds_cb.isChecked()

        self.options.crossfade_mode = self.crossfade_cmb.currentText()
        self.options.crossfade = self.crossfade_dsb.value()

        self.options.attenuation = self.attenuation_dsb.value()
        self.options.vel_track = self.ampveltrk_dsb.value()
        self.options.note_pan = self.notepan_dsb.value()

        self.options.monophonic = self.monophonic_cb.isChecked()

        self.options.note_spread = self.spread_cmb.currentText()

        bg_text_mode = self.bg_text_cmb.currentText()
        if bg_text_mode == 'root_dir':
            bg_text = beautify_str(Path(self.root_dir).stem)
        elif bg_text_mode == 'custom':
            bg_text = self.bg_text_le.text()
        else:
            bg_text = None
        self.options.bg_text = bg_text

        self.options.color_plt_cfg = self.palette_cfg[self.palette_cmb.currentText()]
        self.options.plt_adjust = [self.hue_dsb.value(), self.sat_dsb.value(), self.val_dsb.value()]

        font = resource_path(self.current_dir / 'HelveticaNeueThin.otf')
        self.options.text_font = (font, 24)

        self.options.group_knobs_rows = self.grp_knob_rows_sb.value()
        self.options.no_solo_grp_knob = self.no_solo_grp_knob_cb.isChecked()
        self.options.adsr_knobs = self.adsr_knobs_cb.isChecked()

        self.options.use_reverb = self.use_reverb_cb.isChecked()
        self.options.reverb_wet = self.reverb_wet_dsb.value()

        self.options.knob_scl = (1, 2)[self.hd_knobs_cb.isChecked()]

        self.options.max_adsr_knobs = self.max_adsr_dsb.value()

        self.options.multi_out = self.multi_out_cb.isChecked()

        self.options.estimate_delay = self.estimate_delay_cb.isChecked()

    def create_dspreset(self):
        if not self.root_dir:
            QMessageBox.information(self, 'Notification', 'Please set a valid root directory')
            return False

        add_suffix = (None, self.suffix_le.text())[self.add_suffix_cb.isChecked()]
        auto_increment = self.auto_incr_cb.isChecked()

        # Confirm overwriting
        basename = Path(self.root_dir).stem
        if add_suffix:
            basename += add_suffix
        filepath = Path.joinpath(Path(self.root_dir), f'{basename}.dspreset')

        if filepath.is_file() and not auto_increment:
            confirm_dlg = QMessageBox.question(self, 'Confirmation', f'{filepath.name} already exists\nOverwrite?',
                                               Qt.QMessageBox.Yes | Qt.QMessageBox.No, Qt.QMessageBox.No)
            if not confirm_dlg == Qt.QMessageBox.Yes:
                return False

        result = None
        importlib.reload(smp2ds)

        try:
            self.as_worker(self.create_dspreset_process)
            self.event_loop.exec_()  # Wait for result
            result = self.worker_result
            self.event_loop.quit()

            if result:
                # Write settings used to create the preset
                p = Path(result)
                settings_path = p.parent / f'{p.stem}.{self.settings_ext}'
                write_settings(widget=self, filepath=settings_path)

        except Exception as e:
            traceback.print_exc()
            self.update_progress(0)
            self.update_message(f'Error when processing: {e}')
            play_notification(audio_file=self.current_dir / 'process_error.flac')

        if result:
            play_notification(audio_file=self.current_dir / 'process_complete.flac')
        else:
            play_notification(audio_file=self.current_dir / 'process_error.flac')

    def create_dspreset_process(self, worker, progress_callback, message_callback):
        self.get_options()

        options = vars(self.options)
        func_args = inspect.getfullargspec(smp2ds.create_dspreset)[0]
        option_kwargs = {k: v for k, v in options.items() if k in func_args}

        result = smp2ds.create_dspreset(**option_kwargs,
                                        worker=worker, progress_callback=progress_callback,
                                        message_callback=message_callback)

        return result

    def create_dslibrary(self, worker, progress_callback, message_callback):
        if not self.root_dir:
            QMessageBox.information(self, 'Notification', 'Please set a valid root directory')
            return False

        result = smp2ds.create_dslibrary(self.root_dir)

        if result is None:
            progress_callback.emit(0)
            message_callback.emit('No dspreset file found')
        else:
            progress_callback.emit(100)
            message_callback.emit(f'{shorten_path(result, 30)} successfully created')
            play_notification(audio_file=self.current_dir / 'process_complete.flac')

    def set_rootdir(self):
        startdir = self.root_dir or os.getcwd()
        flags = QFileDialog.DontResolveSymlinks | QFileDialog.ShowDirsOnly
        path = QFileDialog.getExistingDirectory(self, "Select preset ROOT directory", startdir, flags)
        if path:
            self.root_dir = path
            self.path_l.setText(path)
            os.chdir(path)
            self.set_settings_path()
            self.load_settings_from_rootdir()

    def set_settings_path(self):
        if self.root_dir:
            p = Path(self.root_dir)
            self.settings_path = p / f'{p.stem}.{self.settings_ext}'
        else:
            p = Path(os.getcwd())
            self.settings_path = p / f'settings.{self.settings_ext}'

    def limit_ctx(self):
        names = [f'{k}\t{v}' if k != v and not k.lower().startswith('auto') else f'{k}' for k, v in
                 self.instr_range_cfg.items()]
        add_ctx(self.limit_le, values=list(self.instr_range_cfg.values()), names=names, trigger=self.limit_pb)

    def set_instr_range_cfg(self, cfg_file='instr_range_cfg.json'):
        self.instr_range_cfg = {'Samples Notes': '',
                                'Auto': 'auto',
                                'AutoX2': 'autox2',
                                '-3 +3': '-3 +3',
                                '-6 +6': '-6 +6',
                                'Full MIDI Range': '0 127'}
        if not Path(cfg_file).is_file():
            cfg_data = {'Acoustic Guitar': '40 88',
                        'French Horn': '34 77',
                        'Clarinet': '50 94',
                        'Harp': '24 103',
                        'Marimba': '45 96',
                        'Harpsichord': '29 89',
                        'Organ': '36 96',
                        'Piano': '21 108'}
        else:
            cfg_data = read_json(cfg_file)
        self.instr_range_cfg.update(
            {k: ' '.join([f'{s}' for s in v]) if isinstance(v, list) else f'{v}' for k, v in cfg_data.items()})

    def adsr_ctx(self):
        names = ['Sustained Short Release (Default, Organ)\t0.001 .25 1 .375',
                 'Looped Samples with Decay (Guitar, Keys)\t0.001 8 0 .5',
                 'Sustained Very Long Release (Bell, Percussion)\t0.001 1 1 20',
                 'Sustained Slow Attack/Release (Pad)\t1 1 1 1']
        values = [re.sub(r'[^0-9+\-.]', ' ', name).strip() for name in names]
        content = [{'type': 'cmds', 'name': name, 'cmd': partial(self.set_adsr, value)}
                   for name, value in zip(names, values)]
        popup_menu(content=content, parent=self.ADSR_pb)

    def adr_ctx(self):
        names = ['Log A,Exp DR (Default)\t-100 100 100',
                 'Log A,Lin D,Exp R (Looped end)\t-100 50 100',
                 'Log A,Lin D,Exp R\t-100 0 100',
                 'Log ADR (One shot, Bell, Drum)\t-100 -100 -100']
        values = [re.sub(r'[^0-9+\-.]', ' ', name).strip() for name in names]
        content = [{'type': 'cmds', 'name': name, 'cmd': partial(self.set_adr, value)}
                   for name, value in zip(names, values)]
        popup_menu(content=content, parent=self.ADRr_pb)

    def fk_adsr_ctx(self):
        names = ['Short (Default)\t0.001 .05 .05',
                 'Medium\t0.01 .1 .1',
                 'Soft\t0.025 .25 .25']
        values = [re.sub(r'[^0-9+\-.]', ' ', name).strip() for name in names]
        content = [{'type': 'cmds', 'name': name, 'cmd': partial(self.set_fk_adsr, value)}
                   for name, value in zip(names, values)]
        popup_menu(content=content, parent=self.fk_ADSR_pb)

    def fk_adr_ctx(self):
        names = ['Log A,Exp DR (Default)\t-100 100 100',
                 'Log A,Lin D,Exp R (Looped end)\t-100 50 100',
                 'Log A,Lin D,Exp R\t-100 0 100',
                 'Log ADR (One shot, Bell, Drum)\t-100 -100 -100']
        values = [re.sub(r'[^0-9+\-.]', ' ', name).strip() for name in names]
        content = [{'type': 'cmds', 'name': name, 'cmd': partial(self.set_fk_adr, value)}
                   for name, value in zip(names, values)]
        popup_menu(content=content, parent=self.fk_ADRr_pb)

    def set_adsr_wid_style(self, value: int):
        """
        Alter ADSR widgets style to indicate full or limited effect
        :param value:
        :return:
        """
        if value == 0:
            style = 'color: rgb(95, 127, 159);'
        else:
            style = ''
        self.adsr_wid.setStyleSheet(style)

    def set_adsr(self, value):
        values = [eval(v) for v in value.split()]
        widgets = [self.A_dsb, self.D_dsb, self.S_dsb, self.R_dsb]
        for wd, val in zip(widgets, values):
            wd.setValue(val)

    def set_adr(self, value):
        values = [eval(v) for v in value.split()]
        widgets = [self.Ar_dsb, self.Dr_dsb, self.Rr_dsb]
        for wd, val in zip(widgets, values):
            wd.setValue(val)

    def set_fk_adsr(self, value):
        values = [eval(v) for v in value.split()]
        widgets = [self.fk_A_dsb, self.fk_D_dsb, self.fk_R_dsb]
        for wd, val in zip(widgets, values):
            wd.setValue(val)

    def set_fk_adr(self, value):
        values = [eval(v) for v in value.split()]
        widgets = [self.fk_Ar_dsb, self.fk_Dr_dsb, self.fk_Rr_dsb]
        for wd, val in zip(widgets, values):
            wd.setValue(val)

    def populate_palette_cmb(self):
        plt_cfg_files = [str(item) for item in self.plt_cfg_dir.glob(f'*{self.plt_cfg_suffix}.json')]
        plt_names = [Path(item).stem.removesuffix(self.plt_cfg_suffix) for item in plt_cfg_files]
        self.palette_cfg = dict(zip(plt_names, plt_cfg_files))
        self.palette_cmb.addItems(plt_names)
        if self.default_plt in plt_names:
            self.palette_cmb.setCurrentIndex(plt_names.index(self.default_plt))

    def hsv_adjust_ctx(self):
        names = ['Reset (Default)\t0 1 1',
                 'Hue Invert\t.5 1 1',
                 'Desaturated\t0 .5 1',
                 'Gray\t0 0 1']
        values = [re.sub(r'[^0-9+\-.]', ' ', name).strip() for name in names]
        content = [{'type': 'cmds', 'name': name, 'cmd': partial(self.set_hsv, value)}
                   for name, value in zip(names, values)]
        popup_menu(content=content, parent=self.hsv_pb)

    def set_hsv(self, value):
        values = [eval(v) for v in value.split()]
        widgets = [self.hue_dsb, self.sat_dsb, self.val_dsb]
        for wd, val in zip(widgets, values):
            wd.setValue(val)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            file_path = event.mimeData().urls()[0].toLocalFile()
            if Path(file_path).is_dir():
                self.root_dir = file_path
                self.path_l.setText(file_path)
                os.chdir(file_path)
                self.set_settings_path()
                self.load_settings_from_rootdir()

    def load_settings(self):
        self.update_progress(0)
        p = Path(self.settings_path)
        if p.suffix == f'.{self.settings_ext}':
            p = p.parent
        result = read_settings(widget=self, filepath=None, startdir=p, ext=self.settings_ext)
        if result:
            os.chdir(result.parent)
            self.progress_pb.setFormat(f'{result.name} loaded')

    def save_settings(self):
        self.update_progress(0)
        result = write_settings(widget=self, filepath=None, startdir=self.settings_path, ext=self.settings_ext)
        if result:
            os.chdir(result.parent)
            self.progress_pb.setFormat(f'{result.name} saved')

    def get_defaults(self):
        get_settings(self, self.default_settings)

    def restore_defaults(self):
        self.update_progress(0)
        set_settings(widget=self, node=self.default_settings)
        self.progress_pb.setFormat(f'Default settings restored')

    def load_settings_from_rootdir(self):
        self.update_progress(0)
        self.set_settings_path()
        settings_files = [f for f in Path(self.root_dir).glob(f'*.{self.settings_ext}')]
        if settings_files:
            settings_files = sorted(settings_files, key=lambda f: os.path.getmtime(f))
            read_settings(widget=self, filepath=settings_files[-1], startdir=None, ext=None)
            self.progress_pb.setFormat(f'{settings_files[-1].name} found and loaded')
        else:
            self.progress_pb.setFormat(f'No settings found')

    def update_progress(self, value):
        self.progress_pb.setValue(value)

    def update_message(self, message):
        self.progress_pb.setFormat(message)

    def update_range(self, mn, mx):
        self.progress_pb.setRange(mn, mx)
        self.progress_pb.update()

    def handle_result(self, value):
        self.worker_result = value
        self.event_loop.quit()

    def cleanup_worker(self, worker):
        if worker in self.active_workers:
            self.active_workers.remove(worker)

    def closeEvent(self, event):
        print(f'{self.objectName()} closed')
        event.accept()

    def run(self):
        # Center on screen not on its parent
        parent = self.parent()
        if parent:
            screen = self.parent().screen()
        else:
            screen = self.screen()

        self.show()

        screen_geo = screen.geometry()
        x = screen_geo.x() + (screen_geo.width() - self.width()) // 2
        y = screen_geo.y() + (screen_geo.height() - self.height()) // 2
        self.move(x, y)

        return self

    def about_dialog(self):
        try:
            about_dlg = AboutDialog(parent=self)
            about_dlg.icon_file = self.icon_file
            about_dlg.title = f'About {self.tool_name}'
            about_dlg.text = f'{self.tool_name}<br>Version {self.tool_version}<br><br>'
            about_dlg.text += 'MIT License<br>'
            about_dlg.text += "Copyright © 2024 Michel 'Mitch' Pecqueur<br><br>"
            about_dlg.url = 'https://github.com/robotmitchum/sample_tools'
            about_dlg.setup_ui()
            about_dlg.exec_()
        except Exception as e:
            print(e)
            pass

    @staticmethod
    def visit_github():
        url = 'https://github.com/robotmitchum/sample_tools'
        qurl = QtCore.QUrl(url)
        QtGui.QDesktopServices.openUrl(qurl)


def run(mw=Smp2dsUi, parent=None):
    window = mw(parent=parent)
    return window.run()


if __name__ == "__main__":
    app_id = f'mitch.smp2Ds.{__version__}'
    if platform.system() == 'Windows':
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    app = QApplication(sys.argv)

    apply_dark_theme(app)

    font = app.font()
    font.setPointSize(11)
    app.setFont(font)

    run()

    sys.exit(app.exec_())
