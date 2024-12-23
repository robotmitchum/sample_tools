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
import platform
import os
import re
import sys
import traceback
from functools import partial
from pathlib import Path

import qdarkstyle
from PyQt5 import QtGui, QtCore, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox

from UI import smp_to_ds as gui
import smp_to_dspreset as smp2ds
from audio_player import play_notification
from common_ui_utils import add_ctx, add_insert_ctx, popup_menu
from common_ui_utils import beautify_str, resource_path, resource_path_alt, shorten_path
from jsonFile import read_json
from smp_to_dspreset import __version__

from common_prefs_utils import Node, get_settings, set_settings, read_settings, write_settings


class Smp2dsUi(gui.Ui_smp_to_ds_ui, QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setupUi(self)
        self.setWindowTitle(f'SMP2ds v{__version__}')
        self.setAttribute(Qt.Qt.WA_DeleteOnClose)

        self.root_dir = None  # Instrument root directory
        self.ir_subdir = None

        self.setAcceptDrops(True)

        self.current_dir = Path(__file__).parent
        self.base_dir = self.current_dir.parent

        self.smp_attrib_cfg = resource_path_alt(self.base_dir / 'smp_attrib_cfg.json', parent_dir='')

        self.pattern_le.setText('{group}_{note}_{vel}')
        self.limit_le.setText('')
        self.override_cb.setChecked(False)

        self.instr_range_cfg = {}
        self.instr_range_cfg_path = resource_path_alt(self.base_dir / 'instr_range_cfg.json', parent_dir='')
        self.set_instr_range_cfg(cfg_file=self.instr_range_cfg_path)

        self.limit_le.setText('autox2')
        self.set_adsr('.001 .25 1.0 .375')
        self.set_adr('-100 100 100')
        self.max_adsr_dsb.setValue(10)

        self.attenuation_dsb.setValue(-9)
        self.fk_volume_dsb.setValue(-15)

        self.plt_cfg_dir = resource_path_alt(self.base_dir / 'plt_cfg', parent_dir='', as_str=False)
        text_font_path = resource_path(self.current_dir / 'HelveticaNeueThin.otf')
        self.text_font = (text_font_path, 24)

        self.plt_cfg_suffix = '_plt_cfg'
        self.default_plt = 'Dark'
        self.palette_cfg = {}
        self.populate_palette_cmb()

        self.default_settings = Node()
        self.settings_ext = 'smp2ds'
        self.settings_path = None
        self.set_settings_path()

        self.setup_connections()

        self.crossfade_cmb.setCurrentText('linear')

        app_icon = QtGui.QIcon()
        img_file = resource_path(self.current_dir / 'UI/icons/smp2ds_64.png')
        app_icon.addFile(img_file, QtCore.QSize(64, 64))
        self.setWindowIcon(app_icon)

        self.progress_pb.setTextVisible(True)
        self.progress_pb.setFormat('Create a Decent Sampler preset from samples')

        # Init defaults settings
        get_settings(self, self.default_settings)

    def setup_connections(self):
        self.setpath_tb.clicked.connect(self.set_rootdir)

        # Pattern widgets
        add_ctx(self.pattern_le,
                values=['{group}_{note}', '{group}_{note}_{vel}', '{group}_{note}_{seqPosition}',
                        '{group}_{note}_{trigger}'],
                trigger=self.pattern_pb)
        smp_attrib = ['group', 'note', 'pitchFraction', 'vel', 'trigger', 'seqPosition']
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
        self.create_dslib_pb.clicked.connect(self.create_dslibrary)

        # Settings
        self.load_settings_a.triggered.connect(self.load_settings)
        self.save_settings_a.triggered.connect(self.save_settings)
        self.restore_defaults_a.triggered.connect(self.restore_defaults)

    def create_dspreset(self):
        if not self.root_dir:
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

        self.ir_subdir = (None, 'IR')[self.use_ir_cb.isChecked()]

        pattern = self.pattern_le.text()
        groupnaming = self.groupnaming_cmb.currentText()

        override = self.override_cb.isChecked()
        loop = self.loop_cb.isChecked()

        ADSR = self.A_dsb.value(), self.D_dsb.value(), self.S_dsb.value(), self.R_dsb.value()
        ADRr = self.Ar_dsb.value(), self.Dr_dsb.value(), self.Rr_dsb.value()

        fake_legato = self.fake_leg_cb.isChecked()
        fk_leg_start = self.fk_leg_start_dsb.value()
        fk_leg_a = self.fk_leg_a_dsb.value()
        fk_leg_a_curve = self.fk_leg_a_curve_dsb.value()

        fake_release = self.fake_rls_cb.isChecked()
        fk_rls_mode = self.fk_rls_mode_cmb.currentText()
        fk_rls_volume = self.fk_volume_dsb.value()
        fk_rls_cutoff = self.fk_cutoff_dsb.value()
        fk_rls_tuning = self.fk_tuning_dsb.value()
        fk_rls_ADSR = self.fk_A_dsb.value(), self.fk_D_dsb.value(), 0, self.fk_R_dsb.value()
        fk_rls_ADRr = self.fk_Ar_dsb.value(), self.fk_Dr_dsb.value(), self.fk_Rr_dsb.value()

        transpose = self.transpose_sb.value()
        pf_mode = self.pf_mode_cmb.currentText()
        pf_th = self.pf_th_dsb.value()
        tuning = self.tuning_dsb.value()

        pad_vel = self.pad_vel_cb.isChecked()

        seq_mode = self.seq_mode_cmb.currentText()

        limit = self.limit_le.text()
        if limit.lower().startswith('auto'):
            limit = limit
        elif '-' in limit or '+' in limit:
            limit = [val for val in limit.split()]
        else:
            limit = [int(val) for val in limit.split()] or True

        note_limit_mode = self.note_limit_cmb.currentText()

        rr_ofs = self.rrofs_le.text()
        if rr_ofs:
            rr_ofs = rr_ofs.strip(' ').lower()
            if rr_ofs[0] == 'x' and rr_ofs[1:].isdigit():
                rr_offset = int(rr_ofs[1:])
            else:
                rr_offset = [int(val) for val in self.rrofs_le.text().split()] or [0]
        else:
            rr_offset = None

        rr_bounds = self.rr_bounds_cb.isChecked()

        crossfade_mode = self.crossfade_cmb.currentText()
        crossfade = self.crossfade_dsb.value()

        attenuation = self.attenuation_dsb.value()
        vel_track = self.ampveltrk_dsb.value()
        note_pan = self.notepan_dsb.value()

        monophonic = self.monophonic_cb.isChecked()

        note_spread = self.spread_cmb.currentText()

        bg_text_mode = self.bg_text_cmb.currentText()
        if bg_text_mode == 'root_dir':
            bg_text = beautify_str(Path(self.root_dir).stem)
        elif bg_text_mode == 'custom':
            bg_text = self.bg_text_le.text()
        else:
            bg_text = None

        palette_cfg = self.palette_cfg[self.palette_cmb.currentText()]
        hsv_adjust = [self.hue_dsb.value(), self.sat_dsb.value(), self.val_dsb.value()]

        group_knobs_rows = self.grp_knob_rows_sb.value()
        no_solo_grp_knob = self.no_solo_grp_knob_cb.isChecked()
        adsr_knobs = self.adsr_knobs_cb.isChecked()

        use_reverb = self.use_reverb_cb.isChecked()
        reverb_wet = self.reverb_wet_dsb.value()

        max_adsr_knobs = self.max_adsr_dsb.value()

        result = None
        # importlib.reload(smp2ds)
        try:
            result = smp2ds.create_dspreset(root_dir=self.root_dir,

                                            smp_attrib_cfg=self.smp_attrib_cfg,

                                            pattern=pattern, group_naming=groupnaming,
                                            override=override, loop=loop,
                                            adsr=ADSR, adr_curve=ADRr,

                                            fake_legato=fake_legato,
                                            fk_leg_start=fk_leg_start, fk_leg_a=fk_leg_a, fk_leg_a_curve=fk_leg_a_curve,

                                            fake_release=fake_release, fk_rls_mode=fk_rls_mode,
                                            fk_rls_volume=fk_rls_volume, fk_rls_tuning=fk_rls_tuning,
                                            fk_rls_cutoff=fk_rls_cutoff,
                                            fk_rls_adsr=fk_rls_ADSR, fk_rls_adr_curve=fk_rls_ADRr,

                                            note_spread=note_spread, seq_mode=seq_mode,
                                            limit=limit, note_limit_mode=note_limit_mode,
                                            rr_offset=rr_offset, rr_bounds=rr_bounds,
                                            transpose=transpose, pf_mode=pf_mode, pf_th=pf_th, tuning=tuning,
                                            pad_vel=pad_vel,

                                            crossfade_mode=crossfade_mode, crossfade=crossfade,

                                            attenuation=attenuation, vel_track=vel_track, note_pan=note_pan,
                                            monophonic=monophonic,

                                            bg_text=bg_text, text_font=self.text_font,

                                            color_plt_cfg=palette_cfg, plt_adjust=hsv_adjust,
                                            group_knobs_rows=group_knobs_rows, no_solo_grp_knob=no_solo_grp_knob,
                                            adsr_knobs=adsr_knobs, max_adsr_knobs=max_adsr_knobs,
                                            use_reverb=use_reverb, reverb_wet=reverb_wet, ir_subdir=self.ir_subdir,

                                            add_suffix=add_suffix, auto_increment=auto_increment,

                                            progress=self.progress_pb)
        except Exception as e:
            print(e)
            traceback.print_exc()
            self.progress_pb.setFormat('Error when processing. Please check settings.')
            play_notification(audio_file=self.current_dir / 'process_error.flac')

        if result:
            play_notification(audio_file=self.current_dir / 'process_complete.flac')

    def create_dslibrary(self):
        if not self.root_dir:
            return False

        result = smp2ds.create_dslibrary(self.root_dir)

        if result is None:
            self.progress_pb.setFormat('No dspreset file found')
        else:
            self.progress_pb.setFormat(f'{shorten_path(result, 30)} successfully created')
            play_notification(audio_file=self.current_dir / 'process_complete.flac')

    def set_rootdir(self):
        startdir = self.root_dir or os.getcwd()
        flags = QFileDialog.DontResolveSymlinks | QFileDialog.ShowDirsOnly
        path = QFileDialog.getExistingDirectory(self, "Select preset ROOT directory", startdir, flags)
        if path:
            self.root_dir = path
            self.path_l.setText(path)
        self.set_settings_path()

    def set_settings_path(self):
        if self.root_dir:
            p = Path(self.root_dir)
            self.settings_path = p / f'{p.stem}.{self.settings_ext}'
        else:
            self.settings_path = self.current_dir / f'settings.{self.settings_ext}'

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
            self.set_settings_path()

    def load_settings(self):
        p = Path(self.settings_path)
        if p.suffix == f'.{self.settings_ext}':
            p = p.parent
        read_settings(widget=self, filepath=None, startdir=p, ext=self.settings_ext)

    def save_settings(self):
        write_settings(widget=self, filepath=None, startdir=self.settings_path, ext=self.settings_ext)

    def restore_defaults(self):
        set_settings(widget=self, node=self.default_settings)

    def closeEvent(self, event):
        print(f'{self.objectName()} closed')
        event.accept()

    def run(self):
        parent = self.parent()
        if parent:
            screen = self.parent().screen()
        else:
            screen = self.screen()

        # Center on screen not on its parent
        screen_geo = screen.geometry()
        x = screen_geo.x() + (screen_geo.width() - self.width()) // 2
        y = screen_geo.y() + (screen_geo.height() - self.height()) // 2
        self.move(x, y)

        self.show()
        return self


def run(mw=Smp2dsUi, parent=None):
    window = mw(parent=parent)
    return window.run()


if __name__ == "__main__":
    app_id = f'mitch.smp2Ds.{__version__}'
    if platform.system() == 'Windows':
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
    font = app.font()
    font.setPointSize(11)
    app.setFont(font)

    run()

    sys.exit(app.exec_())
