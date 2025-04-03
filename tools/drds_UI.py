# coding:utf-8
"""
    :module: dr_ds_UI.py
    :description: DR-DS UI
    :author: Michel 'Mitch' Pecqueur
    :date: 2025.03
"""

import ctypes
import importlib
import inspect
import os
import platform
import re
import sys
import threading
import traceback
from functools import partial
from pathlib import Path

import numpy as np
import soundfile as sf
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import drums_to_dspreset as drds
import smp_to_dspreset as smp2ds
from audio_player import AudioPlayer, play_notification
from common_audio_utils import pitch_audio, balance_audio, db_to_lin
from common_math_utils import lerp, np_to_rgbint
from common_prefs_utils import get_settings, set_settings, read_settings, write_settings
from common_ui_utils import FilePathLabel, resource_path, get_custom_font, resource_path_alt
from common_ui_utils import Node, KeyPressHandler, shorten_path
from common_ui_utils import add_ctx, add_insert_ctx, shorten_str, beautify_str, popup_menu
from dark_fusion_style import apply_dark_theme
from drums_to_dspreset import __version__
from parseAttrString import parse_string
from sample_utils import Sample
from tools.worker import Worker
from utils import note_to_name, is_note_name
from waveform_widgets import WaveformDialog, LoopPointDialog

if getattr(sys, 'frozen', False):
    import pyi_splash  # noqa

    pyi_splash.close()

file_types = ['.wav', '.flac', '.aif']
file_types.extend([ext.upper() for ext in file_types])  # Also add uppercase variant


class DrDsUi(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        self.setObjectName('dr_ds_ui')
        self.setWindowTitle(f'DR-DS v{__version__}')

        self.cw = QWidget(self)
        self.cw.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(self.cw)
        self.lyt = QVBoxLayout()
        margin = 8
        self.lyt.setContentsMargins(margin, margin, margin, margin)
        self.lyt.setSpacing(0)
        self.cw.setLayout(self.lyt)

        self.setAttribute(Qt.WA_DeleteOnClose)
        self.options = Node()

        self.threadpool = QThreadPool(parent=self)
        self.worker = None
        self.active_workers = []
        self.worker_result = None
        self.event_loop = QEventLoop(parent=self)

        self.current_dir = Path(__file__).parent
        self.base_dir = self.current_dir.parent

        if getattr(sys, 'frozen', False):
            self.app_dir = Path(sys.executable).parent
        else:
            self.app_dir = self.base_dir
        os.chdir(self.app_dir)

        self.root_dir = None

        self.smp_attrib_cfg = resource_path_alt(self.base_dir / 'smp_attrib_cfg.json', parent_dir=self.app_dir)

        custom_font = get_custom_font(self.current_dir / 'RobotoMono-Medium.ttf')

        # lbl_size_policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        # lbl_size_policy.setHeightForWidth(True)
        self.drumpad_count = 32

        self.player = AudioPlayer()
        self.player.signals.message.connect(self.update_message)

        self.temp_audio = Node()
        self.playback_thread = None

        self.plt_cfg_dir = resource_path_alt(self.base_dir / 'plt_cfg', parent_dir=self.app_dir, as_str=False)
        text_font_path = resource_path(self.current_dir / 'HelveticaNeueThin.otf')
        self.text_font = (text_font_path, 24)

        self.plt_cfg_suffix = '_plt_cfg'
        self.default_plt = 'Dark'
        self.palette_cfg = {}

        self.setup_menu_bar()
        self.default_settings = Node()
        self.settings_ext = 'drds'
        self.settings_path = ''

        # Root dir widgets
        self.output_path_lyt = QHBoxLayout()
        self.output_path_lyt.setContentsMargins(0, 0, 0, 0)
        self.output_path_lyt.setSpacing(4)
        self.lyt.addLayout(self.output_path_lyt)

        self.rootdir_l = QLabel('Root Directory', parent=self.cw)
        self.rootdir_l.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred))
        self.output_path_lyt.addWidget(self.rootdir_l)

        self.output_path_l = FilePathLabel(parent=self.cw)
        self.output_path_l.setPlaceholderText('Drag and drop a preset ROOT directory here')
        self.output_path_l.setToolTip('Preset root directory')
        self.output_path_l.setMinimumHeight(40)
        self.output_path_lyt.addWidget(self.output_path_l)
        self.output_path_l.setStyleSheet(f'QLabel{{color: rgb(95,191,143);}}')
        font = self.output_path_l.font()
        font.setBold(True)
        self.output_path_l.setFont(font)

        self.output_path_tb = QToolButton(parent=self.cw)
        self.output_path_tb.setText('...')
        font = self.output_path_tb.font()
        font.setBold(True)
        self.output_path_tb.setFont(font)
        self.output_path_lyt.addWidget(self.output_path_tb)
        self.output_path_tb.clicked.connect(self.output_path_l.browse_path)
        self.output_path_tb.setToolTip('Browse root directory')
        self.output_path_l.pathChanged.connect(lambda _: setattr(self, 'root_dir', self.output_path_l.fullPath()))

        line = QFrame(parent=self.cw)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.lyt.addWidget(line)

        # - Mapping widgets -
        self.maping_lyt = QHBoxLayout()
        self.maping_lyt.setContentsMargins(0, 8, 0, 8)

        # Pattern widgets
        self.pattern_lyt = QHBoxLayout()
        self.pattern_lyt.setContentsMargins(0, 0, 0, 0)
        self.maping_lyt.addLayout(self.pattern_lyt)

        self.pattern_pb = QPushButton('Pattern', self.cw)
        self.pattern_lyt.addWidget(self.pattern_pb)
        values = ['', '{group}_{vel}_{seqPosition}', '{group}_{seqPosition}_{vel}',
                  '{_}-{group}_{vel}_{seqPosition}', '{_}-{group}_{seqPosition}_{vel}']
        self.pattern_le = QLineEdit(values[1], self.cw)
        self.pattern_le.setObjectName('pattern_le')
        self.pattern_pb.setMinimumWidth(96)
        self.pattern_pb.setMaximumWidth(96)
        self.pattern_pb.setToolTip(
            'Pattern used to convert sample names to attribute values (group, velocity, round-robin...)\n\n'
            'Also used to set Name and Label automatically for a drum pad'
            'Click for some pattern examples')
        self.pattern_le.setFrame(False)
        self.pattern_le.setToolTip(
            'Pattern used to convert sample names to attribute values (group, velocity, round-robin...)\n\n'
            'Attribute names must be enclosed with curly braces {}\n'
            'Right click to insert a supported attribute\n'
            'Use {_} to discard a part of the name')
        self.pattern_lyt.addWidget(self.pattern_le)
        self.lyt.addLayout(self.maping_lyt)
        add_ctx(self.pattern_le, values=values, trigger=self.pattern_pb)
        add_insert_ctx(self.pattern_le, values=['{group}', '{vel}', '{seqPosition}', '{_}'])

        # - Other mapping options
        self.r_mapping_lyt = QHBoxLayout()
        self.r_mapping_lyt.setContentsMargins(0, 0, 0, 0)
        self.r_mapping_lyt.setSpacing(4)

        self.maping_lyt.addLayout(self.r_mapping_lyt)
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.r_mapping_lyt.addItem(spacer)

        # Base note widgets
        self.r_mapping_lyt.addWidget(QLabel('Base Note', parent=self.cw))
        self.basenote_sb = QSpinBox(parent=self.cw)
        self.basenote_sb.setObjectName('basenote_sb')
        self.basenote_sb.setAlignment(Qt.AlignCenter)
        self.basenote_sb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.r_mapping_lyt.addWidget(self.basenote_sb)
        self.basenote_sb.setFixedWidth(48)
        self.basenote_sb.setValue(36)

        self.basenote_sb.setMinimum(0)
        self.basenote_sb.setMaximum(127 - self.drumpad_count)

        self.basenote_sb.setToolTip('Set the base note for the drumpad keyboard')

        add_ctx(self.basenote_sb, values=[0, 24, 36, 48], default_idx=2)

        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.r_mapping_lyt.addItem(spacer)

        # Seq mode widgets
        self.r_mapping_lyt.addWidget(QLabel('Seq Mode'))
        self.seq_mode_cmb = QComboBox()
        self.seq_mode_cmb.setObjectName('seq_mode_cmb')
        self.seq_mode_cmb.setToolTip('Round-robin mode\n\n'
                                     'Your samples must have a seqPosition index in their name\n\n'
                                     'For example:\nSnare_v127_seq1.flac\nSnare_v127_seq2.flac\nSnare_v127_seq3.flac')
        self.r_mapping_lyt.addWidget(self.seq_mode_cmb)
        self.seq_mode_cmb.addItems(['round_robin', 'random', 'true_random'])
        self.seq_mode_cmb.setCurrentIndex(1)
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.r_mapping_lyt.addItem(spacer)

        # Pads and properties widgets
        self.drum_pads = []
        self.drum_props = []

        self.drumprops_wid = QWidget(parent=self.cw)
        self.lyt.addWidget(self.drumprops_wid)
        self.drumprops_wid.setContentsMargins(0, 0, 0, 0)
        self.drumprops_lyt = QVBoxLayout()
        self.drumprops_lyt.setContentsMargins(0, 0, 0, 0)
        self.drumprops_wid.setLayout(self.drumprops_lyt)

        self.drumpads_wid = QWidget(parent=self.cw)
        self.lyt.addWidget(self.drumpads_wid)
        self.drumpads_wid.setContentsMargins(0, 4, 0, 4)
        self.drumpads_lyt = FlowLayout(margin=0, spacing=0)
        self.drumpads_wid.setLayout(self.drumpads_lyt)

        for i in range(self.drumpad_count):
            drum_prop = DrumpadProperties(parent=self.drumprops_wid, index=i, player=self.player, parent_window=self)
            drum_prop.setObjectName(f'drum_prop_{i + 1:02d}')
            self.output_path_l.pathChanged.connect(drum_prop.refresh_lw)

            self.drumprops_lyt.addWidget(drum_prop)
            self.drum_props.append(drum_prop)
            if i > 0:
                drum_prop.setVisible(False)
            drum_pad = DrumpadButton(parent=self.drumpads_wid, index=i, drum_prop=drum_prop)
            drum_pad.setObjectName(f'drum_pad_{i + 1:02d}')
            drum_prop.drum_pad = drum_pad

            drum_pad.parent_window = self
            drum_pad.drumpad_pb.clicked.connect(self.toggle_pad_display)

            drum_pad.drumpad_key_pb.clicked.connect(self.toggle_pad_display)
            self.drumpads_lyt.addWidget(drum_pad)
            self.drum_pads.append(drum_pad)

        self.current_pad = 0
        self.drum_pads[0].drumpad_pb.setChecked(True)

        self.basenote_sb.valueChanged.connect(self.update_drumpads)

        # - Output options -

        line = QFrame(self.cw)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.lyt.addWidget(line)

        # - Background Options -

        self.bg_lyt = QHBoxLayout()
        self.bg_lyt.setContentsMargins(4, 4, 4, 4)
        self.bg_lyt.setSpacing(4)
        self.lyt.addLayout(self.bg_lyt)

        # Background text
        self.bg_text_lyt = QHBoxLayout()
        self.bg_text_lyt.setContentsMargins(0, 0, 0, 0)
        self.bg_text_lyt.setSpacing(4)
        self.bg_text_lyt.addWidget(QLabel('Background Text', self.cw))
        self.bg_text_cmb = QComboBox(self.cw)
        self.bg_text_cmb.setObjectName('bg_text_cmb')
        self.bg_text_cmb.setToolTip('Write text to the background image\n'
                                    'none:\tNo text\n'
                                    'rootdir:\tUse root directory name as text (default)\n'
                                    'custom:\tType a custom text')
        self.bg_text_cmb.addItems(['none', 'root_dir', 'custom'])
        self.bg_text_lyt.addWidget(self.bg_text_cmb)
        self.bg_text_le = QLineEdit(self.cw)
        self.bg_text_le.setObjectName('bg_text_le')
        self.bg_text_le.setPlaceholderText('Drum Kit Name')
        add_ctx(self.bg_text_le, values=[''], names=['Clear'])
        self.bg_text_cmb.currentTextChanged.connect(lambda state: self.bg_text_le.setEnabled(state == 'custom'))
        self.bg_text_cmb.setCurrentIndex(1)
        self.bg_text_lyt.addWidget(self.bg_text_le)
        self.bg_lyt.addLayout(self.bg_text_lyt)

        self.bg_lyt.addSpacing(40)

        # Background palette
        self.bg_plt_lyt = QHBoxLayout()
        self.bg_plt_lyt.setContentsMargins(0, 0, 0, 0)
        self.bg_plt_lyt.setSpacing(4)
        self.bg_plt_lyt.addWidget(QLabel('Palette'))
        self.palette_cmb = QComboBox(self.cw)
        self.palette_cmb.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))
        self.palette_cmb.setObjectName('palette_cmb')
        self.palette_cmb.setToolTip('Color Palette used to generate the dspreset UI and its background image')
        self.bg_plt_lyt.addWidget(self.palette_cmb)
        self.hsv_pb = QPushButton('HSV Adjust', self.cw)
        self.hsv_pb.setToolTip('Globally adjust chosen color palette')
        self.bg_plt_lyt.addWidget(self.hsv_pb)

        self.populate_palette_cmb()

        self.hsv_widgets = []
        for name, value, mx in zip(['Hue', 'Saturation', 'Value'], [0, 1, 1], [1, 2, 2]):
            dsb = QDoubleSpinBox(self.cw)
            dsb.setObjectName(f'{name.lower()[:3]}_dsb')
            dsb.setToolTip(name)
            dsb.setMaximum(mx)
            dsb.setDecimals(3)
            dsb.setValue(value)
            dsb.setSingleStep(.1)
            dsb.setFrame(False)
            dsb.setAlignment(Qt.AlignCenter)
            dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
            dsb.setContextMenuPolicy(Qt.NoContextMenu)
            self.hsv_widgets.append(dsb)
            self.bg_plt_lyt.addWidget(dsb)
        self.bg_lyt.addLayout(self.bg_plt_lyt)

        self.hsv_pb.clicked.connect(self.hsv_adjust_ctx)

        # Knobs widgets
        self.knobs_lyt = QHBoxLayout()
        self.knobs_lyt.setContentsMargins(0, 4, 0, 4)
        self.knobs_lyt.setSpacing(8)
        self.lyt.addLayout(self.knobs_lyt)

        # self.tvp_cb_lyt = QHBoxLayout()
        # self.tvp_cb_lyt.setContentsMargins(0, 4, 0, 4)
        # self.tvp_cb_lyt.setSpacing(8)
        # self.knobs_lyt.addLayout(self.tvp_cb_lyt)

        self.knob_cb_widgets = []
        for name, checked in zip(['Tuning', 'Volume', 'Pan'], [0, 1, 1]):
            wid = QCheckBox(f'{name} Knobs', parent=self.cw)
            wid.setObjectName(f'add_{name.lower()}_cb')
            wid.setToolTip(f'Add {name.lower()} knobs to generated UI')
            # wid.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            wid.setChecked(checked)
            self.knob_cb_widgets.append(wid)
            self.knobs_lyt.addWidget(wid)

        # Reverb effect widgets
        self.reverb_lyt = QHBoxLayout()
        self.reverb_lyt.setContentsMargins(0, 4, 0, 4)
        self.reverb_lyt.setSpacing(4)
        self.knobs_lyt.addLayout(self.reverb_lyt)

        self.reverb_cb = QCheckBox('Reverb', self.cw)
        # self.reverb_cb.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.reverb_cb.setObjectName('reverb_cb')
        self.reverb_cb.setToolTip('Add reverb effect')
        self.reverb_lyt.addWidget(self.reverb_cb)

        self.reverb_wet_dsb = QDoubleSpinBox(self.cw)
        self.reverb_wet_dsb.setObjectName('reverb_wet_dsb')
        self.reverb_wet_dsb.setAlignment(Qt.AlignCenter)
        # self.reverb_wet_dsb.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.reverb_wet_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.reverb_wet_dsb.setFrame(False)
        self.reverb_wet_dsb.setMaximum(1.0)
        self.reverb_wet_dsb.setSingleStep(.1)
        self.reverb_wet_dsb.setValue(.2)
        self.reverb_wet_dsb.setToolTip('Default wet value for reverb')
        self.reverb_cb.stateChanged.connect(lambda state: self.reverb_wet_dsb.setEnabled(state))
        add_ctx(self.reverb_wet_dsb, values=[0, .2, .5, .8, 1])
        self.reverb_lyt.addWidget(self.reverb_wet_dsb)

        self.use_ir_cb = QCheckBox('Use IR', self.cw)
        self.use_ir_cb.setObjectName('use_ir_cb')
        # self.use_ir_cb.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.use_ir_cb.setToolTip('Use or ignore IR samples\n'
                                  'IR samples must be located in a separate subdirectory called \'IR\'')
        self.reverb_cb.stateChanged.connect(lambda state: self.use_ir_cb.setEnabled(state))
        self.reverb_cb.setChecked(True)
        self.reverb_lyt.addWidget(self.use_ir_cb)

        # File options
        self.output_options_lyt = QHBoxLayout()
        self.output_options_lyt.setContentsMargins(0, 0, 0, 0)
        self.output_options_lyt.setSpacing(4)
        self.lyt.addLayout(self.output_options_lyt)

        # Suffix widgets
        self.suffix_lyt = QHBoxLayout()
        self.suffix_lyt.setContentsMargins(0, 4, 0, 4)
        self.suffix_lyt.setSpacing(4)
        self.output_options_lyt.addLayout(self.suffix_lyt)

        self.add_suffix_cb = QCheckBox('Add Suffix', parent=self.cw)
        self.add_suffix_cb.setObjectName('add_suffix_cb')
        self.suffix_lyt.addWidget(self.add_suffix_cb)

        self.suffix_le = QLineEdit(parent=self.cw)
        self.suffix_le.setObjectName('suffix_le')
        self.suffix_le.setToolTip('Add custom suffix to disambiguate a preset variant')
        self.suffix_le.setEnabled(False)
        self.suffix_le.setPlaceholderText('_suffix')
        add_ctx(self.suffix_le, values=[''])
        self.add_suffix_cb.stateChanged.connect(lambda state: self.suffix_le.setEnabled(state))
        self.suffix_lyt.addWidget(self.suffix_le)

        self.auto_incr_cb = QCheckBox('Auto Increment', parent=self.cw)
        self.auto_incr_cb.setObjectName('auto_incr_cb')
        self.auto_incr_cb.setToolTip('Auto-increment to avoid overwriting')

        self.suffix_lyt.addWidget(self.auto_incr_cb)

        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.output_options_lyt.addItem(spacer)

        # - Process buttons -
        self.buttons_lyt = QHBoxLayout()
        self.buttons_lyt.setContentsMargins(0, 4, 0, 4)
        self.lyt.addLayout(self.buttons_lyt)

        font.setBold(True)
        font.setPointSize(12)

        self.dspreset_pb = QPushButton('Create dspreset', parent=self.cw)
        self.dspreset_pb.setToolTip('Create a dspreset file in root directory and from current settings')
        self.dspreset_pb.setStyleSheet(f'QPushButton{{background-color: rgb(63,127,95);border-radius:12;}}')
        self.dspreset_pb.setFixedSize(256, 32)
        self.buttons_lyt.addWidget(self.dspreset_pb)
        self.dspreset_pb.setFont(font)
        self.dspreset_pb.clicked.connect(self.create_dspreset)

        self.dslibrary_pb = QPushButton('Create dslibrary', parent=self.cw)
        self.dslibrary_pb.setToolTip('Create dslibrary from root directory by archiving only required files\n'
                                     'At least one valid dspreset must exist in the directory')
        self.dslibrary_pb.setStyleSheet(f'QPushButton{{border-radius:12;}}')
        self.dslibrary_pb.setFixedSize(256, 32)
        self.buttons_lyt.addWidget(self.dslibrary_pb)
        self.dslibrary_pb.setFont(font)
        self.dslibrary_pb.clicked.connect(partial(self.as_worker, self.create_dslibrary))

        # Progress Bar
        self.progress_pb = QProgressBar(parent=self.cw)
        self.lyt.addWidget(self.progress_pb)
        self.progress_pb.setStyleSheet('QProgressBar{border: none;}')
        self.progress_pb.setAlignment(Qt.AlignCenter)
        self.progress_pb.setTextVisible(True)
        self.progress_pb.setFont(custom_font)
        self.progress_pb.setValue(0)
        self.update_message('Create a Decent Sampler drum preset from samples')

        app_icon = QIcon()
        img_file = resource_path(self.current_dir / 'UI/icons/drds_64.png')
        app_icon.addFile(img_file, QSize(64, 64))
        self.setWindowIcon(app_icon)

        self.setFixedSize(1024 + margin * 2, 768)

        # Settings
        self.load_settings_a.triggered.connect(self.load_settings)
        self.save_settings_a.triggered.connect(self.save_settings)
        self.restore_defaults_a.triggered.connect(self.restore_defaults)
        self.get_defaults()

        self.output_path_l.pathChanged.connect(self.load_settings_from_rootdir)

    def update_drumpads(self):
        base_note = self.basenote_sb.value()
        for pad in self.drum_pads:
            note = pad.pad_idx + base_note
            pad.update_note(note)

    def setup_menu_bar(self):
        self.menu_bar = QMenuBar(parent=self)
        self.menu_bar.setNativeMenuBar(False)

        plt = self.menu_bar.palette()
        plt.setColor(QPalette.Background, QColor(39, 39, 39))
        self.menu_bar.setPalette(plt)

        self.settings_menu = QMenu(self.menu_bar)
        self.settings_menu.setTitle('Settings')
        self.setMenuBar(self.menu_bar)

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

    def get_options(self):
        self.options.smp_attrib_cfg = self.smp_attrib_cfg

        self.options.root_dir = self.output_path_l.fullPath()
        self.options.bg_text = None

        self.options.pattern = self.pattern_le.text()
        self.options.seq_mode = self.seq_mode_cmb.currentText()

        font = resource_path(self.current_dir / 'HelveticaNeueThin.otf')
        self.options.text_font = (font, 24)

        # Drum pads data
        data = {'groups': []}

        active_drums = [drum_prop for drum_prop in self.drum_props if drum_prop.is_active()]
        active_number = {item.pad_idx + 1: i for i, item in enumerate(active_drums)}

        for drum_prop in active_drums:
            choke = [active_number.get(item, None) for item in drum_prop.choke_cpd.value()]
            choke = [c for c in choke if c is not None]
            group = {
                'name': drum_prop.name(),
                'label': drum_prop.label(),
                'samples': drum_prop.get_samples(),
                'note': drum_prop.note(),
                'choke': choke,
                'tuning': drum_prop.tuning_dsb.value(),
                'volume': drum_prop.volume_dsb.value(),
                'pan': drum_prop.pan_dsb.value()
            }
            data['groups'].append(group)
        self.options.data = data

        attrs = ['tuning_knobs', 'volume_knobs', 'pan_knobs']
        for wid, attr in zip(self.knob_cb_widgets, attrs):
            setattr(self.options, attr, wid.isChecked())

        # self.options.attenuation = self.attenuation_dsb.value()
        # self.options.vel_track = self.ampveltrk_dsb.value()

        match self.bg_text_cmb.currentText():
            case 'root_dir':
                bg_text = beautify_str(Path(self.root_dir).stem)
            case 'custom':
                bg_text = self.bg_text_le.text()
            case _:
                bg_text = None
        self.options.bg_text = bg_text

        self.options.add_suffix = (None, self.suffix_le.text())[self.add_suffix_cb.isChecked()]
        self.options.auto_increment = self.auto_incr_cb.isChecked()

        self.options.use_reverb = self.reverb_cb.isChecked()
        self.options.reverb_wet = self.reverb_wet_dsb.value()
        self.options.ir_subdir = (None, 'IR')[self.use_ir_cb.isChecked()]

        self.options.color_plt_cfg = self.palette_cfg[self.palette_cmb.currentText()]
        self.options.plt_adjust = [wid.value() for wid in self.hsv_widgets]

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

    def create_dslibrary(self, worker, progress_callback, message_callback):
        self.root_dir = self.output_path_l.fullPath()

        if not self.root_dir:
            return False

        result = smp2ds.create_dslibrary(self.root_dir)

        if result is None:
            progress_callback.emit(0)
            message_callback.emit('No dspreset file found')
        else:
            progress_callback.emit(100)
            message_callback.emit(f'{shorten_path(result, 30)} successfully created')
            play_notification(audio_file=self.current_dir / 'process_complete.flac')

    def create_dspreset(self):
        self.root_dir = self.output_path_l.fullPath()

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
                                               QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if not confirm_dlg == QMessageBox.Yes:
                return False

        result = None
        importlib.reload(drds)

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

    def create_dspreset_process(self, worker, progress_callback, message_callback):
        self.get_options()

        options = vars(self.options)
        func_args = inspect.getfullargspec(drds.create_drums_dspreset)[0]
        option_kwargs = {k: v for k, v in options.items() if k in func_args}

        print(option_kwargs)

        result = drds.create_drums_dspreset(**option_kwargs,
                                            worker=worker, progress_callback=progress_callback,
                                            message_callback=message_callback)

        print(result)

        return result

    def toggle_pad_display(self):
        clicked_pad = self.sender().parent()
        if self.current_pad != clicked_pad.pad_idx:
            for i, (pad, prop) in enumerate(zip(self.drum_pads, self.drum_props)):
                value = bool(pad is clicked_pad)
                pad.drumpad_pb.setChecked(value)
                prop.setVisible(value)
                if value:
                    self.current_pad = i
        else:
            self.drum_pads[self.current_pad].drumpad_pb.setChecked(True)

    def play_stop_toggle(self):
        if self.player.is_playing.is_set():
            self.player.stop()
        else:
            files_lw = self.drum_props[self.current_pad].files_lw
            items = files_lw.selectedItems()
            if not items and files_lw.count() > 0:
                items = [files_lw.item(0)]
            if items:
                audio_file = Path(items[0].data(Qt.UserRole))
                if audio_file.is_file():
                    data, sr = sf.read(audio_file)
                    info = Sample(audio_file)
                    self.playback_thread = threading.Thread(target=self.player.play,
                                                            args=(data, sr, info.loopStart, info.loopEnd), daemon=True)
                    self.playback_thread.start()

    def play_sample(self, audio_file):
        self.player.stop()
        if Path(audio_file).is_file():
            # data, sr = sf.read(audio_file)
            data, sr = self.read_sample(audio_file, use_settings=True)
            self.playback_thread = threading.Thread(target=self.player.play, args=(data, sr, None, None), daemon=True)
            self.playback_thread.start()

    def read_sample(self, audio_file, use_settings=True, tol=1e-3):
        data, sr = sf.read(audio_file)
        if use_settings:
            if abs(self.player.tuning) > tol:
                data = pitch_audio(data, self.player.tuning)
            if abs(self.player.pan) > tol:
                data = balance_audio(data, self.player.pan)
            if abs(self.player.volume - 1) > tol:
                data *= self.player.volume
        return data, sr

    def set_settings_path(self):
        p = Path(os.getcwd())
        output_path = self.output_path_l.fullPath()
        if output_path:
            p = Path(output_path)
        self.settings_path = p / f'settings.{self.settings_ext}'

    def load_settings(self):
        self.update_progress(0)
        p = Path(self.settings_path)
        if p.suffix == f'.{self.settings_ext}':
            p = p.parent
        result = read_settings(widget=self, filepath=None, startdir=p, ext=self.settings_ext)
        if result:
            os.chdir(result.parent)
            self.update_message(f'{result.name} loaded')

    def save_settings(self):
        self.update_progress(0)
        self.set_settings_path()
        result = write_settings(widget=self, filepath=None, startdir=self.settings_path, ext=self.settings_ext)
        if result:
            os.chdir(result.parent)
            self.update_message(f'{result.name} saved')

    def get_defaults(self):
        get_settings(self, self.default_settings)

    def restore_defaults(self):
        self.update_progress(0)
        set_settings(widget=self, node=self.default_settings)
        self.update_message('Default settings restored')

    def load_settings_from_rootdir(self):
        self.update_progress(0)
        self.set_settings_path()
        settings_files = [f for f in Path(self.output_path_l.fullPath()).glob(f'*.{self.settings_ext}')]
        if settings_files:
            settings_files = sorted(settings_files, key=lambda f: os.path.getmtime(f))
            read_settings(widget=self, filepath=settings_files[-1], startdir=None, ext=None)
            self.update_message(f'{settings_files[-1].name} found and loaded')
        else:
            self.update_message('No settings found')

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

    def cleanup_drum_prop(self):
        for item in self.drum_props:
            item.cleanup_lw()

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
        for wid, val in zip(self.hsv_widgets, values):
            wid.setValue(val)

    def closeEvent(self, event):
        self.player.stop()
        self.removeEventFilter(self)
        self.cleanup_drum_prop()
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


# Custom widgets

class AudioFilesLw(QListWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.current_dir = Path(__file__).parent
        self.base_dir = self.current_dir.parent
        self.root_dir = None
        self.player = None

        self.file_types = file_types

        # Set a mono font to the list widget to simplify white spaces handling
        custom_font = get_custom_font(self.current_dir / 'RobotoMono-Medium.ttf')
        custom_font.setPointSize(11)
        self.setFont(custom_font)
        self.setUniformItemSizes(True)

        self.setContextMenuPolicy(3)
        self.customContextMenuRequested.connect(self.files_lw_ctx)
        self.doubleClicked.connect(self.play_lw_item)

        self.setAcceptDrops(True)

        self.last_dir = None

    def browse_files(self):
        self.refresh_lw_items()
        if not self.last_dir:
            items = self.selectedItems() or self.get_lw_items()
            items = [s.data(Qt.UserRole) for s in items]
            if items:
                self.last_dir = Path(items[-1]).parent

        if self.last_dir:
            startdir = str(self.last_dir)
        else:
            startdir = os.getcwd()

        fmts = [f'*{fmt}' for fmt in self.file_types]
        fltr = 'Audio Files ({});;All Files (*)'.format(' '.join(fmts))
        new_files, _ = QFileDialog.getOpenFileNames(self, "Select audio files", startdir, fltr)

        if new_files:
            files = self.get_lw_items()
            files.extend(new_files)
            self.add_lw_items(files)

    def files_lw_ctx(self):
        menu = QMenu(self)
        names = [menu.addAction(item) for item in
                 ['Display Waveform\tW', 'Remove item(s) from list\tDel', 'Clear list']]
        menu.insertSeparator(menu.actions()[2])
        cmds = [self.waveform_dlg, self.del_lw_items, self.clear]
        action = menu.exec_(QCursor.pos())
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
        items = [self.item(i).data(Qt.UserRole) for i in range(self.count())]
        return list(filter(None, items))

    def get_sel_lw_items(self):
        items = [item.data(Qt.UserRole) for item in self.selectedItems()]
        return list(filter(None, items))

    def current_lw_item(self):
        if not self.count():
            return None
        items = self.get_sel_lw_items() or self.get_lw_items()
        return items[0]

    def get_prefs(self):
        result = []
        for item in self.get_lw_items():
            p = Path(item).resolve()
            if self.root_dir:
                if p.is_relative_to(self.root_dir):
                    p = p.relative_to(self.root_dir)
            result.append(p.as_posix())
        return result

    def set_prefs(self, value):
        files = []
        for item in value:
            p = Path(item)
            if not p.is_file() and self.root_dir:
                p = Path(self.root_dir) / item
            if p.is_file():
                files.append(str(p.resolve()))
            else:
                print(f'{item} missing')
        self.add_lw_items(files)

    def del_lw_items(self):
        for item in self.selectedItems():
            self.takeItem(self.row(item))

    def add_lw_items(self, files):
        files = [str(Path(f)) for f in files]
        files = list(dict.fromkeys(files))
        names = [simplify_path(f, root_dir=self.root_dir) for f in files]

        self.clear()
        self.addItems(names)

        for i, file_path in enumerate(files):
            self.item(i).setData(Qt.UserRole, file_path)

        if files:
            self.last_dir = Path(files[-1]).parent

    def refresh_lw_items(self):
        lw_items = [self.item(i) for i in range(self.count())]
        for i, item in enumerate(lw_items):
            p = Path(item.data(Qt.UserRole)).resolve()
            if p.is_file():
                self.item(i).setData(Qt.UserRole, p)
                item.setText(simplify_path(p, root_dir=self.root_dir))
            else:
                self.takeItem(self.row(item))
        self.update()

    def play_lw_item(self, *args):
        self.player.stop()
        audio_file = args[0].data(Qt.UserRole)
        if Path(audio_file).is_file():
            data, sr = sf.read(audio_file)
            self.playback_thread = threading.Thread(target=self.player.play, args=(data, sr, None, None), daemon=True)
            self.playback_thread.start()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            items = self.selectedItems()
            for item in items:
                self.takeItem(self.row(item))

        if event.key() == Qt.Key_W:
            self.waveform_dlg()
        # if event.key() == Qt.Key_L:
        #     self.loop_point_dlg()

        if event.key() == Qt.Key_Down:
            mx = self.count() - 1
            sel_indices = [a.row() + 1 if a.row() < mx else mx for a in self.selectedIndexes()]
            self.clearSelection()
            for idx in sel_indices:
                self.item(idx).setSelected(True)
        elif event.key() == Qt.Key_Up:
            sel_indices = [a.row() - 1 if a.row() > 0 else 0 for a in self.selectedIndexes()]
            self.clearSelection()
            for idx in sel_indices:
                self.item(idx).setSelected(True)

        elif event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_A:
                self.selectAll()
            elif event.key() == Qt.Key_I:
                items = self.selectedItems()
                self.selectAll()
                for item in items:
                    item.setSelected(False)
        else:
            super().keyPressEvent(event)

    def dropEvent(self, event):
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

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.LinkAction)
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()


class DrumpadProperties(QWidget):
    def __init__(self, parent, index, player, parent_window=None):
        super().__init__(parent=parent)
        self.lyt = QVBoxLayout()
        self.lyt.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.lyt)

        self.pad_idx = index
        self.drum_pad = None
        self.parent_window = parent_window
        self.root_dir = ''

        lbl_size_policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        lbl_size_policy.setHeightForWidth(True)

        # Drum Properties Title
        self.title_l = QLabel(f'Drum Pad {self.pad_idx + 1:02d}', self)
        font = self.title_l.font()
        font.setPointSize(16)
        font.setBold(True)
        self.title_l.setFont(font)
        self.title_l.setStyleSheet(f'QLabel{{color: rgb(95,191,143);}}')

        self.lyt.addWidget(self.title_l)

        self.h_lyt = QHBoxLayout()
        self.lyt.addLayout(self.h_lyt)

        self.files_lyt = QHBoxLayout()
        self.files_lyt.setContentsMargins(0, 0, 0, 0)
        self.h_lyt.addLayout(self.files_lyt)

        # - Files widget -
        self.files_lw = AudioFilesLw(parent=self)
        self.files_lw.setObjectName(f'files_lw_{index + 1:02d}')
        self.files_lw.setToolTip('Drop audio files for current drum pad')
        self.files_lyt.addWidget(self.files_lw)
        self.files_tb = QToolButton(parent=self)
        self.files_tb.setText('...')
        self.files_tb.setToolTip('Browse audio files for current drum pad')
        self.files_lyt.addWidget(self.files_tb)

        self.files_tb.clicked.connect(self.files_lw.browse_files)

        self.files_lw.player = player

        # - Attribute widgets -
        self.attrib_lyt = QVBoxLayout()
        self.attrib_lyt.setContentsMargins(8, 0, 8, 0)
        self.h_lyt.addLayout(self.attrib_lyt)

        # Name Widgets
        self.name_dict = {'BassDrum': 'BD', 'Kick': 'K', 'Stick': 'ST', 'Snare': 'SD', 'Clap': 'CP',
                          'ClosedHiHat': 'CH', 'PedalHiHat': 'PH', 'OpenedHiHat': 'OH',
                          'LoTom': 'LT', 'MidTom': 'MT', 'HiTom': 'HT',
                          'Crash': 'CR', 'Ride': 'RC', 'Splash': 'SP',
                          'Tambourine': 'TM', 'Cowbell': 'CB',
                          'LoConga': 'LC', 'MidConga': 'MC', 'HiConga': 'HC',
                          'Cabasa': 'CA', 'Claves': 'CL', 'Shaker': 'SH'}

        drum_names = list(self.name_dict.keys())
        drum_names.insert(0, '')

        drum_labels = list(self.name_dict.values())
        drum_labels.insert(0, '')

        self.name_lyt = QHBoxLayout()
        self.name_lyt.setContentsMargins(4, 0, 4, 0)
        self.name_lyt.setSpacing(8)
        self.attrib_lyt.addLayout(self.name_lyt)

        self.name_lyt.addWidget(QLabel('Name', self))
        self.name_le = QLineEdit(self)
        self.name_le.setObjectName(f'name_le_{index + 1:02d}')
        add_ctx(self.name_le, values=drum_names, alignment=Qt.AlignCenter)
        self.name_le.setFrame(False)
        self.name_le.setToolTip('Name used for group and tool tips in generated UI')
        self.name_le.setAlignment(Qt.AlignCenter)
        self.name_lyt.addWidget(self.name_le)

        # Spacer
        self.name_lyt.addSpacing(32)

        # Label
        self.name_lyt.addWidget(QLabel('Label', self))
        self.label_le = QLineEdit(self)
        add_ctx(self.label_le, values=drum_labels, alignment=Qt.AlignCenter)
        self.label_le.setObjectName(f'label_le_{index + 1:02d}')
        self.label_le.setFrame(False)
        self.label_le.setAlignment(Qt.AlignCenter)
        self.label_le.setToolTip('Shortened label to identify knobs in the generated UI')
        self.label_le.setFixedWidth(64)
        self.name_lyt.addWidget(self.label_le)

        # - Attribute Widgets -
        self.grid_lyt = QGridLayout()
        self.grid_lyt.setContentsMargins(0, 0, 0, 0)
        self.attrib_lyt.addLayout(self.grid_lyt)

        # Tuning
        self.tuning_l = QLabel('Tuning', self)
        self.tuning_l.setSizePolicy(lbl_size_policy)
        self.tuning_l.setAlignment(Qt.AlignCenter)
        self.grid_lyt.addWidget(self.tuning_l, 0, 0, 1, 1)

        self.tuning_dsb = QDoubleSpinBox(self)
        add_ctx(self.tuning_dsb, values=[-12, -6, 0, 6, 12], alignment=Qt.AlignCenter)
        self.tuning_dsb.setObjectName(f'tuning_dsb_{index + 1:02d}')
        self.tuning_dsb.setFrame(False)
        self.tuning_dsb.setAlignment(Qt.AlignCenter)
        self.tuning_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.tuning_dsb.setDecimals(3)
        self.tuning_dsb.setMinimum(-36)
        self.tuning_dsb.setMaximum(36)
        self.tuning_dsb.setSingleStep(.5)
        self.tuning_dsb.setFixedWidth(96)
        self.grid_lyt.addWidget(self.tuning_dsb, 1, 0, 1, 1)

        # Volume
        self.volume_l = QLabel('Volume', self)
        self.volume_l.setSizePolicy(lbl_size_policy)
        self.volume_l.setAlignment(Qt.AlignCenter)
        self.grid_lyt.addWidget(self.volume_l, 0, 1, 1, 1)

        self.volume_dsb = QDoubleSpinBox(self)
        add_ctx(self.volume_dsb, values=[.25, .5, 1], alignment=Qt.AlignCenter)
        self.volume_dsb.setObjectName(f'volume_dsb_{index + 1:02d}')
        self.volume_dsb.setFrame(False)
        self.volume_dsb.setAlignment(Qt.AlignCenter)
        self.volume_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.volume_dsb.setMinimum(0)
        self.volume_dsb.setMaximum(1)
        self.volume_dsb.setValue(1)
        self.volume_dsb.setSingleStep(.05)
        self.volume_dsb.setFixedWidth(96)
        self.grid_lyt.addWidget(self.volume_dsb, 1, 1, 1, 1)

        # Pan
        self.pan_l = QLabel('Pan', self)
        self.pan_l.setSizePolicy(lbl_size_policy)
        self.pan_l.setAlignment(Qt.AlignCenter)
        self.grid_lyt.addWidget(self.pan_l, 0, 2, 1, 1)

        self.pan_dsb = QDoubleSpinBox(self)
        self.pan_dsb.setObjectName(f'pan_dsb_{index + 1:02d}')
        add_ctx(self.pan_dsb, values=[-75, -50, -25, 0, 25, 50, 75], alignment=Qt.AlignCenter)
        self.pan_dsb.setFrame(False)
        self.pan_dsb.setAlignment(Qt.AlignCenter)
        self.pan_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.pan_dsb.setMinimum(-100)
        self.pan_dsb.setMaximum(100)
        self.pan_dsb.setFixedWidth(96)
        self.grid_lyt.addWidget(self.pan_dsb, 1, 2, 1, 1)

        # - Choke widgets -
        self.choke_lyt = QHBoxLayout()
        self.choke_lyt.setContentsMargins(4, 0, 4, 0)
        self.choke_lyt.setSpacing(8)
        self.choke_l = QLabel('Choke', self)
        lbl_size_policy.setHeightForWidth(self.choke_l.sizePolicy().hasHeightForWidth())
        self.choke_l.setSizePolicy(lbl_size_policy)
        self.choke_lyt.addWidget(self.choke_l)

        drumpad_count = self.parent_window.drumpad_count
        choke_items = {k: v for k, v in zip(range(1, drumpad_count + 1), [False] * (drumpad_count + 1))}
        self.choke_cpd = CheckPullDown(parent=self, items=choke_items)

        self.choke_cpd.setObjectName(f'choke_cpd_{index + 1:02d}')
        self.choke_cpd.setToolTip('Pick drum pad(s) silencing the current drum pad')
        self.choke_lyt.addWidget(self.choke_cpd)

        self.self_choke_cb = QCheckBox('Self', self)
        self.self_choke_cb.setObjectName(f'self_choke_cb_{index + 1:02d}')
        self.self_choke_cb.setToolTip('The current drum pad will silence itself if checked (default)')
        self.choke_lyt.addWidget(self.self_choke_cb)

        self.attrib_lyt.addLayout(self.choke_lyt)

        # Function Aliases
        self.get_samples = self.files_lw.get_lw_items
        self.current_sample = self.files_lw.current_lw_item

        # Connect updates
        self.self_choke_cb.stateChanged.connect(self.choke_cb_update)
        self.self_choke_cb.setChecked(True)

        self.files_lw.model().dataChanged.connect(partial(self.update_pad, clear_pad=False))
        self.files_lw.model().rowsRemoved.connect(partial(self.update_pad, clear_pad=False))
        self.files_lw.model().modelReset.connect(partial(self.update_pad, clear_pad=True))
        self.name_le.textChanged.connect(partial(self.update_pad, clear_pad=False))
        self.label_le.textChanged.connect(partial(self.update_pad, clear_pad=False))

    def cleanup_lw(self):
        self.files_lw = None

    def update_pad(self, clear_pad=False):
        if self.files_lw is None:
            return
        if clear_pad or self.name() == '':
            self.set_name_from_samples()
        if clear_pad or self.label() == '':
            self.set_label_from_name()
        self.drum_pad.update_pad(self.label())

    def choke_cb_update(self):
        self.choke_cpd.items[self.pad_idx + 1] = int(self.self_choke_cb.isChecked())
        self.choke_cpd.update_text()

    def is_active(self):
        return self.files_lw.count() > 0

    def name(self):
        """
        :return:
        :rtype: str
        """
        return self.name_le.text()

    def set_name(self, value):
        self.name_le.setText(value)

    def label(self):
        return self.label_le.text()

    def set_label(self, value):
        self.label_le.setText(value)

    def note(self):
        return self.parent_window.basenote_sb.value() + self.pad_idx

    def set_name_from_samples(self):
        name = ''
        samples = self.get_samples()
        if samples:
            if samples[0]:
                name = Path(self.current_sample()).stem
                pattern = self.parent_window.pattern_le.text()
                res = parse_string(name=name, pattern=pattern)
                name = res.get('group', name).replace('-', '_')
                name = beautify_str(name).replace(' ', '')
        self.set_name(name)

    def set_label_from_name(self):
        name = self.name().replace('-', '_')
        nl_dict = {k.lower(): v for k, v in self.name_dict.items()}
        label = nl_dict.get(name.lower(), shorten_str(name).upper())
        self.set_label(label)

    def clear_pad(self):
        self.files_lw.clear()

    def refresh_lw(self):
        self.root_dir = self.parent_window.output_path_l.fullPath()
        self.files_lw.root_dir = self.root_dir
        self.files_lw.refresh_lw_items()
        self.files_lw.last_dir = self.root_dir

    def set_player(self):
        attrs = ['tuning', 'pan', 'volume']
        attenuation = -3
        values = [self.tuning_dsb.value(),
                  self.pan_dsb.value() / 100,
                  self.volume_dsb.value() * db_to_lin(attenuation)]
        for attr, value in zip(attrs, values):
            setattr(self.parent_window.player, attr, value)


class CheckPullDown(QPushButton):
    """
    Button displaying a pull-down list with checkable items
    """

    def __init__(self, parent=None, items=None):
        """
        :param parent:
        :param dict items: Items to display in the pull-down list as a dict
        items with a non-bool value can't be toggled
        """
        super().__init__(parent=parent)
        size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        size_policy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(size_policy)

        self.items = dict()
        self.set_items(items)

        self.current_dir = Path(__file__).parent
        font = get_custom_font(self.current_dir / 'RobotoMono-Medium.ttf')
        self.setFont(font)

        self.clicked.connect(self.pulldown_menu)
        self.ctx_menu()

        self.update_text()

    def toggle_item(self, key):
        value = self.items[key]
        if isinstance(value, bool):
            self.items[key] = not value
        self.update_text()

    def clear_items(self):
        self.items = {k: (v, False)[bool(v is True)] for k, v in self.items.items()}
        self.update_text()

    def get_items(self):
        return self.items

    def set_items(self, items):
        self.items = items or dict()
        self.update_text()

    def value(self):
        result = [k for k, v in self.items.items() if bool(v)]
        return result

    def get_prefs(self):
        result = [k for k, v in self.items.items() if bool(v) and isinstance(v, bool)]
        return result

    def set_prefs(self, value):
        self.items = {k: (v, k in value)[isinstance(v, bool)] for k, v in self.items.items()}
        self.update_text()

    def update_text(self):
        text = ' '.join([str(k) for k, v in self.items.items() if bool(v)])
        self.setText(text)

    def pulldown_menu(self):
        menu = QMenu(self)
        for key, value in self.items.items():
            check = (' ', '•')[bool(value > 0)]
            action_label = QLabel(f' {key:02d}{check}', parent=self)
            action_label.setFont(self.font())
            action_label.setAlignment(Qt.AlignCenter)
            action_label.setAttribute(Qt.WA_Hover, True)
            if not isinstance(value, bool):
                action_label.setEnabled(False)
            action_label.setMouseTracking(True)
            plt = self.palette()
            style = (f'QLabel {{background-color: {plt.alternateBase().color().name()}; '
                     f'color: {plt.text().color().name()};}}')
            style += (f'QLabel:hover {{background-color: {plt.highlight().color().name()}; '
                      f'color: {plt.highlightedText().color().name()};}}')
            action_label.setStyleSheet(style)
            action = QWidgetAction(self)
            action.setDefaultWidget(action_label)

            action.triggered.connect(lambda _, v=key: self.toggle_item(v))
            menu.addAction(action)

        pos = self.mapToGlobal(self.contentsRect().bottomLeft())
        menu.setMinimumWidth(self.width())
        menu.exec_(pos)

    def ctx_menu(self):
        def show_context_menu():
            names = ['Clear Items']
            cmds = [self.clear_items]

            menu = QMenu(self)

            for name, cmd in zip(names, cmds):
                action_label = QLabel(name, self)
                action_label.setAlignment(Qt.AlignCenter)
                action_label.setAttribute(Qt.WA_Hover, True)
                action_label.setMouseTracking(True)
                plt = self.palette()
                style = (f'QLabel {{background-color: {plt.alternateBase().color().name()}; '
                         f'color: {plt.text().color().name()};}}')
                style += (f'QLabel:hover {{background-color: {plt.highlight().color().name()}; '
                          f'color: {plt.highlightedText().color().name()};}}')
                action_label.setStyleSheet(style)
                action = QWidgetAction(self)
                action.setDefaultWidget(action_label)

                action.triggered.connect(cmd)
                menu.addAction(action)

            pos = self.mapToGlobal(self.contentsRect().bottomLeft())
            menu.setMinimumWidth(self.width())
            menu.exec_(pos)

        self.setContextMenuPolicy(3)
        self.customContextMenuRequested.connect(show_context_menu)


class DrumpadButton(QWidget):
    def __init__(self, parent, index, drum_prop):
        super().__init__(parent=parent)
        self.setFixedSize(64, 96)
        self.setContentsMargins(0, 0, 0, 0)
        self.parent_window = None

        self.file_types = file_types

        self.pad_idx = index
        self.pad_num = f'{index + 1:02d}'
        self.drum_prop = drum_prop

        self.note = 36
        self.note_name = 'C2'

        self.lyt = QVBoxLayout()
        self.lyt.setContentsMargins(4, 4, 4, 4)
        self.lyt.setSpacing(0)
        self.setLayout(self.lyt)

        self.drumpad_pb = QPushButton(parent=self)
        self.drumpad_pb.setCheckable(True)
        self.drumpad_pb.setFixedSize(56, 56)
        font = QFont()
        font.setPointSize(10)
        self.drumpad_pb.setFont(font)
        self.lyt.addWidget(self.drumpad_pb)

        self.drumpad_key_pb = QPushButton(parent=self)
        self.drumpad_key_pb.setFixedSize(56, 24)
        font = QFont()
        font.setPointSize(8)
        self.drumpad_key_pb.setFont(font)
        self.lyt.addWidget(self.drumpad_key_pb)

        self.update_pad('')
        self.update_note(drum_prop.note())

        self.drumpad_pb.clicked.connect(self.play)
        self.ctx_menu()
        self.setAcceptDrops(True)

    def update_pad(self, value=None):
        active = self.drum_prop.is_active()

        # Pad label
        text = (f'\n{self.pad_num}', f'{value}\n{self.pad_num}')[bool(value) and active]

        font = self.drumpad_pb.font()
        font.setBold(active)
        self.drumpad_pb.setFont(font)
        self.drumpad_pb.setText(text)

        # Pad tool tip
        tool_tip = 'Drop audio file(s) to this pad'
        if active:
            tool_tip = self.drum_prop.name() + ':\n'
            smps = self.drum_prop.get_samples()
            tool_tip += '\n'.join([Path(s).name for s in smps])
            tool_tip += '\n\nClick to play with simulated tuning, volume, pan settings'
        self.drumpad_pb.setToolTip(tool_tip)

        # Pad color
        bgc = (.2, .8)[active]
        bgc = np.array([bgc, bgc, bgc])

        col = 1 - bgc
        chk = lerp(bgc, np.array([0.25, 1.0, 0.75]), .5)
        hvc = lerp(bgc, 1.0, .25)
        chk_hvc = lerp(chk, 1.0, .25)

        style = f'QPushButton{{background-color:rgb{np_to_rgbint(bgc)};}}'
        style += f'QPushButton{{color:rgb{np_to_rgbint(col)};}}'
        style += f'QPushButton::checked{{background-color:rgb{np_to_rgbint(chk)};}}'
        style += f'QPushButton::hover{{background-color:rgb{np_to_rgbint(hvc)};}}'
        style += f'QPushButton::checked:hover{{background-color:rgb{np_to_rgbint(chk_hvc)};}}'
        style += f'QPushButton{{border-radius:12;}}'

        self.drumpad_pb.setStyleSheet(style)

    def update_note(self, value):
        if is_note_name(str(value)):
            self.note_name = str(value).upper()
        else:
            self.note = value
            n, o = note_to_name(value)
            self.note_name = f'{n}{o}'

        self.drumpad_key_pb.setText(f'{value:03d} - {self.note_name}')

        bgc = (.667, .25)['#' in self.note_name]
        bgc = np.array([bgc, bgc, bgc])

        col = 1 - bgc
        chk = lerp(bgc, np.array([.125, .5, 1.0]), .5)
        hvc = lerp(bgc, 1.0, .25)
        chk_hvc = lerp(chk, 1.0, .25)

        style = f'QPushButton{{background-color:rgb{np_to_rgbint(bgc)};}}'
        style += f'QPushButton{{color:rgb{np_to_rgbint(col)};}}'
        style += f'QPushButton::checked{{background-color:rgb{np_to_rgbint(chk)};}}'
        style += f'QPushButton::hover{{background-color:rgb{np_to_rgbint(hvc)};}}'
        style += f'QPushButton::checked:hover{{background-color:rgb{np_to_rgbint(chk_hvc)};}}'
        style += f'QPushButton{{border-radius:4;}}'

        self.drumpad_key_pb.setStyleSheet(style)

    def play(self):
        smp = self.drum_prop.current_sample()
        if smp and self.parent_window:
            self.drum_prop.set_player()
            self.parent_window.play_sample(smp)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            items = event.mimeData().urls()
            items = [Path(item.toLocalFile()) for item in items]

            files = [item for item in items if item.is_file() and item.suffix in self.file_types]
            dirs = [item for item in items if item.is_dir()]

            for d in dirs:
                for ext in self.file_types:
                    files.extend(Path(d).glob(f'*{ext}'))

            self.drum_prop.files_lw.add_lw_items(files)
        else:
            event.ignore()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.LinkAction)
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def ctx_menu(self):
        def show_context_menu():
            menu = QMenu(self)
            names = ['Clear Drum Pad']
            cmds = [self.drum_prop.clear_pad]

            for name, cmd in zip(names, cmds):
                action = QAction(f'{name}', self)
                action.triggered.connect(cmd)
                menu.addAction(action)

            pos = QCursor.pos()
            menu.exec_(pos)

        self.setContextMenuPolicy(3)
        self.customContextMenuRequested.connect(show_context_menu)


class FlowLayout(QLayout):
    def __init__(self, margin=0, spacing=-1):
        super().__init__()
        self.itemList = []
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)

    def addItem(self, item):
        self.itemList.append(item)

    def count(self):
        return len(self.itemList)

    def itemAt(self, index):
        return self.itemList[index] if 0 <= index < len(self.itemList) else None

    def takeAt(self, index):
        return self.itemList.pop(index) if 0 <= index < len(self.itemList) else None

    def expandingDirections(self):
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self.do_layout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self.do_layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self.itemList:
            size = size.expandedTo(item.minimumSize())
        size += QSize(2 * self.contentsMargins().top(), 2 * self.contentsMargins().top())
        return size

    def do_layout(self, rect, test_only):
        x, y = rect.x(), rect.y()
        line_height = 0
        rect_right = x + rect.width()

        for item in self.itemList:
            spacing = self.spacing()
            next_x = x + item.sizeHint().width() + spacing

            if next_x - spacing > rect_right and line_height > 0:
                x = rect.x()
                y += line_height + spacing
                next_x = x + item.sizeHint().width() + spacing
                line_height = 0

            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = next_x
            line_height = max(line_height, item.sizeHint().height())

        return y + line_height - rect.y()


def simplify_path(file_path, root_dir=None, max_length=40):
    p = Path(file_path)
    if root_dir and p.is_relative_to(root_dir):
        p = p.relative_to(root_dir)
    return shorten_path(str(p.as_posix()), max_length=max_length)


def run(mw=DrDsUi, parent=None):
    window = mw(parent=parent)
    return window.run()


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

    app = QApplication(sys.argv)
    apply_dark_theme(app)

    font = app.font()
    font.setPointSize(11)
    app.setFont(font)

    window = mw()
    key_press_handler = KeyPressHandler(window)
    app.installEventFilter(key_press_handler)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    launch(mw=DrDsUi, app_id=f'mitch.DrDs.{__version__}')
