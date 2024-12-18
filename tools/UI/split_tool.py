# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'split_tool.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_split_tool_mw(object):
    def setupUi(self, split_tool_mw):
        split_tool_mw.setObjectName("split_tool_mw")
        split_tool_mw.resize(640, 600)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setStrikeOut(False)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        split_tool_mw.setFont(font)
        split_tool_mw.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.centralwidget = QtWidgets.QWidget(split_tool_mw)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.files_title_l = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.files_title_l.sizePolicy().hasHeightForWidth())
        self.files_title_l.setSizePolicy(sizePolicy)
        self.files_title_l.setMinimumSize(QtCore.QSize(64, 0))
        self.files_title_l.setStyleSheet("background-color: rgb(127, 63, 95);\n"
"color: rgb(255, 255, 255);")
        self.files_title_l.setObjectName("files_title_l")
        self.verticalLayout.addWidget(self.files_title_l)
        self.input_lyt = QtWidgets.QHBoxLayout()
        self.input_lyt.setObjectName("input_lyt")
        self.files_lw = QtWidgets.QListWidget(self.centralwidget)
        self.files_lw.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.files_lw.setDragDropMode(QtWidgets.QAbstractItemView.DropOnly)
        self.files_lw.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.files_lw.setObjectName("files_lw")
        self.input_lyt.addWidget(self.files_lw)
        self.set_files_tb = QtWidgets.QToolButton(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.set_files_tb.setFont(font)
        self.set_files_tb.setObjectName("set_files_tb")
        self.input_lyt.addWidget(self.set_files_tb)
        self.verticalLayout.addLayout(self.input_lyt)
        self.options_title_l = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.options_title_l.sizePolicy().hasHeightForWidth())
        self.options_title_l.setSizePolicy(sizePolicy)
        self.options_title_l.setMinimumSize(QtCore.QSize(64, 0))
        self.options_title_l.setStyleSheet("background-color: rgb(127, 63, 95);\n"
"color: rgb(255, 255, 255);")
        self.options_title_l.setObjectName("options_title_l")
        self.verticalLayout.addWidget(self.options_title_l)
        self.naming_lyt = QtWidgets.QHBoxLayout()
        self.naming_lyt.setObjectName("naming_lyt")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.naming_lyt.addWidget(self.label_3)
        self.basename_le = QtWidgets.QLineEdit(self.centralwidget)
        self.basename_le.setFrame(False)
        self.basename_le.setObjectName("basename_le")
        self.naming_lyt.addWidget(self.basename_le)
        self.suffix_l = QtWidgets.QLabel(self.centralwidget)
        self.suffix_l.setObjectName("suffix_l")
        self.naming_lyt.addWidget(self.suffix_l)
        self.suffix_mode_cmb = QtWidgets.QComboBox(self.centralwidget)
        self.suffix_mode_cmb.setObjectName("suffix_mode_cmb")
        self.suffix_mode_cmb.addItem("")
        self.suffix_mode_cmb.addItem("")
        self.suffix_mode_cmb.addItem("")
        self.suffix_mode_cmb.addItem("")
        self.naming_lyt.addWidget(self.suffix_mode_cmb)
        self.suffix_le = QtWidgets.QLineEdit(self.centralwidget)
        self.suffix_le.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.suffix_le.sizePolicy().hasHeightForWidth())
        self.suffix_le.setSizePolicy(sizePolicy)
        self.suffix_le.setFrame(False)
        self.suffix_le.setObjectName("suffix_le")
        self.naming_lyt.addWidget(self.suffix_le)
        self.verticalLayout.addLayout(self.naming_lyt)
        self.pitch_mode_wid = QtWidgets.QWidget(self.centralwidget)
        self.pitch_mode_wid.setEnabled(False)
        self.pitch_mode_wid.setObjectName("pitch_mode_wid")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.pitch_mode_wid)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.pitch_mode_l = QtWidgets.QLabel(self.pitch_mode_wid)
        self.pitch_mode_l.setObjectName("pitch_mode_l")
        self.horizontalLayout.addWidget(self.pitch_mode_l)
        self.pitch_mode_cmb = QtWidgets.QComboBox(self.pitch_mode_wid)
        self.pitch_mode_cmb.setObjectName("pitch_mode_cmb")
        self.pitch_mode_cmb.addItem("")
        self.pitch_mode_cmb.addItem("")
        self.pitch_mode_cmb.addItem("")
        self.horizontalLayout.addWidget(self.pitch_mode_cmb)
        self.use_pitch_fraction_cb = QtWidgets.QCheckBox(self.pitch_mode_wid)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.use_pitch_fraction_cb.sizePolicy().hasHeightForWidth())
        self.use_pitch_fraction_cb.setSizePolicy(sizePolicy)
        self.use_pitch_fraction_cb.setObjectName("use_pitch_fraction_cb")
        self.horizontalLayout.addWidget(self.use_pitch_fraction_cb)
        self.verticalLayout.addWidget(self.pitch_mode_wid)
        self.extra_suffix_lyt = QtWidgets.QHBoxLayout()
        self.extra_suffix_lyt.setObjectName("extra_suffix_lyt")
        self.extra_suffix_cb = QtWidgets.QCheckBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.extra_suffix_cb.sizePolicy().hasHeightForWidth())
        self.extra_suffix_cb.setSizePolicy(sizePolicy)
        self.extra_suffix_cb.setObjectName("extra_suffix_cb")
        self.extra_suffix_lyt.addWidget(self.extra_suffix_cb)
        self.extra_suffix_le = QtWidgets.QLineEdit(self.centralwidget)
        self.extra_suffix_le.setEnabled(False)
        self.extra_suffix_le.setText("")
        self.extra_suffix_le.setMaxLength(16)
        self.extra_suffix_le.setFrame(False)
        self.extra_suffix_le.setObjectName("extra_suffix_le")
        self.extra_suffix_lyt.addWidget(self.extra_suffix_le)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.extra_suffix_lyt.addItem(spacerItem2)
        self.verticalLayout.addLayout(self.extra_suffix_lyt)
        self.split_lyt = QtWidgets.QHBoxLayout()
        self.split_lyt.setObjectName("split_lyt")
        self.mind_l = QtWidgets.QLabel(self.centralwidget)
        self.mind_l.setObjectName("mind_l")
        self.split_lyt.addWidget(self.mind_l)
        self.min_dur_dsb = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.min_dur_dsb.setMinimumSize(QtCore.QSize(48, 0))
        self.min_dur_dsb.setFrame(False)
        self.min_dur_dsb.setAlignment(QtCore.Qt.AlignCenter)
        self.min_dur_dsb.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.min_dur_dsb.setMinimum(0.0)
        self.min_dur_dsb.setMaximum(4.0)
        self.min_dur_dsb.setSingleStep(0.1)
        self.min_dur_dsb.setProperty("value", 0.2)
        self.min_dur_dsb.setObjectName("min_dur_dsb")
        self.split_lyt.addWidget(self.min_dur_dsb)
        self.split_l = QtWidgets.QLabel(self.centralwidget)
        self.split_l.setObjectName("split_l")
        self.split_lyt.addWidget(self.split_l)
        self.split_db_dsb = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.split_db_dsb.setMinimumSize(QtCore.QSize(48, 0))
        self.split_db_dsb.setFrame(False)
        self.split_db_dsb.setAlignment(QtCore.Qt.AlignCenter)
        self.split_db_dsb.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.split_db_dsb.setMinimum(-120.0)
        self.split_db_dsb.setMaximum(0.0)
        self.split_db_dsb.setProperty("value", -80.0)
        self.split_db_dsb.setObjectName("split_db_dsb")
        self.split_lyt.addWidget(self.split_db_dsb)
        self.fade_l = QtWidgets.QLabel(self.centralwidget)
        self.fade_l.setObjectName("fade_l")
        self.split_lyt.addWidget(self.fade_l)
        self.fade_db_dsb = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.fade_db_dsb.setMinimumSize(QtCore.QSize(48, 0))
        self.fade_db_dsb.setFrame(False)
        self.fade_db_dsb.setAlignment(QtCore.Qt.AlignCenter)
        self.fade_db_dsb.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.fade_db_dsb.setMinimum(-120.0)
        self.fade_db_dsb.setMaximum(0.0)
        self.fade_db_dsb.setProperty("value", -60.0)
        self.fade_db_dsb.setObjectName("fade_db_dsb")
        self.split_lyt.addWidget(self.fade_db_dsb)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.split_lyt.addItem(spacerItem3)
        self.write_cue_cb = QtWidgets.QCheckBox(self.centralwidget)
        self.write_cue_cb.setObjectName("write_cue_cb")
        self.split_lyt.addWidget(self.write_cue_cb)
        self.verticalLayout.addLayout(self.split_lyt)
        self.audio_options_lyt = QtWidgets.QHBoxLayout()
        self.audio_options_lyt.setObjectName("audio_options_lyt")
        self.dc_offset_cb = QtWidgets.QCheckBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dc_offset_cb.sizePolicy().hasHeightForWidth())
        self.dc_offset_cb.setSizePolicy(sizePolicy)
        self.dc_offset_cb.setObjectName("dc_offset_cb")
        self.audio_options_lyt.addWidget(self.dc_offset_cb)
        self.dither_cb = QtWidgets.QCheckBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dither_cb.sizePolicy().hasHeightForWidth())
        self.dither_cb.setSizePolicy(sizePolicy)
        self.dither_cb.setObjectName("dither_cb")
        self.audio_options_lyt.addWidget(self.dither_cb)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.audio_options_lyt.addItem(spacerItem4)
        self.verticalLayout.addLayout(self.audio_options_lyt)
        self.output_path_title_l = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.output_path_title_l.sizePolicy().hasHeightForWidth())
        self.output_path_title_l.setSizePolicy(sizePolicy)
        self.output_path_title_l.setMinimumSize(QtCore.QSize(64, 0))
        self.output_path_title_l.setStyleSheet("background-color: rgb(127, 63, 95);\n"
"color: rgb(255, 255, 255);")
        self.output_path_title_l.setObjectName("output_path_title_l")
        self.verticalLayout.addWidget(self.output_path_title_l)
        self.output_lyt = QtWidgets.QHBoxLayout()
        self.output_lyt.setObjectName("output_lyt")
        self.output_path_l = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.output_path_l.sizePolicy().hasHeightForWidth())
        self.output_path_l.setSizePolicy(sizePolicy)
        self.output_path_l.setText("")
        self.output_path_l.setObjectName("output_path_l")
        self.output_lyt.addWidget(self.output_path_l)
        self.set_output_path_tb = QtWidgets.QToolButton(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.set_output_path_tb.setFont(font)
        self.set_output_path_tb.setObjectName("set_output_path_tb")
        self.output_lyt.addWidget(self.set_output_path_tb)
        self.verticalLayout.addLayout(self.output_lyt)
        self.subdir_lyt = QtWidgets.QHBoxLayout()
        self.subdir_lyt.setObjectName("subdir_lyt")
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.subdir_lyt.addItem(spacerItem5)
        self.subdir_cb = QtWidgets.QCheckBox(self.centralwidget)
        self.subdir_cb.setObjectName("subdir_cb")
        self.subdir_lyt.addWidget(self.subdir_cb)
        self.subdir_le = QtWidgets.QLineEdit(self.centralwidget)
        self.subdir_le.setEnabled(False)
        self.subdir_le.setObjectName("subdir_le")
        self.subdir_lyt.addWidget(self.subdir_le)
        self.verticalLayout.addLayout(self.subdir_lyt)
        self.settings_title_l = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.settings_title_l.sizePolicy().hasHeightForWidth())
        self.settings_title_l.setSizePolicy(sizePolicy)
        self.settings_title_l.setMinimumSize(QtCore.QSize(64, 0))
        self.settings_title_l.setStyleSheet("background-color: rgb(127, 63, 95);\n"
"color: rgb(255, 255, 255);")
        self.settings_title_l.setObjectName("settings_title_l")
        self.verticalLayout.addWidget(self.settings_title_l)
        self.file_options_lyt = QtWidgets.QHBoxLayout()
        self.file_options_lyt.setObjectName("file_options_lyt")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setObjectName("label_2")
        self.file_options_lyt.addWidget(self.label_2)
        self.format_cmb = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.format_cmb.sizePolicy().hasHeightForWidth())
        self.format_cmb.setSizePolicy(sizePolicy)
        self.format_cmb.setFrame(False)
        self.format_cmb.setObjectName("format_cmb")
        self.format_cmb.addItem("")
        self.format_cmb.addItem("")
        self.format_cmb.addItem("")
        self.format_cmb.addItem("")
        self.file_options_lyt.addWidget(self.format_cmb)
        self.label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.file_options_lyt.addWidget(self.label)
        self.bitdepth_cmb = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bitdepth_cmb.sizePolicy().hasHeightForWidth())
        self.bitdepth_cmb.setSizePolicy(sizePolicy)
        self.bitdepth_cmb.setFrame(False)
        self.bitdepth_cmb.setObjectName("bitdepth_cmb")
        self.bitdepth_cmb.addItem("")
        self.bitdepth_cmb.addItem("")
        self.bitdepth_cmb.addItem("")
        self.bitdepth_cmb.addItem("")
        self.file_options_lyt.addWidget(self.bitdepth_cmb)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.file_options_lyt.addItem(spacerItem6)
        self.verticalLayout.addLayout(self.file_options_lyt)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        self.buttons_lyt = QtWidgets.QGridLayout()
        self.buttons_lyt.setObjectName("buttons_lyt")
        self.process_pb = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.process_pb.sizePolicy().hasHeightForWidth())
        self.process_pb.setSizePolicy(sizePolicy)
        self.process_pb.setMinimumSize(QtCore.QSize(160, 0))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.process_pb.setFont(font)
        self.process_pb.setStyleSheet("QPushButton{background-color: rgb(127, 63, 95);\n"
"color: rgb(255, 255, 255);}")
        self.process_pb.setObjectName("process_pb")
        self.buttons_lyt.addWidget(self.process_pb, 1, 0, 1, 1)
        self.process_sel_pb = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.process_sel_pb.sizePolicy().hasHeightForWidth())
        self.process_sel_pb.setSizePolicy(sizePolicy)
        self.process_sel_pb.setMinimumSize(QtCore.QSize(160, 0))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.process_sel_pb.setFont(font)
        self.process_sel_pb.setObjectName("process_sel_pb")
        self.buttons_lyt.addWidget(self.process_sel_pb, 0, 0, 1, 1)
        self.verticalLayout.addLayout(self.buttons_lyt)
        self.progress_pb = QtWidgets.QProgressBar(self.centralwidget)
        self.progress_pb.setStyleSheet("QProgressBar{border: none;}")
        self.progress_pb.setProperty("value", 0)
        self.progress_pb.setAlignment(QtCore.Qt.AlignCenter)
        self.progress_pb.setTextVisible(False)
        self.progress_pb.setFormat("")
        self.progress_pb.setObjectName("progress_pb")
        self.verticalLayout.addWidget(self.progress_pb)
        split_tool_mw.setCentralWidget(self.centralwidget)

        self.retranslateUi(split_tool_mw)
        self.format_cmb.setCurrentIndex(0)
        self.bitdepth_cmb.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(split_tool_mw)

    def retranslateUi(self, split_tool_mw):
        _translate = QtCore.QCoreApplication.translate
        split_tool_mw.setWindowTitle(_translate("split_tool_mw", "Split Audio Tool"))
        self.files_title_l.setText(_translate("split_tool_mw", "Input Files"))
        self.files_lw.setToolTip(_translate("split_tool_mw", "Drag and drop files or directories\n"
"Right click for context actions\n"
"Double-click to play"))
        self.set_files_tb.setToolTip(_translate("split_tool_mw", "Set files to process"))
        self.set_files_tb.setText(_translate("split_tool_mw", "..."))
        self.options_title_l.setText(_translate("split_tool_mw", "Options"))
        self.label_3.setText(_translate("split_tool_mw", "Base Name"))
        self.basename_le.setToolTip(_translate("split_tool_mw", "Base name for split files\n"
"Use input file name if empty"))
        self.basename_le.setText(_translate("split_tool_mw", "Sample"))
        self.suffix_l.setText(_translate("split_tool_mw", "Suffix"))
        self.suffix_mode_cmb.setToolTip(_translate("split_tool_mw", "\'increment\'    Add increment to base name\n"
"\'noteName\'    Note name\n"
"\'note\'    MIDI note number (0-127)\n"
"\'suffix\'    Suffix from list  in order, increment as a fall back"))
        self.suffix_mode_cmb.setItemText(0, _translate("split_tool_mw", "increment"))
        self.suffix_mode_cmb.setItemText(1, _translate("split_tool_mw", "noteName"))
        self.suffix_mode_cmb.setItemText(2, _translate("split_tool_mw", "note"))
        self.suffix_mode_cmb.setItemText(3, _translate("split_tool_mw", "suffix"))
        self.suffix_le.setToolTip(_translate("split_tool_mw", "Suffix list added to the base name when using \' suffix\' mode\n"
"Use names separated by spaces\n"
"\n"
"Right click for context menu"))
        self.suffix_le.setText(_translate("split_tool_mw", "attack release"))
        self.pitch_mode_l.setText(_translate("split_tool_mw", "Pitch Mode"))
        self.pitch_mode_cmb.setToolTip(_translate("split_tool_mw", "Pitch detection algorithm\n"
"\n"
"\'corr\'    auto-correlation, fastest method by far but can be wrong about octave\n"
"\'pyin\'    pyin algorithm, good results with average speed\n"
"\'crepe\'    deep learning, different results with longer overhead\n"
"\n"
"NOTE:    None is perfect so always double-check results in an audio editor such as Audacity or RX using spectrum/frequency analysis\n"
"    Pitch detection does not work on bell-like sounds"))
        self.pitch_mode_cmb.setItemText(0, _translate("split_tool_mw", "corr"))
        self.pitch_mode_cmb.setItemText(1, _translate("split_tool_mw", "pyin"))
        self.pitch_mode_cmb.setItemText(2, _translate("split_tool_mw", "crepe"))
        self.use_pitch_fraction_cb.setToolTip(_translate("split_tool_mw", "Set pitch correction metadata"))
        self.use_pitch_fraction_cb.setText(_translate("split_tool_mw", "Use Pitch Fraction"))
        self.extra_suffix_cb.setText(_translate("split_tool_mw", "Extra Suffix"))
        self.mind_l.setText(_translate("split_tool_mw", "Min Duration"))
        self.min_dur_dsb.setToolTip(_translate("split_tool_mw", "Minimum silence/sound duration in s "))
        self.split_l.setText(_translate("split_tool_mw", "Split dB"))
        self.split_db_dsb.setToolTip(_translate("split_tool_mw", "Split dB threshold"))
        self.fade_l.setText(_translate("split_tool_mw", "Fade dB"))
        self.fade_db_dsb.setToolTip(_translate("split_tool_mw", "Fade dB threshold"))
        self.write_cue_cb.setToolTip(_translate("split_tool_mw", "Write a copy of input file with markers to check split points"))
        self.write_cue_cb.setText(_translate("split_tool_mw", "Write Cue File"))
        self.dc_offset_cb.setToolTip(_translate("split_tool_mw", "Recenter audio on a per-split basis "))
        self.dc_offset_cb.setText(_translate("split_tool_mw", "DC Offset"))
        self.dither_cb.setToolTip(_translate("split_tool_mw", "Apply noise with triangular distribution to fades for 16 bits bit depth"))
        self.dither_cb.setText(_translate("split_tool_mw", "Dither"))
        self.output_path_title_l.setText(_translate("split_tool_mw", "Output Path"))
        self.set_output_path_tb.setToolTip(_translate("split_tool_mw", "Set output path\n"
"Process files in their respective directory if empty"))
        self.set_output_path_tb.setText(_translate("split_tool_mw", "..."))
        self.subdir_cb.setToolTip(_translate("split_tool_mw", "Add sub directory to output path"))
        self.subdir_cb.setText(_translate("split_tool_mw", "Use sub-directory"))
        self.subdir_le.setToolTip(_translate("split_tool_mw", "Name of sub-directory"))
        self.subdir_le.setText(_translate("split_tool_mw", "Instrument/Samples"))
        self.settings_title_l.setText(_translate("split_tool_mw", "File Settings"))
        self.label_2.setText(_translate("split_tool_mw", "Format"))
        self.format_cmb.setToolTip(_translate("split_tool_mw", "flac only allows integer format up to 24 bits\n"
"Use wav or aif for 32 bits float"))
        self.format_cmb.setItemText(0, _translate("split_tool_mw", "same"))
        self.format_cmb.setItemText(1, _translate("split_tool_mw", "wav"))
        self.format_cmb.setItemText(2, _translate("split_tool_mw", "flac"))
        self.format_cmb.setItemText(3, _translate("split_tool_mw", "aif"))
        self.label.setText(_translate("split_tool_mw", "Bit Depth"))
        self.bitdepth_cmb.setItemText(0, _translate("split_tool_mw", "same"))
        self.bitdepth_cmb.setItemText(1, _translate("split_tool_mw", "16"))
        self.bitdepth_cmb.setItemText(2, _translate("split_tool_mw", "24"))
        self.bitdepth_cmb.setItemText(3, _translate("split_tool_mw", "32"))
        self.process_pb.setToolTip(_translate("split_tool_mw", "Process all input files with current settings"))
        self.process_pb.setText(_translate("split_tool_mw", "Batch Process"))
        self.process_sel_pb.setToolTip(_translate("split_tool_mw", "Process selected item(s) with current settings"))
        self.process_sel_pb.setText(_translate("split_tool_mw", "Process Selected"))
