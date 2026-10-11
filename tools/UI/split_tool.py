# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'split_tool.ui'
##
## Created by: Qt User Interface Compiler version 6.10.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractItemView, QAbstractSpinBox, QApplication, QCheckBox,
    QComboBox, QDoubleSpinBox, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QProgressBar, QPushButton,
    QSizePolicy, QSpacerItem, QToolButton, QVBoxLayout,
    QWidget)

class Ui_split_tool_mw(object):
    def setupUi(self, split_tool_mw):
        if not split_tool_mw.objectName():
            split_tool_mw.setObjectName(u"split_tool_mw")
        split_tool_mw.resize(720, 720)
        split_tool_mw.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        self.centralwidget = QWidget(split_tool_mw)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.files_title_l = QLabel(self.centralwidget)
        self.files_title_l.setObjectName(u"files_title_l")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.files_title_l.sizePolicy().hasHeightForWidth())
        self.files_title_l.setSizePolicy(sizePolicy)
        self.files_title_l.setMinimumSize(QSize(64, 0))
        self.files_title_l.setStyleSheet(u"background-color: rgb(127, 63, 95);\n"
"color: rgb(255, 255, 255);")

        self.verticalLayout.addWidget(self.files_title_l)

        self.input_lyt = QHBoxLayout()
        self.input_lyt.setSpacing(4)
        self.input_lyt.setObjectName(u"input_lyt")
        self.files_lw = QListWidget(self.centralwidget)
        self.files_lw.setObjectName(u"files_lw")
        self.files_lw.setFrameShape(QFrame.NoFrame)
        self.files_lw.setDragDropMode(QAbstractItemView.DropOnly)
        self.files_lw.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.input_lyt.addWidget(self.files_lw)

        self.set_files_tb = QToolButton(self.centralwidget)
        self.set_files_tb.setObjectName(u"set_files_tb")
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.set_files_tb.setFont(font)

        self.input_lyt.addWidget(self.set_files_tb)


        self.verticalLayout.addLayout(self.input_lyt)

        self.options_title_l = QLabel(self.centralwidget)
        self.options_title_l.setObjectName(u"options_title_l")
        sizePolicy.setHeightForWidth(self.options_title_l.sizePolicy().hasHeightForWidth())
        self.options_title_l.setSizePolicy(sizePolicy)
        self.options_title_l.setMinimumSize(QSize(64, 0))
        self.options_title_l.setStyleSheet(u"background-color: rgb(127, 63, 95);\n"
"color: rgb(255, 255, 255);")

        self.verticalLayout.addWidget(self.options_title_l)

        self.split_mode_lyt = QHBoxLayout()
        self.split_mode_lyt.setSpacing(4)
        self.split_mode_lyt.setObjectName(u"split_mode_lyt")
        self.split_mode_l = QLabel(self.centralwidget)
        self.split_mode_l.setObjectName(u"split_mode_l")

        self.split_mode_lyt.addWidget(self.split_mode_l)

        self.split_mode_cmb = QComboBox(self.centralwidget)
        self.split_mode_cmb.addItem("")
        self.split_mode_cmb.addItem("")
        self.split_mode_cmb.setObjectName(u"split_mode_cmb")
        self.split_mode_cmb.setFrame(False)

        self.split_mode_lyt.addWidget(self.split_mode_cmb)

        self.horizontalSpacer_8 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.split_mode_lyt.addItem(self.horizontalSpacer_8)


        self.verticalLayout.addLayout(self.split_mode_lyt)

        self.line_2 = QFrame(self.centralwidget)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.Shape.HLine)
        self.line_2.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout.addWidget(self.line_2)

        self.split_options_wid = QWidget(self.centralwidget)
        self.split_options_wid.setObjectName(u"split_options_wid")
        self.split_options_wid.setEnabled(True)
        self.verticalLayout_2 = QVBoxLayout(self.split_options_wid)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.naming_lyt = QHBoxLayout()
        self.naming_lyt.setSpacing(4)
        self.naming_lyt.setObjectName(u"naming_lyt")
        self.label_3 = QLabel(self.split_options_wid)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setMinimumSize(QSize(80, 0))
        self.label_3.setMaximumSize(QSize(80, 16777215))

        self.naming_lyt.addWidget(self.label_3)

        self.basename_le = QLineEdit(self.split_options_wid)
        self.basename_le.setObjectName(u"basename_le")
        self.basename_le.setFrame(False)

        self.naming_lyt.addWidget(self.basename_le)

        self.suffix_l = QLabel(self.split_options_wid)
        self.suffix_l.setObjectName(u"suffix_l")

        self.naming_lyt.addWidget(self.suffix_l)

        self.suffix_mode_cmb = QComboBox(self.split_options_wid)
        self.suffix_mode_cmb.addItem("")
        self.suffix_mode_cmb.addItem("")
        self.suffix_mode_cmb.addItem("")
        self.suffix_mode_cmb.addItem("")
        self.suffix_mode_cmb.setObjectName(u"suffix_mode_cmb")
        self.suffix_mode_cmb.setFrame(False)

        self.naming_lyt.addWidget(self.suffix_mode_cmb)

        self.suffix_le = QLineEdit(self.split_options_wid)
        self.suffix_le.setObjectName(u"suffix_le")
        self.suffix_le.setEnabled(False)
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.suffix_le.sizePolicy().hasHeightForWidth())
        self.suffix_le.setSizePolicy(sizePolicy1)
        self.suffix_le.setFrame(False)

        self.naming_lyt.addWidget(self.suffix_le)


        self.verticalLayout_2.addLayout(self.naming_lyt)

        self.pitch_mode_wid = QWidget(self.split_options_wid)
        self.pitch_mode_wid.setObjectName(u"pitch_mode_wid")
        self.pitch_mode_wid.setEnabled(False)
        self.horizontalLayout = QHBoxLayout(self.pitch_mode_wid)
        self.horizontalLayout.setSpacing(4)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalSpacer_7 = QSpacerItem(80, 20, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_7)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_4)

        self.pitch_mode_l = QLabel(self.pitch_mode_wid)
        self.pitch_mode_l.setObjectName(u"pitch_mode_l")

        self.horizontalLayout.addWidget(self.pitch_mode_l)

        self.pitch_mode_cmb = QComboBox(self.pitch_mode_wid)
        self.pitch_mode_cmb.addItem("")
        self.pitch_mode_cmb.addItem("")
        self.pitch_mode_cmb.setObjectName(u"pitch_mode_cmb")
        self.pitch_mode_cmb.setFrame(False)

        self.horizontalLayout.addWidget(self.pitch_mode_cmb)

        self.use_pitch_fraction_cb = QCheckBox(self.pitch_mode_wid)
        self.use_pitch_fraction_cb.setObjectName(u"use_pitch_fraction_cb")
        sizePolicy1.setHeightForWidth(self.use_pitch_fraction_cb.sizePolicy().hasHeightForWidth())
        self.use_pitch_fraction_cb.setSizePolicy(sizePolicy1)

        self.horizontalLayout.addWidget(self.use_pitch_fraction_cb)


        self.verticalLayout_2.addWidget(self.pitch_mode_wid)


        self.verticalLayout.addWidget(self.split_options_wid)

        self.extra_suffix_lyt = QHBoxLayout()
        self.extra_suffix_lyt.setSpacing(4)
        self.extra_suffix_lyt.setObjectName(u"extra_suffix_lyt")
        self.extra_suffix_cb = QCheckBox(self.centralwidget)
        self.extra_suffix_cb.setObjectName(u"extra_suffix_cb")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.extra_suffix_cb.sizePolicy().hasHeightForWidth())
        self.extra_suffix_cb.setSizePolicy(sizePolicy2)

        self.extra_suffix_lyt.addWidget(self.extra_suffix_cb)

        self.extra_suffix_le = QLineEdit(self.centralwidget)
        self.extra_suffix_le.setObjectName(u"extra_suffix_le")
        self.extra_suffix_le.setEnabled(False)
        self.extra_suffix_le.setMaxLength(16)
        self.extra_suffix_le.setFrame(False)

        self.extra_suffix_lyt.addWidget(self.extra_suffix_le)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.extra_suffix_lyt.addItem(self.horizontalSpacer_5)


        self.verticalLayout.addLayout(self.extra_suffix_lyt)

        self.split_lyt = QHBoxLayout()
        self.split_lyt.setObjectName(u"split_lyt")
        self.min_dur_wid = QWidget(self.centralwidget)
        self.min_dur_wid.setObjectName(u"min_dur_wid")
        self.horizontalLayout_2 = QHBoxLayout(self.min_dur_wid)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.mind_l = QLabel(self.min_dur_wid)
        self.mind_l.setObjectName(u"mind_l")

        self.horizontalLayout_2.addWidget(self.mind_l)

        self.min_dur_dsb = QDoubleSpinBox(self.min_dur_wid)
        self.min_dur_dsb.setObjectName(u"min_dur_dsb")
        self.min_dur_dsb.setMinimumSize(QSize(64, 0))
        self.min_dur_dsb.setFrame(False)
        self.min_dur_dsb.setAlignment(Qt.AlignCenter)
        self.min_dur_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.min_dur_dsb.setDecimals(3)
        self.min_dur_dsb.setMinimum(0.000000000000000)
        self.min_dur_dsb.setMaximum(4.000000000000000)
        self.min_dur_dsb.setSingleStep(0.100000000000000)
        self.min_dur_dsb.setValue(0.200000000000000)

        self.horizontalLayout_2.addWidget(self.min_dur_dsb)


        self.split_lyt.addWidget(self.min_dur_wid)

        self.split_l = QLabel(self.centralwidget)
        self.split_l.setObjectName(u"split_l")

        self.split_lyt.addWidget(self.split_l)

        self.split_db_dsb = QDoubleSpinBox(self.centralwidget)
        self.split_db_dsb.setObjectName(u"split_db_dsb")
        self.split_db_dsb.setMinimumSize(QSize(48, 0))
        self.split_db_dsb.setFrame(False)
        self.split_db_dsb.setAlignment(Qt.AlignCenter)
        self.split_db_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.split_db_dsb.setMinimum(-120.000000000000000)
        self.split_db_dsb.setMaximum(0.000000000000000)
        self.split_db_dsb.setValue(-80.000000000000000)

        self.split_lyt.addWidget(self.split_db_dsb)

        self.fade_l = QLabel(self.centralwidget)
        self.fade_l.setObjectName(u"fade_l")

        self.split_lyt.addWidget(self.fade_l)

        self.fade_db_dsb = QDoubleSpinBox(self.centralwidget)
        self.fade_db_dsb.setObjectName(u"fade_db_dsb")
        self.fade_db_dsb.setMinimumSize(QSize(48, 0))
        self.fade_db_dsb.setFrame(False)
        self.fade_db_dsb.setAlignment(Qt.AlignCenter)
        self.fade_db_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.fade_db_dsb.setMinimum(-120.000000000000000)
        self.fade_db_dsb.setMaximum(0.000000000000000)
        self.fade_db_dsb.setValue(-60.000000000000000)

        self.split_lyt.addWidget(self.fade_db_dsb)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.split_lyt.addItem(self.horizontalSpacer)

        self.write_cue_cb = QCheckBox(self.centralwidget)
        self.write_cue_cb.setObjectName(u"write_cue_cb")

        self.split_lyt.addWidget(self.write_cue_cb)


        self.verticalLayout.addLayout(self.split_lyt)

        self.audio_options_lyt = QHBoxLayout()
        self.audio_options_lyt.setSpacing(4)
        self.audio_options_lyt.setObjectName(u"audio_options_lyt")
        self.dc_offset_cb = QCheckBox(self.centralwidget)
        self.dc_offset_cb.setObjectName(u"dc_offset_cb")
        sizePolicy2.setHeightForWidth(self.dc_offset_cb.sizePolicy().hasHeightForWidth())
        self.dc_offset_cb.setSizePolicy(sizePolicy2)

        self.audio_options_lyt.addWidget(self.dc_offset_cb)

        self.dither_cb = QCheckBox(self.centralwidget)
        self.dither_cb.setObjectName(u"dither_cb")
        sizePolicy2.setHeightForWidth(self.dither_cb.sizePolicy().hasHeightForWidth())
        self.dither_cb.setSizePolicy(sizePolicy2)

        self.audio_options_lyt.addWidget(self.dither_cb)

        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.audio_options_lyt.addItem(self.horizontalSpacer_6)


        self.verticalLayout.addLayout(self.audio_options_lyt)

        self.output_path_title_l = QLabel(self.centralwidget)
        self.output_path_title_l.setObjectName(u"output_path_title_l")
        sizePolicy.setHeightForWidth(self.output_path_title_l.sizePolicy().hasHeightForWidth())
        self.output_path_title_l.setSizePolicy(sizePolicy)
        self.output_path_title_l.setMinimumSize(QSize(64, 0))
        self.output_path_title_l.setStyleSheet(u"background-color: rgb(127, 63, 95);\n"
"color: rgb(255, 255, 255);")

        self.verticalLayout.addWidget(self.output_path_title_l)

        self.output_lyt = QHBoxLayout()
        self.output_lyt.setSpacing(4)
        self.output_lyt.setObjectName(u"output_lyt")
        self.output_path_l = QLabel(self.centralwidget)
        self.output_path_l.setObjectName(u"output_path_l")
        sizePolicy.setHeightForWidth(self.output_path_l.sizePolicy().hasHeightForWidth())
        self.output_path_l.setSizePolicy(sizePolicy)

        self.output_lyt.addWidget(self.output_path_l)

        self.set_output_path_tb = QToolButton(self.centralwidget)
        self.set_output_path_tb.setObjectName(u"set_output_path_tb")
        self.set_output_path_tb.setFont(font)

        self.output_lyt.addWidget(self.set_output_path_tb)


        self.verticalLayout.addLayout(self.output_lyt)

        self.subdir_lyt = QHBoxLayout()
        self.subdir_lyt.setSpacing(4)
        self.subdir_lyt.setObjectName(u"subdir_lyt")
        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.subdir_lyt.addItem(self.horizontalSpacer_3)

        self.subdir_cb = QCheckBox(self.centralwidget)
        self.subdir_cb.setObjectName(u"subdir_cb")

        self.subdir_lyt.addWidget(self.subdir_cb)

        self.subdir_le = QLineEdit(self.centralwidget)
        self.subdir_le.setObjectName(u"subdir_le")
        self.subdir_le.setEnabled(False)

        self.subdir_lyt.addWidget(self.subdir_le)


        self.verticalLayout.addLayout(self.subdir_lyt)

        self.settings_title_l = QLabel(self.centralwidget)
        self.settings_title_l.setObjectName(u"settings_title_l")
        sizePolicy.setHeightForWidth(self.settings_title_l.sizePolicy().hasHeightForWidth())
        self.settings_title_l.setSizePolicy(sizePolicy)
        self.settings_title_l.setMinimumSize(QSize(64, 0))
        self.settings_title_l.setStyleSheet(u"background-color: rgb(127, 63, 95);\n"
"color: rgb(255, 255, 255);")

        self.verticalLayout.addWidget(self.settings_title_l)

        self.file_options_lyt = QHBoxLayout()
        self.file_options_lyt.setSpacing(4)
        self.file_options_lyt.setObjectName(u"file_options_lyt")
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy3)

        self.file_options_lyt.addWidget(self.label_2)

        self.format_cmb = QComboBox(self.centralwidget)
        self.format_cmb.addItem("")
        self.format_cmb.addItem("")
        self.format_cmb.addItem("")
        self.format_cmb.addItem("")
        self.format_cmb.setObjectName(u"format_cmb")
        sizePolicy.setHeightForWidth(self.format_cmb.sizePolicy().hasHeightForWidth())
        self.format_cmb.setSizePolicy(sizePolicy)
        self.format_cmb.setFrame(False)

        self.file_options_lyt.addWidget(self.format_cmb)

        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        sizePolicy2.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy2)

        self.file_options_lyt.addWidget(self.label)

        self.bitdepth_cmb = QComboBox(self.centralwidget)
        self.bitdepth_cmb.addItem("")
        self.bitdepth_cmb.addItem("")
        self.bitdepth_cmb.addItem("")
        self.bitdepth_cmb.addItem("")
        self.bitdepth_cmb.setObjectName(u"bitdepth_cmb")
        sizePolicy.setHeightForWidth(self.bitdepth_cmb.sizePolicy().hasHeightForWidth())
        self.bitdepth_cmb.setSizePolicy(sizePolicy)
        self.bitdepth_cmb.setFrame(False)

        self.file_options_lyt.addWidget(self.bitdepth_cmb)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.file_options_lyt.addItem(self.horizontalSpacer_2)

        self.no_overwriting_cb = QCheckBox(self.centralwidget)
        self.no_overwriting_cb.setObjectName(u"no_overwriting_cb")
        self.no_overwriting_cb.setEnabled(False)
        self.no_overwriting_cb.setChecked(True)

        self.file_options_lyt.addWidget(self.no_overwriting_cb)


        self.verticalLayout.addLayout(self.file_options_lyt)

        self.line = QFrame(self.centralwidget)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout.addWidget(self.line)

        self.buttons_lyt = QGridLayout()
        self.buttons_lyt.setSpacing(4)
        self.buttons_lyt.setObjectName(u"buttons_lyt")
        self.process_pb = QPushButton(self.centralwidget)
        self.process_pb.setObjectName(u"process_pb")
        sizePolicy2.setHeightForWidth(self.process_pb.sizePolicy().hasHeightForWidth())
        self.process_pb.setSizePolicy(sizePolicy2)
        self.process_pb.setMinimumSize(QSize(160, 0))
        font1 = QFont()
        font1.setBold(True)
        self.process_pb.setFont(font1)
        self.process_pb.setStyleSheet(u"QPushButton{background-color: rgb(127, 63, 95);\n"
"color: rgb(255, 255, 255);}")

        self.buttons_lyt.addWidget(self.process_pb, 1, 0, 1, 1)

        self.process_sel_pb = QPushButton(self.centralwidget)
        self.process_sel_pb.setObjectName(u"process_sel_pb")
        sizePolicy2.setHeightForWidth(self.process_sel_pb.sizePolicy().hasHeightForWidth())
        self.process_sel_pb.setSizePolicy(sizePolicy2)
        self.process_sel_pb.setMinimumSize(QSize(160, 0))
        self.process_sel_pb.setFont(font1)

        self.buttons_lyt.addWidget(self.process_sel_pb, 0, 0, 1, 1)


        self.verticalLayout.addLayout(self.buttons_lyt)

        self.progress_pb = QProgressBar(self.centralwidget)
        self.progress_pb.setObjectName(u"progress_pb")
        self.progress_pb.setStyleSheet(u"QProgressBar{border: none;}")
        self.progress_pb.setValue(0)
        self.progress_pb.setAlignment(Qt.AlignCenter)
        self.progress_pb.setTextVisible(False)
        self.progress_pb.setFormat(u"")

        self.verticalLayout.addWidget(self.progress_pb)

        split_tool_mw.setCentralWidget(self.centralwidget)

        self.retranslateUi(split_tool_mw)

        self.format_cmb.setCurrentIndex(0)
        self.bitdepth_cmb.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(split_tool_mw)
    # setupUi

    def retranslateUi(self, split_tool_mw):
        split_tool_mw.setWindowTitle(QCoreApplication.translate("split_tool_mw", u"Split Audio Tool", None))
        self.files_title_l.setText(QCoreApplication.translate("split_tool_mw", u"Input Files", None))
#if QT_CONFIG(tooltip)
        self.files_lw.setToolTip(QCoreApplication.translate("split_tool_mw", u"Drag and drop files or directories\n"
"Right click for context actions\n"
"Double-click to play", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.set_files_tb.setToolTip(QCoreApplication.translate("split_tool_mw", u"Set files to process", None))
#endif // QT_CONFIG(tooltip)
        self.set_files_tb.setText(QCoreApplication.translate("split_tool_mw", u"...", None))
        self.options_title_l.setText(QCoreApplication.translate("split_tool_mw", u"Options", None))
        self.split_mode_l.setText(QCoreApplication.translate("split_tool_mw", u"Split Mode", None))
        self.split_mode_cmb.setItemText(0, QCoreApplication.translate("split_tool_mw", u"split", None))
        self.split_mode_cmb.setItemText(1, QCoreApplication.translate("split_tool_mw", u"trim_only", None))

#if QT_CONFIG(tooltip)
        self.split_mode_cmb.setToolTip(QCoreApplication.translate("split_tool_mw", u"Choose between Split or Trim files only", None))
#endif // QT_CONFIG(tooltip)
        self.label_3.setText(QCoreApplication.translate("split_tool_mw", u"Base Name", None))
#if QT_CONFIG(tooltip)
        self.basename_le.setToolTip(QCoreApplication.translate("split_tool_mw", u"Base name for split files\n"
"Use input file name if empty", None))
#endif // QT_CONFIG(tooltip)
        self.basename_le.setText(QCoreApplication.translate("split_tool_mw", u"Sample", None))
        self.suffix_l.setText(QCoreApplication.translate("split_tool_mw", u"Suffix", None))
        self.suffix_mode_cmb.setItemText(0, QCoreApplication.translate("split_tool_mw", u"increment", None))
        self.suffix_mode_cmb.setItemText(1, QCoreApplication.translate("split_tool_mw", u"noteName", None))
        self.suffix_mode_cmb.setItemText(2, QCoreApplication.translate("split_tool_mw", u"note", None))
        self.suffix_mode_cmb.setItemText(3, QCoreApplication.translate("split_tool_mw", u"suffix", None))

#if QT_CONFIG(tooltip)
        self.suffix_mode_cmb.setToolTip(QCoreApplication.translate("split_tool_mw", u"'increment'	Add increment to base name\n"
"'noteName'	Note name\n"
"'note'	MIDI note number (0-127)\n"
"'suffix'	Suffix from list  in order, increment as a fall back", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.suffix_le.setToolTip(QCoreApplication.translate("split_tool_mw", u"Suffix list added to the base name when using ' suffix' mode\n"
"Use names separated by spaces\n"
"\n"
"Right click for context menu", None))
#endif // QT_CONFIG(tooltip)
        self.suffix_le.setText(QCoreApplication.translate("split_tool_mw", u"attack release", None))
        self.pitch_mode_l.setText(QCoreApplication.translate("split_tool_mw", u"Pitch Mode", None))
        self.pitch_mode_cmb.setItemText(0, QCoreApplication.translate("split_tool_mw", u"yin", None))
        self.pitch_mode_cmb.setItemText(1, QCoreApplication.translate("split_tool_mw", u"corr", None))

#if QT_CONFIG(tooltip)
        self.pitch_mode_cmb.setToolTip(QCoreApplication.translate("split_tool_mw", u"Pitch detection algorithm\n"
"NOTE:	No algorithm is perfect so always double-check results in an audio editor such as Audacity or RX using spectrum/frequency analysis\n"
"	Pitch detection does not work on bell-like sounds\n"
"\n"
"'yin'	good results and rather fast, best combined with fine tuning\n"
"'corr'	auto-correlation, fastest method but can be wrong about octave", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.use_pitch_fraction_cb.setToolTip(QCoreApplication.translate("split_tool_mw", u"Set pitch correction metadata", None))
#endif // QT_CONFIG(tooltip)
        self.use_pitch_fraction_cb.setText(QCoreApplication.translate("split_tool_mw", u"Use Pitch Fraction", None))
        self.extra_suffix_cb.setText(QCoreApplication.translate("split_tool_mw", u"Extra Suffix", None))
        self.extra_suffix_le.setText("")
        self.mind_l.setText(QCoreApplication.translate("split_tool_mw", u"Min Duration", None))
#if QT_CONFIG(tooltip)
        self.min_dur_dsb.setToolTip(QCoreApplication.translate("split_tool_mw", u"Minimum silence/sound duration in s ", None))
#endif // QT_CONFIG(tooltip)
        self.split_l.setText(QCoreApplication.translate("split_tool_mw", u"Split dB", None))
#if QT_CONFIG(tooltip)
        self.split_db_dsb.setToolTip(QCoreApplication.translate("split_tool_mw", u"Split/Trim dB threshold", None))
#endif // QT_CONFIG(tooltip)
        self.fade_l.setText(QCoreApplication.translate("split_tool_mw", u"Fade dB", None))
#if QT_CONFIG(tooltip)
        self.fade_db_dsb.setToolTip(QCoreApplication.translate("split_tool_mw", u"Fade dB threshold", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.write_cue_cb.setToolTip(QCoreApplication.translate("split_tool_mw", u"Write a copy of input file with markers to check split points", None))
#endif // QT_CONFIG(tooltip)
        self.write_cue_cb.setText(QCoreApplication.translate("split_tool_mw", u"Write Cue File", None))
#if QT_CONFIG(tooltip)
        self.dc_offset_cb.setToolTip(QCoreApplication.translate("split_tool_mw", u"Recenter audio on a per-split basis ", None))
#endif // QT_CONFIG(tooltip)
        self.dc_offset_cb.setText(QCoreApplication.translate("split_tool_mw", u"DC Offset", None))
#if QT_CONFIG(tooltip)
        self.dither_cb.setToolTip(QCoreApplication.translate("split_tool_mw", u"Apply noise with triangular distribution to fades for 16 bits bit depth", None))
#endif // QT_CONFIG(tooltip)
        self.dither_cb.setText(QCoreApplication.translate("split_tool_mw", u"Dither", None))
        self.output_path_title_l.setText(QCoreApplication.translate("split_tool_mw", u"Output Path", None))
        self.output_path_l.setText("")
#if QT_CONFIG(tooltip)
        self.set_output_path_tb.setToolTip(QCoreApplication.translate("split_tool_mw", u"Set output path\n"
"Process files in their respective directory if empty", None))
#endif // QT_CONFIG(tooltip)
        self.set_output_path_tb.setText(QCoreApplication.translate("split_tool_mw", u"...", None))
#if QT_CONFIG(tooltip)
        self.subdir_cb.setToolTip(QCoreApplication.translate("split_tool_mw", u"Add sub directory to output path", None))
#endif // QT_CONFIG(tooltip)
        self.subdir_cb.setText(QCoreApplication.translate("split_tool_mw", u"Use sub-directory", None))
#if QT_CONFIG(tooltip)
        self.subdir_le.setToolTip(QCoreApplication.translate("split_tool_mw", u"Name of sub-directory", None))
#endif // QT_CONFIG(tooltip)
        self.subdir_le.setText(QCoreApplication.translate("split_tool_mw", u"Instrument/Samples", None))
        self.settings_title_l.setText(QCoreApplication.translate("split_tool_mw", u"File Settings", None))
        self.label_2.setText(QCoreApplication.translate("split_tool_mw", u"Format", None))
        self.format_cmb.setItemText(0, QCoreApplication.translate("split_tool_mw", u"same", None))
        self.format_cmb.setItemText(1, QCoreApplication.translate("split_tool_mw", u"wav", None))
        self.format_cmb.setItemText(2, QCoreApplication.translate("split_tool_mw", u"flac", None))
        self.format_cmb.setItemText(3, QCoreApplication.translate("split_tool_mw", u"aif", None))

#if QT_CONFIG(tooltip)
        self.format_cmb.setToolTip(QCoreApplication.translate("split_tool_mw", u"flac only allows integer format up to 24 bits\n"
"Use wav or aif for 32 bits float", None))
#endif // QT_CONFIG(tooltip)
        self.label.setText(QCoreApplication.translate("split_tool_mw", u"Bit Depth", None))
        self.bitdepth_cmb.setItemText(0, QCoreApplication.translate("split_tool_mw", u"same", None))
        self.bitdepth_cmb.setItemText(1, QCoreApplication.translate("split_tool_mw", u"16", None))
        self.bitdepth_cmb.setItemText(2, QCoreApplication.translate("split_tool_mw", u"24", None))
        self.bitdepth_cmb.setItemText(3, QCoreApplication.translate("split_tool_mw", u"32", None))

#if QT_CONFIG(tooltip)
        self.no_overwriting_cb.setToolTip(QCoreApplication.translate("split_tool_mw", u"Avoid overwriting original files by moving them to a backup directory", None))
#endif // QT_CONFIG(tooltip)
        self.no_overwriting_cb.setText(QCoreApplication.translate("split_tool_mw", u"Avoid Overwriting", None))
#if QT_CONFIG(tooltip)
        self.process_pb.setToolTip(QCoreApplication.translate("split_tool_mw", u"Process all input files with current settings", None))
#endif // QT_CONFIG(tooltip)
        self.process_pb.setText(QCoreApplication.translate("split_tool_mw", u"Batch Process", None))
#if QT_CONFIG(tooltip)
        self.process_sel_pb.setToolTip(QCoreApplication.translate("split_tool_mw", u"Process selected item(s) with current settings", None))
#endif // QT_CONFIG(tooltip)
        self.process_sel_pb.setText(QCoreApplication.translate("split_tool_mw", u"Process Selected", None))
    # retranslateUi

