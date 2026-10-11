# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'rename_tool.ui'
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
    QSizePolicy, QSpacerItem, QSpinBox, QToolButton,
    QVBoxLayout, QWidget)

class Ui_rename_tool_mw(object):
    def setupUi(self, rename_tool_mw):
        if not rename_tool_mw.objectName():
            rename_tool_mw.setObjectName(u"rename_tool_mw")
        rename_tool_mw.resize(800, 800)
        rename_tool_mw.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        self.centralwidget = QWidget(rename_tool_mw)
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
        self.files_title_l.setStyleSheet(u"background-color: rgb(159, 95, 95);\n"
"color: rgb(255, 255, 255);")

        self.verticalLayout.addWidget(self.files_title_l)

        self.files_lyt = QHBoxLayout()
        self.files_lyt.setObjectName(u"files_lyt")
        self.files_lw = QListWidget(self.centralwidget)
        self.files_lw.setObjectName(u"files_lw")
        self.files_lw.setFrameShape(QFrame.NoFrame)
        self.files_lw.setDragDropMode(QAbstractItemView.DropOnly)
        self.files_lw.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.files_lyt.addWidget(self.files_lw)

        self.set_files_tb = QToolButton(self.centralwidget)
        self.set_files_tb.setObjectName(u"set_files_tb")
        font = QFont()
        font.setBold(True)
        self.set_files_tb.setFont(font)

        self.files_lyt.addWidget(self.set_files_tb)


        self.verticalLayout.addLayout(self.files_lyt)

        self.naming_title_l = QLabel(self.centralwidget)
        self.naming_title_l.setObjectName(u"naming_title_l")
        sizePolicy.setHeightForWidth(self.naming_title_l.sizePolicy().hasHeightForWidth())
        self.naming_title_l.setSizePolicy(sizePolicy)
        self.naming_title_l.setMinimumSize(QSize(64, 0))
        self.naming_title_l.setStyleSheet(u"background-color: rgb(159, 95, 95);\n"
"color: rgb(255, 255, 255);")

        self.verticalLayout.addWidget(self.naming_title_l)

        self.basename_lyt = QHBoxLayout()
        self.basename_lyt.setObjectName(u"basename_lyt")
        self.groupname_l = QLabel(self.centralwidget)
        self.groupname_l.setObjectName(u"groupname_l")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.groupname_l.sizePolicy().hasHeightForWidth())
        self.groupname_l.setSizePolicy(sizePolicy1)
        self.groupname_l.setMinimumSize(QSize(128, 0))
        self.groupname_l.setAlignment(Qt.AlignCenter)

        self.basename_lyt.addWidget(self.groupname_l)

        self.groupname_le = QLineEdit(self.centralwidget)
        self.groupname_le.setObjectName(u"groupname_le")
        self.groupname_le.setFrame(False)
        self.groupname_le.setAlignment(Qt.AlignCenter)

        self.basename_lyt.addWidget(self.groupname_le)

        self.repstr_l = QLabel(self.centralwidget)
        self.repstr_l.setObjectName(u"repstr_l")
        sizePolicy1.setHeightForWidth(self.repstr_l.sizePolicy().hasHeightForWidth())
        self.repstr_l.setSizePolicy(sizePolicy1)
        self.repstr_l.setMinimumSize(QSize(128, 0))
        self.repstr_l.setAlignment(Qt.AlignCenter)

        self.basename_lyt.addWidget(self.repstr_l)

        self.repstr_le = QLineEdit(self.centralwidget)
        self.repstr_le.setObjectName(u"repstr_le")
        self.repstr_le.setFrame(False)
        self.repstr_le.setAlignment(Qt.AlignCenter)

        self.basename_lyt.addWidget(self.repstr_le)


        self.verticalLayout.addLayout(self.basename_lyt)

        self.pattern_lyt = QHBoxLayout()
        self.pattern_lyt.setObjectName(u"pattern_lyt")
        self.src_pattern_pb = QPushButton(self.centralwidget)
        self.src_pattern_pb.setObjectName(u"src_pattern_pb")
        sizePolicy1.setHeightForWidth(self.src_pattern_pb.sizePolicy().hasHeightForWidth())
        self.src_pattern_pb.setSizePolicy(sizePolicy1)
        self.src_pattern_pb.setMinimumSize(QSize(128, 0))
        self.src_pattern_pb.setFont(font)

        self.pattern_lyt.addWidget(self.src_pattern_pb)

        self.src_pattern_le = QLineEdit(self.centralwidget)
        self.src_pattern_le.setObjectName(u"src_pattern_le")
        self.src_pattern_le.setFrame(False)
        self.src_pattern_le.setAlignment(Qt.AlignCenter)

        self.pattern_lyt.addWidget(self.src_pattern_le)

        self.tgt_pattern_pb = QPushButton(self.centralwidget)
        self.tgt_pattern_pb.setObjectName(u"tgt_pattern_pb")
        sizePolicy1.setHeightForWidth(self.tgt_pattern_pb.sizePolicy().hasHeightForWidth())
        self.tgt_pattern_pb.setSizePolicy(sizePolicy1)
        self.tgt_pattern_pb.setMinimumSize(QSize(128, 0))
        self.tgt_pattern_pb.setFont(font)

        self.pattern_lyt.addWidget(self.tgt_pattern_pb)

        self.tgt_pattern_le = QLineEdit(self.centralwidget)
        self.tgt_pattern_le.setObjectName(u"tgt_pattern_le")
        self.tgt_pattern_le.setFrame(False)
        self.tgt_pattern_le.setAlignment(Qt.AlignCenter)

        self.pattern_lyt.addWidget(self.tgt_pattern_le)


        self.verticalLayout.addLayout(self.pattern_lyt)

        self.presuffix_lyt = QHBoxLayout()
        self.presuffix_lyt.setObjectName(u"presuffix_lyt")
        self.prefix_l = QLabel(self.centralwidget)
        self.prefix_l.setObjectName(u"prefix_l")
        sizePolicy1.setHeightForWidth(self.prefix_l.sizePolicy().hasHeightForWidth())
        self.prefix_l.setSizePolicy(sizePolicy1)
        self.prefix_l.setMinimumSize(QSize(128, 0))
        self.prefix_l.setAlignment(Qt.AlignCenter)

        self.presuffix_lyt.addWidget(self.prefix_l)

        self.prefix_le = QLineEdit(self.centralwidget)
        self.prefix_le.setObjectName(u"prefix_le")
        self.prefix_le.setFrame(False)
        self.prefix_le.setAlignment(Qt.AlignCenter)

        self.presuffix_lyt.addWidget(self.prefix_le)

        self.suffix_l = QLabel(self.centralwidget)
        self.suffix_l.setObjectName(u"suffix_l")
        sizePolicy1.setHeightForWidth(self.suffix_l.sizePolicy().hasHeightForWidth())
        self.suffix_l.setSizePolicy(sizePolicy1)
        self.suffix_l.setMinimumSize(QSize(128, 0))
        self.suffix_l.setAlignment(Qt.AlignCenter)

        self.presuffix_lyt.addWidget(self.suffix_l)

        self.suffix_le = QLineEdit(self.centralwidget)
        self.suffix_le.setObjectName(u"suffix_le")
        self.suffix_le.setFrame(False)
        self.suffix_le.setAlignment(Qt.AlignCenter)

        self.presuffix_lyt.addWidget(self.suffix_le)


        self.verticalLayout.addLayout(self.presuffix_lyt)

        self.note_loop_title = QLabel(self.centralwidget)
        self.note_loop_title.setObjectName(u"note_loop_title")
        sizePolicy.setHeightForWidth(self.note_loop_title.sizePolicy().hasHeightForWidth())
        self.note_loop_title.setSizePolicy(sizePolicy)
        self.note_loop_title.setMinimumSize(QSize(64, 0))
        self.note_loop_title.setStyleSheet(u"background-color: rgb(159, 95, 95);\n"
"color: rgb(255, 255, 255);")

        self.verticalLayout.addWidget(self.note_loop_title)

        self.note_lyt = QHBoxLayout()
        self.note_lyt.setObjectName(u"note_lyt")
        self.force_pitch_name_cb = QCheckBox(self.centralwidget)
        self.force_pitch_name_cb.setObjectName(u"force_pitch_name_cb")
        self.force_pitch_name_cb.setChecked(False)

        self.note_lyt.addWidget(self.force_pitch_name_cb)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.note_lyt.addItem(self.horizontalSpacer)

        self.pitch_detect_cb = QCheckBox(self.centralwidget)
        self.pitch_detect_cb.setObjectName(u"pitch_detect_cb")

        self.note_lyt.addWidget(self.pitch_detect_cb)

        self.pitch_mode_cmb = QComboBox(self.centralwidget)
        self.pitch_mode_cmb.addItem("")
        self.pitch_mode_cmb.addItem("")
        self.pitch_mode_cmb.setObjectName(u"pitch_mode_cmb")
        self.pitch_mode_cmb.setEnabled(False)

        self.note_lyt.addWidget(self.pitch_mode_cmb)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.note_lyt.addItem(self.horizontalSpacer_4)

        self.pitch_fraction_l = QLabel(self.centralwidget)
        self.pitch_fraction_l.setObjectName(u"pitch_fraction_l")

        self.note_lyt.addWidget(self.pitch_fraction_l)

        self.pitch_fraction_cmb = QComboBox(self.centralwidget)
        self.pitch_fraction_cmb.addItem("")
        self.pitch_fraction_cmb.addItem("")
        self.pitch_fraction_cmb.addItem("")
        self.pitch_fraction_cmb.setObjectName(u"pitch_fraction_cmb")
        sizePolicy.setHeightForWidth(self.pitch_fraction_cmb.sizePolicy().hasHeightForWidth())
        self.pitch_fraction_cmb.setSizePolicy(sizePolicy)
        self.pitch_fraction_cmb.setFrame(False)

        self.note_lyt.addWidget(self.pitch_fraction_cmb)

        self.pitchfraction_dsb = QDoubleSpinBox(self.centralwidget)
        self.pitchfraction_dsb.setObjectName(u"pitchfraction_dsb")
        self.pitchfraction_dsb.setEnabled(False)
        self.pitchfraction_dsb.setFrame(False)
        self.pitchfraction_dsb.setAlignment(Qt.AlignCenter)
        self.pitchfraction_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.pitchfraction_dsb.setMinimum(-100.000000000000000)
        self.pitchfraction_dsb.setMaximum(100.000000000000000)

        self.note_lyt.addWidget(self.pitchfraction_dsb)

        self.horizontalSpacer_8 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.note_lyt.addItem(self.horizontalSpacer_8)

        self.transpose_l = QLabel(self.centralwidget)
        self.transpose_l.setObjectName(u"transpose_l")
        self.transpose_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.note_lyt.addWidget(self.transpose_l)

        self.transpose_sb = QSpinBox(self.centralwidget)
        self.transpose_sb.setObjectName(u"transpose_sb")
        self.transpose_sb.setFrame(False)
        self.transpose_sb.setAlignment(Qt.AlignCenter)
        self.transpose_sb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.transpose_sb.setMinimum(-127)
        self.transpose_sb.setMaximum(127)

        self.note_lyt.addWidget(self.transpose_sb)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.note_lyt.addItem(self.horizontalSpacer_3)

        self.use_loop_cb = QCheckBox(self.centralwidget)
        self.use_loop_cb.setObjectName(u"use_loop_cb")
        self.use_loop_cb.setChecked(True)

        self.note_lyt.addWidget(self.use_loop_cb)


        self.verticalLayout.addLayout(self.note_lyt)

        self.bake_pf_title = QLabel(self.centralwidget)
        self.bake_pf_title.setObjectName(u"bake_pf_title")
        sizePolicy.setHeightForWidth(self.bake_pf_title.sizePolicy().hasHeightForWidth())
        self.bake_pf_title.setSizePolicy(sizePolicy)
        self.bake_pf_title.setMinimumSize(QSize(64, 0))
        self.bake_pf_title.setStyleSheet(u"background-color: rgb(159, 95, 95);\n"
"color: rgb(255, 255, 255);")

        self.verticalLayout.addWidget(self.bake_pf_title)

        self.bake_pf_lyt = QHBoxLayout()
        self.bake_pf_lyt.setObjectName(u"bake_pf_lyt")
        self.bake_pf_cb = QCheckBox(self.centralwidget)
        self.bake_pf_cb.setObjectName(u"bake_pf_cb")

        self.bake_pf_lyt.addWidget(self.bake_pf_cb)

        self.pf_mode_wid = QWidget(self.centralwidget)
        self.pf_mode_wid.setObjectName(u"pf_mode_wid")
        self.pf_mode_wid.setEnabled(False)
        self.horizontalLayout = QHBoxLayout(self.pf_mode_wid)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.pf_mode_l = QLabel(self.pf_mode_wid)
        self.pf_mode_l.setObjectName(u"pf_mode_l")

        self.horizontalLayout.addWidget(self.pf_mode_l)

        self.pf_mode_cmb = QComboBox(self.pf_mode_wid)
        self.pf_mode_cmb.addItem("")
        self.pf_mode_cmb.addItem("")
        self.pf_mode_cmb.setObjectName(u"pf_mode_cmb")
        self.pf_mode_cmb.setFrame(False)

        self.horizontalLayout.addWidget(self.pf_mode_cmb)

        self.pf_th_dsb = QDoubleSpinBox(self.pf_mode_wid)
        self.pf_th_dsb.setObjectName(u"pf_th_dsb")
        self.pf_th_dsb.setEnabled(False)
        self.pf_th_dsb.setFrame(False)
        self.pf_th_dsb.setAlignment(Qt.AlignCenter)
        self.pf_th_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.pf_th_dsb.setDecimals(3)
        self.pf_th_dsb.setMaximum(100.000000000000000)
        self.pf_th_dsb.setValue(2.500000000000000)

        self.horizontalLayout.addWidget(self.pf_th_dsb)


        self.bake_pf_lyt.addWidget(self.pf_mode_wid)


        self.verticalLayout.addLayout(self.bake_pf_lyt)

        self.output_path_title_l = QLabel(self.centralwidget)
        self.output_path_title_l.setObjectName(u"output_path_title_l")
        sizePolicy.setHeightForWidth(self.output_path_title_l.sizePolicy().hasHeightForWidth())
        self.output_path_title_l.setSizePolicy(sizePolicy)
        self.output_path_title_l.setMinimumSize(QSize(64, 0))
        self.output_path_title_l.setStyleSheet(u"background-color: rgb(159, 95, 95);\n"
"color: rgb(255, 255, 255);")

        self.verticalLayout.addWidget(self.output_path_title_l)

        self.output_path_lyt = QHBoxLayout()
        self.output_path_lyt.setObjectName(u"output_path_lyt")
        self.output_path_l = QLabel(self.centralwidget)
        self.output_path_l.setObjectName(u"output_path_l")
        sizePolicy.setHeightForWidth(self.output_path_l.sizePolicy().hasHeightForWidth())
        self.output_path_l.setSizePolicy(sizePolicy)

        self.output_path_lyt.addWidget(self.output_path_l)

        self.set_output_path_tb = QToolButton(self.centralwidget)
        self.set_output_path_tb.setObjectName(u"set_output_path_tb")
        self.set_output_path_tb.setFont(font)

        self.output_path_lyt.addWidget(self.set_output_path_tb)


        self.verticalLayout.addLayout(self.output_path_lyt)

        self.settings_title_l = QLabel(self.centralwidget)
        self.settings_title_l.setObjectName(u"settings_title_l")
        sizePolicy.setHeightForWidth(self.settings_title_l.sizePolicy().hasHeightForWidth())
        self.settings_title_l.setSizePolicy(sizePolicy)
        self.settings_title_l.setMinimumSize(QSize(64, 0))
        self.settings_title_l.setStyleSheet(u"background-color: rgb(159, 95, 95);\n"
"color: rgb(255, 255, 255);")

        self.verticalLayout.addWidget(self.settings_title_l)

        self.output_file_lyt = QHBoxLayout()
        self.output_file_lyt.setObjectName(u"output_file_lyt")
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy2)

        self.output_file_lyt.addWidget(self.label_2)

        self.format_cmb = QComboBox(self.centralwidget)
        self.format_cmb.addItem("")
        self.format_cmb.addItem("")
        self.format_cmb.addItem("")
        self.format_cmb.setObjectName(u"format_cmb")
        sizePolicy.setHeightForWidth(self.format_cmb.sizePolicy().hasHeightForWidth())
        self.format_cmb.setSizePolicy(sizePolicy)
        self.format_cmb.setFrame(False)

        self.output_file_lyt.addWidget(self.format_cmb)

        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        sizePolicy2.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy2)

        self.output_file_lyt.addWidget(self.label)

        self.bitdepth_cmb = QComboBox(self.centralwidget)
        self.bitdepth_cmb.addItem("")
        self.bitdepth_cmb.addItem("")
        self.bitdepth_cmb.addItem("")
        self.bitdepth_cmb.addItem("")
        self.bitdepth_cmb.setObjectName(u"bitdepth_cmb")
        sizePolicy.setHeightForWidth(self.bitdepth_cmb.sizePolicy().hasHeightForWidth())
        self.bitdepth_cmb.setSizePolicy(sizePolicy)
        self.bitdepth_cmb.setFrame(False)

        self.output_file_lyt.addWidget(self.bitdepth_cmb)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.output_file_lyt.addItem(self.horizontalSpacer_5)

        self.bake_pf_options_wid = QWidget(self.centralwidget)
        self.bake_pf_options_wid.setObjectName(u"bake_pf_options_wid")
        self.bake_pf_options_wid.setEnabled(False)
        self.horizontalLayout_2 = QHBoxLayout(self.bake_pf_options_wid)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.add_suffix_cb = QCheckBox(self.bake_pf_options_wid)
        self.add_suffix_cb.setObjectName(u"add_suffix_cb")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.add_suffix_cb.sizePolicy().hasHeightForWidth())
        self.add_suffix_cb.setSizePolicy(sizePolicy3)
        self.add_suffix_cb.setChecked(True)

        self.horizontalLayout_2.addWidget(self.add_suffix_cb)

        self.add_suffix_le = QLineEdit(self.bake_pf_options_wid)
        self.add_suffix_le.setObjectName(u"add_suffix_le")
        sizePolicy3.setHeightForWidth(self.add_suffix_le.sizePolicy().hasHeightForWidth())
        self.add_suffix_le.setSizePolicy(sizePolicy3)
        self.add_suffix_le.setMaxLength(16)
        self.add_suffix_le.setFrame(False)

        self.horizontalLayout_2.addWidget(self.add_suffix_le)

        self.horizontalSpacer_6 = QSpacerItem(116, 17, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_6)

        self.no_overwriting_cb = QCheckBox(self.bake_pf_options_wid)
        self.no_overwriting_cb.setObjectName(u"no_overwriting_cb")
        self.no_overwriting_cb.setChecked(True)

        self.horizontalLayout_2.addWidget(self.no_overwriting_cb)


        self.output_file_lyt.addWidget(self.bake_pf_options_wid)


        self.verticalLayout.addLayout(self.output_file_lyt)

        self.line = QFrame(self.centralwidget)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout.addWidget(self.line)

        self.buttons_lyt = QGridLayout()
        self.buttons_lyt.setObjectName(u"buttons_lyt")
        self.process_sel_pb = QPushButton(self.centralwidget)
        self.process_sel_pb.setObjectName(u"process_sel_pb")
        sizePolicy3.setHeightForWidth(self.process_sel_pb.sizePolicy().hasHeightForWidth())
        self.process_sel_pb.setSizePolicy(sizePolicy3)
        self.process_sel_pb.setMinimumSize(QSize(160, 0))
        self.process_sel_pb.setFont(font)

        self.buttons_lyt.addWidget(self.process_sel_pb, 0, 0, 1, 1)

        self.process_pb = QPushButton(self.centralwidget)
        self.process_pb.setObjectName(u"process_pb")
        sizePolicy3.setHeightForWidth(self.process_pb.sizePolicy().hasHeightForWidth())
        self.process_pb.setSizePolicy(sizePolicy3)
        self.process_pb.setMinimumSize(QSize(160, 0))
        self.process_pb.setFont(font)
        self.process_pb.setStyleSheet(u"QPushButton{background-color: rgb(159, 95, 95);\n"
"color: rgb(255, 255, 255);}")

        self.buttons_lyt.addWidget(self.process_pb, 1, 0, 1, 1)


        self.verticalLayout.addLayout(self.buttons_lyt)

        self.progress_pb = QProgressBar(self.centralwidget)
        self.progress_pb.setObjectName(u"progress_pb")
        self.progress_pb.setStyleSheet(u"QProgressBar{border: none;}")
        self.progress_pb.setValue(0)
        self.progress_pb.setAlignment(Qt.AlignCenter)
        self.progress_pb.setTextVisible(False)
        self.progress_pb.setFormat(u"")

        self.verticalLayout.addWidget(self.progress_pb)

        rename_tool_mw.setCentralWidget(self.centralwidget)

        self.retranslateUi(rename_tool_mw)

        self.pitch_fraction_cmb.setCurrentIndex(0)
        self.pf_mode_cmb.setCurrentIndex(0)
        self.format_cmb.setCurrentIndex(0)
        self.bitdepth_cmb.setCurrentIndex(0)
        self.process_pb.setDefault(False)


        QMetaObject.connectSlotsByName(rename_tool_mw)
    # setupUi

    def retranslateUi(self, rename_tool_mw):
        rename_tool_mw.setWindowTitle(QCoreApplication.translate("rename_tool_mw", u"Rename Sample Tool", None))
        self.files_title_l.setText(QCoreApplication.translate("rename_tool_mw", u"Input Files", None))
#if QT_CONFIG(tooltip)
        self.files_lw.setToolTip(QCoreApplication.translate("rename_tool_mw", u"Drag and drop files or directories\n"
"Right click for context actions\n"
"Double-click to play", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.set_files_tb.setToolTip(QCoreApplication.translate("rename_tool_mw", u"Set files to process", None))
#endif // QT_CONFIG(tooltip)
        self.set_files_tb.setText(QCoreApplication.translate("rename_tool_mw", u"...", None))
        self.naming_title_l.setText(QCoreApplication.translate("rename_tool_mw", u"Naming", None))
        self.groupname_l.setText(QCoreApplication.translate("rename_tool_mw", u"Group Name", None))
#if QT_CONFIG(tooltip)
        self.groupname_le.setToolTip(QCoreApplication.translate("rename_tool_mw", u"Override group/base name", None))
#endif // QT_CONFIG(tooltip)
        self.groupname_le.setText("")
        self.repstr_l.setText(QCoreApplication.translate("rename_tool_mw", u"Search/Replace", None))
#if QT_CONFIG(tooltip)
        self.repstr_le.setToolTip(QCoreApplication.translate("rename_tool_mw", u"Enter 2 words separated by commas\n"
"The 1st word is replaced by the 2nd\n"
"The token is then available using {repstr} in the target pattern", None))
#endif // QT_CONFIG(tooltip)
        self.repstr_le.setText("")
#if QT_CONFIG(tooltip)
        self.src_pattern_pb.setToolTip(QCoreApplication.translate("rename_tool_mw", u"Pattern used to convert sample names to attribute values\n"
"Click for context menu and some pattern examples", None))
#endif // QT_CONFIG(tooltip)
        self.src_pattern_pb.setText(QCoreApplication.translate("rename_tool_mw", u"Source Pattern", None))
#if QT_CONFIG(tooltip)
        self.src_pattern_le.setToolTip(QCoreApplication.translate("rename_tool_mw", u"Extract attributes following this pattern\n"
"Attributes names enclosed in curly braces {}", None))
#endif // QT_CONFIG(tooltip)
        self.src_pattern_le.setText("")
#if QT_CONFIG(tooltip)
        self.tgt_pattern_pb.setToolTip(QCoreApplication.translate("rename_tool_mw", u"Pattern used to compose file names from attributes\n"
"Click for context menu and some pattern examples", None))
#endif // QT_CONFIG(tooltip)
        self.tgt_pattern_pb.setText(QCoreApplication.translate("rename_tool_mw", u"Target Pattern", None))
#if QT_CONFIG(tooltip)
        self.tgt_pattern_le.setToolTip(QCoreApplication.translate("rename_tool_mw", u"Recompose attributes following this pattern\n"
"Attributes names enclosed in curly braces {}", None))
#endif // QT_CONFIG(tooltip)
        self.tgt_pattern_le.setText("")
        self.prefix_l.setText(QCoreApplication.translate("rename_tool_mw", u"Prefix", None))
        self.prefix_le.setText("")
        self.suffix_l.setText(QCoreApplication.translate("rename_tool_mw", u"Suffix", None))
        self.suffix_le.setText("")
        self.note_loop_title.setText(QCoreApplication.translate("rename_tool_mw", u"Note/Loop", None))
#if QT_CONFIG(tooltip)
        self.force_pitch_name_cb.setToolTip(QCoreApplication.translate("rename_tool_mw", u"Attempt to read the note from the whole sample name\n"
"Might work even when the name is badly formatted", None))
#endif // QT_CONFIG(tooltip)
        self.force_pitch_name_cb.setText(QCoreApplication.translate("rename_tool_mw", u"Force Note From Name", None))
#if QT_CONFIG(tooltip)
        self.pitch_detect_cb.setToolTip(QCoreApplication.translate("rename_tool_mw", u"Override any existing pitch information", None))
#endif // QT_CONFIG(tooltip)
        self.pitch_detect_cb.setText(QCoreApplication.translate("rename_tool_mw", u"Pitch Detect", None))
        self.pitch_mode_cmb.setItemText(0, QCoreApplication.translate("rename_tool_mw", u"yin", None))
        self.pitch_mode_cmb.setItemText(1, QCoreApplication.translate("rename_tool_mw", u"corr", None))

#if QT_CONFIG(tooltip)
        self.pitch_mode_cmb.setToolTip(QCoreApplication.translate("rename_tool_mw", u"Pitch detection algorithm\n"
"NOTE:	No algorithm is perfect so always double-check results in an audio editor such as Audacity or RX using spectrum/frequency analysis\n"
"	Pitch detection does not work on bell-like sounds\n"
"\n"
"'yin'	good results and rather fast, best combined with fine tuning\n"
"'corr'	auto-correlation, fastest method but can be wrong about octave", None))
#endif // QT_CONFIG(tooltip)
        self.pitch_fraction_l.setText(QCoreApplication.translate("rename_tool_mw", u"Pitch Fraction", None))
        self.pitch_fraction_cmb.setItemText(0, QCoreApplication.translate("rename_tool_mw", u"keep", None))
        self.pitch_fraction_cmb.setItemText(1, QCoreApplication.translate("rename_tool_mw", u"override", None))
        self.pitch_fraction_cmb.setItemText(2, QCoreApplication.translate("rename_tool_mw", u"fine_tune", None))

#if QT_CONFIG(tooltip)
        self.pitch_fraction_cmb.setToolTip(QCoreApplication.translate("rename_tool_mw", u"keep: use embedded value (if applicable)\n"
"override: replace with a given value\n"
"fine_tune: Detect pitch fraction using autocorrelation", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.pitchfraction_dsb.setToolTip(QCoreApplication.translate("rename_tool_mw", u"Pitch fraction in semitone cents", None))
#endif // QT_CONFIG(tooltip)
        self.transpose_l.setText(QCoreApplication.translate("rename_tool_mw", u"Transpose", None))
#if QT_CONFIG(tooltip)
        self.transpose_sb.setToolTip(QCoreApplication.translate("rename_tool_mw", u"Transpose note in semitones", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.use_loop_cb.setToolTip(QCoreApplication.translate("rename_tool_mw", u"Use loop if present\n"
"Clear loop information when disabled", None))
#endif // QT_CONFIG(tooltip)
        self.use_loop_cb.setText(QCoreApplication.translate("rename_tool_mw", u"Use Loop", None))
        self.bake_pf_title.setText(QCoreApplication.translate("rename_tool_mw", u"Bake Pitch Fraction", None))
#if QT_CONFIG(tooltip)
        self.bake_pf_cb.setToolTip(QCoreApplication.translate("rename_tool_mw", u"Apply pitch fraction to the samples by resampling\n"
"Allow to use tuned samples with samplers not supporting pitch fraction information\n"
"SoX resamping library with VHQ preset is used for optimal quality\n"
"\n"
"NOTE: As opposed to just adding pitch fraction information, this is a destructive process", None))
#endif // QT_CONFIG(tooltip)
        self.bake_pf_cb.setText(QCoreApplication.translate("rename_tool_mw", u"Bake Pitch Fraction", None))
        self.pf_mode_l.setText(QCoreApplication.translate("rename_tool_mw", u"Pitch Fraction Mode", None))
        self.pf_mode_cmb.setItemText(0, QCoreApplication.translate("rename_tool_mw", u"on", None))
        self.pf_mode_cmb.setItemText(1, QCoreApplication.translate("rename_tool_mw", u"mean_scale", None))

#if QT_CONFIG(tooltip)
        self.pf_mode_cmb.setToolTip(QCoreApplication.translate("rename_tool_mw", u"Control pitch fraction usage\n"
"\n"
"'on'	Use pitch fraction\n"
"\n"
"'mean_scale'	Scale deviation from mean to given value\n"
"(results will vary depending on the number of processed samples)", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.pf_th_dsb.setToolTip(QCoreApplication.translate("rename_tool_mw", u"Pitch fraction deviation scale in semitone cents", None))
#endif // QT_CONFIG(tooltip)
        self.output_path_title_l.setText(QCoreApplication.translate("rename_tool_mw", u"Output Path", None))
        self.output_path_l.setText("")
#if QT_CONFIG(tooltip)
        self.set_output_path_tb.setToolTip(QCoreApplication.translate("rename_tool_mw", u"Set output path\n"
"Process files in their respective directory if empty", None))
#endif // QT_CONFIG(tooltip)
        self.set_output_path_tb.setText(QCoreApplication.translate("rename_tool_mw", u"...", None))
        self.settings_title_l.setText(QCoreApplication.translate("rename_tool_mw", u"File Settings", None))
        self.label_2.setText(QCoreApplication.translate("rename_tool_mw", u"Format", None))
        self.format_cmb.setItemText(0, QCoreApplication.translate("rename_tool_mw", u"same", None))
        self.format_cmb.setItemText(1, QCoreApplication.translate("rename_tool_mw", u"wav", None))
        self.format_cmb.setItemText(2, QCoreApplication.translate("rename_tool_mw", u"flac", None))

#if QT_CONFIG(tooltip)
        self.format_cmb.setToolTip(QCoreApplication.translate("rename_tool_mw", u"flac only allows integer format up to 24 bits\n"
"Use wav for 32 bits float", None))
#endif // QT_CONFIG(tooltip)
        self.label.setText(QCoreApplication.translate("rename_tool_mw", u"Bit Depth", None))
        self.bitdepth_cmb.setItemText(0, QCoreApplication.translate("rename_tool_mw", u"same", None))
        self.bitdepth_cmb.setItemText(1, QCoreApplication.translate("rename_tool_mw", u"16", None))
        self.bitdepth_cmb.setItemText(2, QCoreApplication.translate("rename_tool_mw", u"24", None))
        self.bitdepth_cmb.setItemText(3, QCoreApplication.translate("rename_tool_mw", u"32", None))

        self.add_suffix_cb.setText(QCoreApplication.translate("rename_tool_mw", u"Add suffix", None))
#if QT_CONFIG(tooltip)
        self.add_suffix_le.setToolTip(QCoreApplication.translate("rename_tool_mw", u"Suffix added to the base name", None))
#endif // QT_CONFIG(tooltip)
        self.add_suffix_le.setText(QCoreApplication.translate("rename_tool_mw", u"_tuned", None))
#if QT_CONFIG(tooltip)
        self.no_overwriting_cb.setToolTip(QCoreApplication.translate("rename_tool_mw", u"Avoid overwriting original files by moving them to a backup directory", None))
#endif // QT_CONFIG(tooltip)
        self.no_overwriting_cb.setText(QCoreApplication.translate("rename_tool_mw", u"Avoid Overwriting", None))
#if QT_CONFIG(tooltip)
        self.process_sel_pb.setToolTip(QCoreApplication.translate("rename_tool_mw", u"Process selected item(s) with current settings", None))
#endif // QT_CONFIG(tooltip)
        self.process_sel_pb.setText(QCoreApplication.translate("rename_tool_mw", u"Process Selected", None))
#if QT_CONFIG(tooltip)
        self.process_pb.setToolTip(QCoreApplication.translate("rename_tool_mw", u"Process all input files with current settings", None))
#endif // QT_CONFIG(tooltip)
        self.process_pb.setText(QCoreApplication.translate("rename_tool_mw", u"Batch Process", None))
    # retranslateUi

