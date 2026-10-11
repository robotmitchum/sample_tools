# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'upsample_tool.ui'
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

class Ui_upsample_tool_mw(object):
    def setupUi(self, upsample_tool_mw):
        if not upsample_tool_mw.objectName():
            upsample_tool_mw.setObjectName(u"upsample_tool_mw")
        upsample_tool_mw.resize(640, 768)
        upsample_tool_mw.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        self.centralwidget = QWidget(upsample_tool_mw)
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
        self.files_title_l.setStyleSheet(u"background-color: rgb(191, 127, 63);\n"
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

        self.options_title_l = QLabel(self.centralwidget)
        self.options_title_l.setObjectName(u"options_title_l")
        sizePolicy.setHeightForWidth(self.options_title_l.sizePolicy().hasHeightForWidth())
        self.options_title_l.setSizePolicy(sizePolicy)
        self.options_title_l.setMinimumSize(QSize(64, 0))
        self.options_title_l.setStyleSheet(u"background-color: rgb(191, 127, 63);\n"
"color: rgb(255, 255, 255);")

        self.verticalLayout.addWidget(self.options_title_l)

        self.declip_lyt = QHBoxLayout()
        self.declip_lyt.setObjectName(u"declip_lyt")
        self.declip_cb = QCheckBox(self.centralwidget)
        self.declip_cb.setObjectName(u"declip_cb")
        self.declip_cb.setChecked(True)

        self.declip_lyt.addWidget(self.declip_cb)

        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.declip_lyt.addItem(self.horizontalSpacer_7)

        self.declip_wid = QWidget(self.centralwidget)
        self.declip_wid.setObjectName(u"declip_wid")
        self.horizontalLayout_2 = QHBoxLayout(self.declip_wid)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.declip_th_l = QLabel(self.declip_wid)
        self.declip_th_l.setObjectName(u"declip_th_l")
        self.declip_th_l.setMinimumSize(QSize(48, 0))
        self.declip_th_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_2.addWidget(self.declip_th_l)

        self.declip_th_dsb = QDoubleSpinBox(self.declip_wid)
        self.declip_th_dsb.setObjectName(u"declip_th_dsb")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.declip_th_dsb.sizePolicy().hasHeightForWidth())
        self.declip_th_dsb.setSizePolicy(sizePolicy1)
        self.declip_th_dsb.setFrame(False)
        self.declip_th_dsb.setAlignment(Qt.AlignCenter)
        self.declip_th_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.declip_th_dsb.setMinimum(-12.000000000000000)
        self.declip_th_dsb.setMaximum(0.000000000000000)
        self.declip_th_dsb.setSingleStep(0.100000000000000)
        self.declip_th_dsb.setValue(-0.100000000000000)

        self.horizontalLayout_2.addWidget(self.declip_th_dsb)

        self.declip_mix_l = QLabel(self.declip_wid)
        self.declip_mix_l.setObjectName(u"declip_mix_l")
        self.declip_mix_l.setMinimumSize(QSize(48, 0))
        self.declip_mix_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_2.addWidget(self.declip_mix_l)

        self.declip_mix_dsb = QDoubleSpinBox(self.declip_wid)
        self.declip_mix_dsb.setObjectName(u"declip_mix_dsb")
        sizePolicy1.setHeightForWidth(self.declip_mix_dsb.sizePolicy().hasHeightForWidth())
        self.declip_mix_dsb.setSizePolicy(sizePolicy1)
        self.declip_mix_dsb.setMinimumSize(QSize(48, 0))
        self.declip_mix_dsb.setFrame(False)
        self.declip_mix_dsb.setAlignment(Qt.AlignCenter)
        self.declip_mix_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.declip_mix_dsb.setDecimals(3)
        self.declip_mix_dsb.setMaximum(1.000000000000000)
        self.declip_mix_dsb.setSingleStep(0.100000000000000)
        self.declip_mix_dsb.setValue(0.250000000000000)

        self.horizontalLayout_2.addWidget(self.declip_mix_dsb)


        self.declip_lyt.addWidget(self.declip_wid)


        self.verticalLayout.addLayout(self.declip_lyt)

        self.line_2 = QFrame(self.centralwidget)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.Shape.HLine)
        self.line_2.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout.addWidget(self.line_2)

        self.denoise_lyt = QHBoxLayout()
        self.denoise_lyt.setObjectName(u"denoise_lyt")
        self.denoise_cb = QCheckBox(self.centralwidget)
        self.denoise_cb.setObjectName(u"denoise_cb")
        self.denoise_cb.setChecked(True)

        self.denoise_lyt.addWidget(self.denoise_cb)

        self.horizontalSpacer_8 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.denoise_lyt.addItem(self.horizontalSpacer_8)

        self.denoise_wid = QWidget(self.centralwidget)
        self.denoise_wid.setObjectName(u"denoise_wid")
        self.horizontalLayout_3 = QHBoxLayout(self.denoise_wid)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.denoise_mode_l = QLabel(self.denoise_wid)
        self.denoise_mode_l.setObjectName(u"denoise_mode_l")
        self.denoise_mode_l.setMinimumSize(QSize(48, 0))
        self.denoise_mode_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_3.addWidget(self.denoise_mode_l)

        self.denoise_mode_cmb = QComboBox(self.denoise_wid)
        self.denoise_mode_cmb.addItem("")
        self.denoise_mode_cmb.addItem("")
        self.denoise_mode_cmb.setObjectName(u"denoise_mode_cmb")

        self.horizontalLayout_3.addWidget(self.denoise_mode_cmb)

        self.denoise_mix_l = QLabel(self.denoise_wid)
        self.denoise_mix_l.setObjectName(u"denoise_mix_l")
        self.denoise_mix_l.setMinimumSize(QSize(48, 0))
        self.denoise_mix_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_3.addWidget(self.denoise_mix_l)

        self.denoise_mix_dsb = QDoubleSpinBox(self.denoise_wid)
        self.denoise_mix_dsb.setObjectName(u"denoise_mix_dsb")
        self.denoise_mix_dsb.setMinimumSize(QSize(48, 0))
        self.denoise_mix_dsb.setFrame(False)
        self.denoise_mix_dsb.setAlignment(Qt.AlignCenter)
        self.denoise_mix_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.denoise_mix_dsb.setDecimals(3)
        self.denoise_mix_dsb.setMaximum(4.000000000000000)
        self.denoise_mix_dsb.setSingleStep(0.100000000000000)
        self.denoise_mix_dsb.setValue(0.990000000000000)

        self.horizontalLayout_3.addWidget(self.denoise_mix_dsb)


        self.denoise_lyt.addWidget(self.denoise_wid)


        self.verticalLayout.addLayout(self.denoise_lyt)

        self.noise_path_lyt = QHBoxLayout()
        self.noise_path_lyt.setObjectName(u"noise_path_lyt")
        self.noise_path_wid = QWidget(self.centralwidget)
        self.noise_path_wid.setObjectName(u"noise_path_wid")
        self.noise_path_wid.setEnabled(False)
        self.horizontalLayout = QHBoxLayout(self.noise_path_wid)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.noise_path_l = QLabel(self.noise_path_wid)
        self.noise_path_l.setObjectName(u"noise_path_l")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.noise_path_l.sizePolicy().hasHeightForWidth())
        self.noise_path_l.setSizePolicy(sizePolicy2)

        self.horizontalLayout.addWidget(self.noise_path_l)

        self.set_noise_path_tb = QToolButton(self.noise_path_wid)
        self.set_noise_path_tb.setObjectName(u"set_noise_path_tb")
        self.set_noise_path_tb.setFont(font)

        self.horizontalLayout.addWidget(self.set_noise_path_tb)


        self.noise_path_lyt.addWidget(self.noise_path_wid)


        self.verticalLayout.addLayout(self.noise_path_lyt)

        self.line_3 = QFrame(self.centralwidget)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setFrameShape(QFrame.Shape.HLine)
        self.line_3.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout.addWidget(self.line_3)

        self.filter_side_lyt = QHBoxLayout()
        self.filter_side_lyt.setObjectName(u"filter_side_lyt")
        self.upsample_cb = QCheckBox(self.centralwidget)
        self.upsample_cb.setObjectName(u"upsample_cb")
        self.upsample_cb.setChecked(True)

        self.filter_side_lyt.addWidget(self.upsample_cb)

        self.upsample_wid = QWidget(self.centralwidget)
        self.upsample_wid.setObjectName(u"upsample_wid")
        self.horizontalLayout_4 = QHBoxLayout(self.upsample_wid)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.f_max_l = QLabel(self.upsample_wid)
        self.f_max_l.setObjectName(u"f_max_l")
        self.f_max_l.setMinimumSize(QSize(48, 0))
        self.f_max_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_4.addWidget(self.f_max_l)

        self.f_max_mode_cmb = QComboBox(self.upsample_wid)
        self.f_max_mode_cmb.addItem("")
        self.f_max_mode_cmb.addItem("")
        self.f_max_mode_cmb.setObjectName(u"f_max_mode_cmb")

        self.horizontalLayout_4.addWidget(self.f_max_mode_cmb)

        self.f_max_sb = QSpinBox(self.upsample_wid)
        self.f_max_sb.setObjectName(u"f_max_sb")
        self.f_max_sb.setEnabled(False)
        self.f_max_sb.setMinimumSize(QSize(48, 0))
        self.f_max_sb.setFrame(False)
        self.f_max_sb.setAlignment(Qt.AlignCenter)
        self.f_max_sb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.f_max_sb.setMinimum(0)
        self.f_max_sb.setMaximum(24000)
        self.f_max_sb.setValue(5512)

        self.horizontalLayout_4.addWidget(self.f_max_sb)

        self.target_sr_l = QLabel(self.upsample_wid)
        self.target_sr_l.setObjectName(u"target_sr_l")
        self.target_sr_l.setMinimumSize(QSize(48, 0))
        self.target_sr_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_4.addWidget(self.target_sr_l)

        self.target_sr_sb = QSpinBox(self.upsample_wid)
        self.target_sr_sb.setObjectName(u"target_sr_sb")
        self.target_sr_sb.setMinimumSize(QSize(48, 0))
        self.target_sr_sb.setFrame(False)
        self.target_sr_sb.setAlignment(Qt.AlignCenter)
        self.target_sr_sb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.target_sr_sb.setMinimum(32000)
        self.target_sr_sb.setMaximum(96000)
        self.target_sr_sb.setValue(48000)

        self.horizontalLayout_4.addWidget(self.target_sr_sb)

        self.upsample_mix_l = QLabel(self.upsample_wid)
        self.upsample_mix_l.setObjectName(u"upsample_mix_l")
        self.upsample_mix_l.setMinimumSize(QSize(48, 0))
        self.upsample_mix_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_4.addWidget(self.upsample_mix_l)

        self.upsample_mix_dsb = QDoubleSpinBox(self.upsample_wid)
        self.upsample_mix_dsb.setObjectName(u"upsample_mix_dsb")
        self.upsample_mix_dsb.setMinimumSize(QSize(48, 0))
        self.upsample_mix_dsb.setFrame(False)
        self.upsample_mix_dsb.setAlignment(Qt.AlignCenter)
        self.upsample_mix_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.upsample_mix_dsb.setDecimals(3)
        self.upsample_mix_dsb.setMaximum(2.000000000000000)
        self.upsample_mix_dsb.setSingleStep(0.100000000000000)
        self.upsample_mix_dsb.setValue(1.000000000000000)

        self.horizontalLayout_4.addWidget(self.upsample_mix_dsb)


        self.filter_side_lyt.addWidget(self.upsample_wid)


        self.verticalLayout.addLayout(self.filter_side_lyt)

        self.output_path_title_l = QLabel(self.centralwidget)
        self.output_path_title_l.setObjectName(u"output_path_title_l")
        sizePolicy.setHeightForWidth(self.output_path_title_l.sizePolicy().hasHeightForWidth())
        self.output_path_title_l.setSizePolicy(sizePolicy)
        self.output_path_title_l.setMinimumSize(QSize(64, 0))
        self.output_path_title_l.setStyleSheet(u"background-color: rgb(191, 127, 63);\n"
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
        self.settings_title_l.setStyleSheet(u"background-color: rgb(191, 127, 63);\n"
"color: rgb(255, 255, 255);")

        self.verticalLayout.addWidget(self.settings_title_l)

        self.file_settings_lyt = QHBoxLayout()
        self.file_settings_lyt.setObjectName(u"file_settings_lyt")
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        sizePolicy1.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy1)

        self.file_settings_lyt.addWidget(self.label_2)

        self.format_cmb = QComboBox(self.centralwidget)
        self.format_cmb.addItem("")
        self.format_cmb.addItem("")
        self.format_cmb.addItem("")
        self.format_cmb.setObjectName(u"format_cmb")
        sizePolicy.setHeightForWidth(self.format_cmb.sizePolicy().hasHeightForWidth())
        self.format_cmb.setSizePolicy(sizePolicy)
        self.format_cmb.setFrame(False)

        self.file_settings_lyt.addWidget(self.format_cmb)

        self.bd_l = QLabel(self.centralwidget)
        self.bd_l.setObjectName(u"bd_l")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.bd_l.sizePolicy().hasHeightForWidth())
        self.bd_l.setSizePolicy(sizePolicy3)

        self.file_settings_lyt.addWidget(self.bd_l)

        self.bitdepth_cmb = QComboBox(self.centralwidget)
        self.bitdepth_cmb.addItem("")
        self.bitdepth_cmb.addItem("")
        self.bitdepth_cmb.addItem("")
        self.bitdepth_cmb.addItem("")
        self.bitdepth_cmb.setObjectName(u"bitdepth_cmb")
        sizePolicy.setHeightForWidth(self.bitdepth_cmb.sizePolicy().hasHeightForWidth())
        self.bitdepth_cmb.setSizePolicy(sizePolicy)
        self.bitdepth_cmb.setFrame(False)

        self.file_settings_lyt.addWidget(self.bitdepth_cmb)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.file_settings_lyt.addItem(self.horizontalSpacer_2)

        self.add_suffix_cb = QCheckBox(self.centralwidget)
        self.add_suffix_cb.setObjectName(u"add_suffix_cb")
        sizePolicy3.setHeightForWidth(self.add_suffix_cb.sizePolicy().hasHeightForWidth())
        self.add_suffix_cb.setSizePolicy(sizePolicy3)
        self.add_suffix_cb.setChecked(True)

        self.file_settings_lyt.addWidget(self.add_suffix_cb)

        self.suffix_le = QLineEdit(self.centralwidget)
        self.suffix_le.setObjectName(u"suffix_le")
        sizePolicy3.setHeightForWidth(self.suffix_le.sizePolicy().hasHeightForWidth())
        self.suffix_le.setSizePolicy(sizePolicy3)
        self.suffix_le.setMaxLength(16)
        self.suffix_le.setFrame(False)

        self.file_settings_lyt.addWidget(self.suffix_le)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.file_settings_lyt.addItem(self.horizontalSpacer_4)

        self.no_overwriting_cb = QCheckBox(self.centralwidget)
        self.no_overwriting_cb.setObjectName(u"no_overwriting_cb")
        self.no_overwriting_cb.setChecked(True)

        self.file_settings_lyt.addWidget(self.no_overwriting_cb)


        self.verticalLayout.addLayout(self.file_settings_lyt)

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

        self.buttons_lyt.addWidget(self.process_sel_pb, 0, 1, 1, 1)

        self.preview_pb = QPushButton(self.centralwidget)
        self.preview_pb.setObjectName(u"preview_pb")
        sizePolicy3.setHeightForWidth(self.preview_pb.sizePolicy().hasHeightForWidth())
        self.preview_pb.setSizePolicy(sizePolicy3)
        self.preview_pb.setMinimumSize(QSize(160, 0))
        self.preview_pb.setFont(font)

        self.buttons_lyt.addWidget(self.preview_pb, 0, 0, 1, 1)

        self.process_pb = QPushButton(self.centralwidget)
        self.process_pb.setObjectName(u"process_pb")
        sizePolicy3.setHeightForWidth(self.process_pb.sizePolicy().hasHeightForWidth())
        self.process_pb.setSizePolicy(sizePolicy3)
        self.process_pb.setMinimumSize(QSize(160, 0))
        self.process_pb.setFont(font)
        self.process_pb.setStyleSheet(u"QPushButton{background-color: rgb(191, 127, 63);color: rgb(255, 255, 255);}")

        self.buttons_lyt.addWidget(self.process_pb, 1, 1, 1, 1)


        self.verticalLayout.addLayout(self.buttons_lyt)

        self.progress_pb = QProgressBar(self.centralwidget)
        self.progress_pb.setObjectName(u"progress_pb")
        self.progress_pb.setStyleSheet(u"QProgressBar{border: none;}")
        self.progress_pb.setValue(0)
        self.progress_pb.setAlignment(Qt.AlignCenter)
        self.progress_pb.setTextVisible(False)
        self.progress_pb.setFormat(u"")

        self.verticalLayout.addWidget(self.progress_pb)

        upsample_tool_mw.setCentralWidget(self.centralwidget)

        self.retranslateUi(upsample_tool_mw)

        self.format_cmb.setCurrentIndex(0)
        self.bitdepth_cmb.setCurrentIndex(0)
        self.process_pb.setDefault(False)


        QMetaObject.connectSlotsByName(upsample_tool_mw)
    # setupUi

    def retranslateUi(self, upsample_tool_mw):
        upsample_tool_mw.setWindowTitle(QCoreApplication.translate("upsample_tool_mw", u"Upsample Tool", None))
        self.files_title_l.setText(QCoreApplication.translate("upsample_tool_mw", u"Input Files", None))
#if QT_CONFIG(tooltip)
        self.files_lw.setToolTip(QCoreApplication.translate("upsample_tool_mw", u"Drag and drop files or directories\n"
"Right click for context actions\n"
"Double-click to play", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.set_files_tb.setToolTip(QCoreApplication.translate("upsample_tool_mw", u"Set files to process", None))
#endif // QT_CONFIG(tooltip)
        self.set_files_tb.setText(QCoreApplication.translate("upsample_tool_mw", u"...", None))
        self.options_title_l.setText(QCoreApplication.translate("upsample_tool_mw", u"Options", None))
#if QT_CONFIG(tooltip)
        self.declip_cb.setToolTip(QCoreApplication.translate("upsample_tool_mw", u"Simplistic declip to help denoise and upsampling", None))
#endif // QT_CONFIG(tooltip)
        self.declip_cb.setText(QCoreApplication.translate("upsample_tool_mw", u"Declip", None))
        self.declip_th_l.setText(QCoreApplication.translate("upsample_tool_mw", u"Threshold", None))
#if QT_CONFIG(tooltip)
        self.declip_th_dsb.setToolTip(QCoreApplication.translate("upsample_tool_mw", u"Declip threshold in dB", None))
#endif // QT_CONFIG(tooltip)
        self.declip_mix_l.setText(QCoreApplication.translate("upsample_tool_mw", u"Mix", None))
#if QT_CONFIG(tooltip)
        self.declip_mix_dsb.setToolTip(QCoreApplication.translate("upsample_tool_mw", u"Blend with original audio", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.denoise_cb.setToolTip(QCoreApplication.translate("upsample_tool_mw", u"Denoise audio using spectral gating", None))
#endif // QT_CONFIG(tooltip)
        self.denoise_cb.setText(QCoreApplication.translate("upsample_tool_mw", u"Denoise", None))
        self.denoise_mode_l.setText(QCoreApplication.translate("upsample_tool_mw", u"Mode", None))
        self.denoise_mode_cmb.setItemText(0, QCoreApplication.translate("upsample_tool_mw", u"8bits", None))
        self.denoise_mode_cmb.setItemText(1, QCoreApplication.translate("upsample_tool_mw", u"custom", None))

#if QT_CONFIG(tooltip)
        self.denoise_mode_cmb.setToolTip(QCoreApplication.translate("upsample_tool_mw", u"Noise type\n"
"\n"
"8bits: Reduce 8 bits quantization noise\n"
"Custom: Use custom noise profile", None))
#endif // QT_CONFIG(tooltip)
        self.denoise_mix_l.setText(QCoreApplication.translate("upsample_tool_mw", u"Mix", None))
#if QT_CONFIG(tooltip)
        self.denoise_mix_dsb.setToolTip(QCoreApplication.translate("upsample_tool_mw", u"Noise reduction factor", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.noise_path_l.setToolTip(QCoreApplication.translate("upsample_tool_mw", u"Path to custom noise profile", None))
#endif // QT_CONFIG(tooltip)
        self.noise_path_l.setText("")
#if QT_CONFIG(tooltip)
        self.set_noise_path_tb.setToolTip(QCoreApplication.translate("upsample_tool_mw", u"Set noise profile to use for denoise", None))
#endif // QT_CONFIG(tooltip)
        self.set_noise_path_tb.setText(QCoreApplication.translate("upsample_tool_mw", u"...", None))
#if QT_CONFIG(tooltip)
        self.upsample_cb.setToolTip(QCoreApplication.translate("upsample_tool_mw", u"Audio up-resolution by replicating upper frequency band", None))
#endif // QT_CONFIG(tooltip)
        self.upsample_cb.setText(QCoreApplication.translate("upsample_tool_mw", u"Upsample", None))
        self.f_max_l.setText(QCoreApplication.translate("upsample_tool_mw", u"Max Freq", None))
        self.f_max_mode_cmb.setItemText(0, QCoreApplication.translate("upsample_tool_mw", u"max", None))
        self.f_max_mode_cmb.setItemText(1, QCoreApplication.translate("upsample_tool_mw", u"custom", None))

#if QT_CONFIG(tooltip)
        self.f_max_mode_cmb.setToolTip(QCoreApplication.translate("upsample_tool_mw", u"Upper frequency threshold before up-sampling\n"
"\n"
"max: Nyquist, Input sampling rate / 2 (default)\n"
"custom: When input audio features a cutoff lower than Nyquist", None))
#endif // QT_CONFIG(tooltip)
        self.target_sr_l.setText(QCoreApplication.translate("upsample_tool_mw", u"Target Sample Rate", None))
        self.upsample_mix_l.setText(QCoreApplication.translate("upsample_tool_mw", u"Mix", None))
#if QT_CONFIG(tooltip)
        self.upsample_mix_dsb.setToolTip(QCoreApplication.translate("upsample_tool_mw", u"Replicated bands multiplying factor", None))
#endif // QT_CONFIG(tooltip)
        self.output_path_title_l.setText(QCoreApplication.translate("upsample_tool_mw", u"Output Path", None))
#if QT_CONFIG(tooltip)
        self.output_path_l.setToolTip(QCoreApplication.translate("upsample_tool_mw", u"Output path", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.set_output_path_tb.setToolTip(QCoreApplication.translate("upsample_tool_mw", u"Set output path\n"
"Process files in their respective directory if empty", None))
#endif // QT_CONFIG(tooltip)
        self.set_output_path_tb.setText(QCoreApplication.translate("upsample_tool_mw", u"...", None))
        self.settings_title_l.setText(QCoreApplication.translate("upsample_tool_mw", u"File Settings", None))
        self.label_2.setText(QCoreApplication.translate("upsample_tool_mw", u"Format", None))
        self.format_cmb.setItemText(0, QCoreApplication.translate("upsample_tool_mw", u"same", None))
        self.format_cmb.setItemText(1, QCoreApplication.translate("upsample_tool_mw", u"wav", None))
        self.format_cmb.setItemText(2, QCoreApplication.translate("upsample_tool_mw", u"flac", None))

#if QT_CONFIG(tooltip)
        self.format_cmb.setToolTip(QCoreApplication.translate("upsample_tool_mw", u"flac only allows integer format up to 24 bits\n"
"Use wav for 32 bits float", None))
#endif // QT_CONFIG(tooltip)
        self.bd_l.setText(QCoreApplication.translate("upsample_tool_mw", u"Bit Depth", None))
        self.bitdepth_cmb.setItemText(0, QCoreApplication.translate("upsample_tool_mw", u"same", None))
        self.bitdepth_cmb.setItemText(1, QCoreApplication.translate("upsample_tool_mw", u"16", None))
        self.bitdepth_cmb.setItemText(2, QCoreApplication.translate("upsample_tool_mw", u"24", None))
        self.bitdepth_cmb.setItemText(3, QCoreApplication.translate("upsample_tool_mw", u"32", None))

        self.add_suffix_cb.setText(QCoreApplication.translate("upsample_tool_mw", u"Add suffix", None))
#if QT_CONFIG(tooltip)
        self.suffix_le.setToolTip(QCoreApplication.translate("upsample_tool_mw", u"Suffix added to the base name", None))
#endif // QT_CONFIG(tooltip)
        self.suffix_le.setText(QCoreApplication.translate("upsample_tool_mw", u"_up", None))
#if QT_CONFIG(tooltip)
        self.no_overwriting_cb.setToolTip(QCoreApplication.translate("upsample_tool_mw", u"Avoid overwriting original files by moving them to a backup directory", None))
#endif // QT_CONFIG(tooltip)
        self.no_overwriting_cb.setText(QCoreApplication.translate("upsample_tool_mw", u"Avoid Overwriting", None))
#if QT_CONFIG(tooltip)
        self.process_sel_pb.setToolTip(QCoreApplication.translate("upsample_tool_mw", u"Process selected item(s) with current settings", None))
#endif // QT_CONFIG(tooltip)
        self.process_sel_pb.setText(QCoreApplication.translate("upsample_tool_mw", u"Process Selected", None))
#if QT_CONFIG(tooltip)
        self.preview_pb.setToolTip(QCoreApplication.translate("upsample_tool_mw", u"Apply current settings to selected item\n"
"and play the result without writing anything", None))
#endif // QT_CONFIG(tooltip)
        self.preview_pb.setText(QCoreApplication.translate("upsample_tool_mw", u"Preview Selected", None))
#if QT_CONFIG(tooltip)
        self.process_pb.setToolTip(QCoreApplication.translate("upsample_tool_mw", u"Process all input files with current settings", None))
#endif // QT_CONFIG(tooltip)
        self.process_pb.setText(QCoreApplication.translate("upsample_tool_mw", u"Batch Process", None))
    # retranslateUi

