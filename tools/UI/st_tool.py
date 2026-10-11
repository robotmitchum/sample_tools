# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'st_tool.ui'
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

class Ui_st_tool_mw(object):
    def setupUi(self, st_tool_mw):
        if not st_tool_mw.objectName():
            st_tool_mw.setObjectName(u"st_tool_mw")
        st_tool_mw.resize(640, 600)
        st_tool_mw.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        self.centralwidget = QWidget(st_tool_mw)
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
        self.files_title_l.setStyleSheet(u"background-color: rgb(127, 63, 127);\n"
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
        self.options_title_l.setStyleSheet(u"background-color: rgb(127, 63, 127);\n"
"color: rgb(255, 255, 255);")

        self.verticalLayout.addWidget(self.options_title_l)

        self.st_mode_lyt = QHBoxLayout()
        self.st_mode_lyt.setObjectName(u"st_mode_lyt")
        self.mode_l = QLabel(self.centralwidget)
        self.mode_l.setObjectName(u"mode_l")

        self.st_mode_lyt.addWidget(self.mode_l)

        self.st_mode_cmb = QComboBox(self.centralwidget)
        self.st_mode_cmb.addItem("")
        self.st_mode_cmb.addItem("")
        self.st_mode_cmb.addItem("")
        self.st_mode_cmb.addItem("")
        self.st_mode_cmb.setObjectName(u"st_mode_cmb")
        sizePolicy.setHeightForWidth(self.st_mode_cmb.sizePolicy().hasHeightForWidth())
        self.st_mode_cmb.setSizePolicy(sizePolicy)
        self.st_mode_cmb.setFrame(False)

        self.st_mode_lyt.addWidget(self.st_mode_cmb)

        self.delay_l = QLabel(self.centralwidget)
        self.delay_l.setObjectName(u"delay_l")

        self.st_mode_lyt.addWidget(self.delay_l)

        self.delay_dsb = QDoubleSpinBox(self.centralwidget)
        self.delay_dsb.setObjectName(u"delay_dsb")
        self.delay_dsb.setMinimumSize(QSize(48, 0))
        self.delay_dsb.setFrame(False)
        self.delay_dsb.setAlignment(Qt.AlignCenter)
        self.delay_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.delay_dsb.setMaximum(30.000000000000000)
        self.delay_dsb.setValue(6.000000000000000)

        self.st_mode_lyt.addWidget(self.delay_dsb)

        self.seed_l = QLabel(self.centralwidget)
        self.seed_l.setObjectName(u"seed_l")
        self.seed_l.setEnabled(False)

        self.st_mode_lyt.addWidget(self.seed_l)

        self.seed_sb = QSpinBox(self.centralwidget)
        self.seed_sb.setObjectName(u"seed_sb")
        self.seed_sb.setEnabled(False)
        self.seed_sb.setMinimumSize(QSize(64, 0))
        self.seed_sb.setFrame(False)
        self.seed_sb.setAlignment(Qt.AlignCenter)
        self.seed_sb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.seed_sb.setMinimum(-1)
        self.seed_sb.setMaximum(99999)
        self.seed_sb.setValue(0)

        self.st_mode_lyt.addWidget(self.seed_sb)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.st_mode_lyt.addItem(self.horizontalSpacer_3)

        self.wet_l = QLabel(self.centralwidget)
        self.wet_l.setObjectName(u"wet_l")
        self.wet_l.setEnabled(False)

        self.st_mode_lyt.addWidget(self.wet_l)

        self.wet_dsb = QDoubleSpinBox(self.centralwidget)
        self.wet_dsb.setObjectName(u"wet_dsb")
        self.wet_dsb.setEnabled(False)
        self.wet_dsb.setMinimumSize(QSize(48, 0))
        self.wet_dsb.setFrame(False)
        self.wet_dsb.setAlignment(Qt.AlignCenter)
        self.wet_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.wet_dsb.setDecimals(3)
        self.wet_dsb.setMaximum(1.000000000000000)
        self.wet_dsb.setSingleStep(0.100000000000000)
        self.wet_dsb.setValue(1.000000000000000)

        self.st_mode_lyt.addWidget(self.wet_dsb)


        self.verticalLayout.addLayout(self.st_mode_lyt)

        self.ir_path_lyt = QHBoxLayout()
        self.ir_path_lyt.setObjectName(u"ir_path_lyt")
        self.mxlen_cb = QCheckBox(self.centralwidget)
        self.mxlen_cb.setObjectName(u"mxlen_cb")
        self.mxlen_cb.setEnabled(False)
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.mxlen_cb.sizePolicy().hasHeightForWidth())
        self.mxlen_cb.setSizePolicy(sizePolicy1)

        self.ir_path_lyt.addWidget(self.mxlen_cb)

        self.ir_path_l = QLabel(self.centralwidget)
        self.ir_path_l.setObjectName(u"ir_path_l")
        self.ir_path_l.setEnabled(False)
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.ir_path_l.sizePolicy().hasHeightForWidth())
        self.ir_path_l.setSizePolicy(sizePolicy2)

        self.ir_path_lyt.addWidget(self.ir_path_l)

        self.set_ir_path_tb = QToolButton(self.centralwidget)
        self.set_ir_path_tb.setObjectName(u"set_ir_path_tb")
        self.set_ir_path_tb.setEnabled(False)
        self.set_ir_path_tb.setFont(font)

        self.ir_path_lyt.addWidget(self.set_ir_path_tb)


        self.verticalLayout.addLayout(self.ir_path_lyt)

        self.filter_side_lyt = QHBoxLayout()
        self.filter_side_lyt.setObjectName(u"filter_side_lyt")
        self.filter_side_cb = QCheckBox(self.centralwidget)
        self.filter_side_cb.setObjectName(u"filter_side_cb")
        self.filter_side_cb.setChecked(True)

        self.filter_side_lyt.addWidget(self.filter_side_cb)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.filter_side_lyt.addItem(self.horizontalSpacer)

        self.filter_side_wid = QWidget(self.centralwidget)
        self.filter_side_wid.setObjectName(u"filter_side_wid")
        self.horizontalLayout_7 = QHBoxLayout(self.filter_side_wid)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.cutoff_l = QLabel(self.filter_side_wid)
        self.cutoff_l.setObjectName(u"cutoff_l")

        self.horizontalLayout_7.addWidget(self.cutoff_l)

        self.cutoff_dsb = QDoubleSpinBox(self.filter_side_wid)
        self.cutoff_dsb.setObjectName(u"cutoff_dsb")
        self.cutoff_dsb.setMinimumSize(QSize(48, 0))
        self.cutoff_dsb.setFrame(False)
        self.cutoff_dsb.setAlignment(Qt.AlignCenter)
        self.cutoff_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.cutoff_dsb.setMaximum(20000.000000000000000)
        self.cutoff_dsb.setValue(500.000000000000000)

        self.horizontalLayout_7.addWidget(self.cutoff_dsb)

        self.band_l = QLabel(self.filter_side_wid)
        self.band_l.setObjectName(u"band_l")

        self.horizontalLayout_7.addWidget(self.band_l)

        self.band_dsb = QDoubleSpinBox(self.filter_side_wid)
        self.band_dsb.setObjectName(u"band_dsb")
        self.band_dsb.setMinimumSize(QSize(48, 0))
        self.band_dsb.setFrame(False)
        self.band_dsb.setAlignment(Qt.AlignCenter)
        self.band_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.band_dsb.setMinimum(-20000.000000000000000)
        self.band_dsb.setMaximum(20000.000000000000000)
        self.band_dsb.setValue(500.000000000000000)

        self.horizontalLayout_7.addWidget(self.band_dsb)


        self.filter_side_lyt.addWidget(self.filter_side_wid)


        self.verticalLayout.addLayout(self.filter_side_lyt)

        self.balance_lyt = QHBoxLayout()
        self.balance_lyt.setObjectName(u"balance_lyt")
        self.balance_wid = QWidget(self.centralwidget)
        self.balance_wid.setObjectName(u"balance_wid")
        self.horizontalLayout = QHBoxLayout(self.balance_wid)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.balance_cb = QCheckBox(self.balance_wid)
        self.balance_cb.setObjectName(u"balance_cb")
        self.balance_cb.setChecked(True)

        self.horizontalLayout.addWidget(self.balance_cb)

        self.align_phase_cb = QCheckBox(self.balance_wid)
        self.align_phase_cb.setObjectName(u"align_phase_cb")

        self.horizontalLayout.addWidget(self.align_phase_cb)


        self.balance_lyt.addWidget(self.balance_wid)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.balance_lyt.addItem(self.horizontalSpacer_5)

        self.width_l = QLabel(self.centralwidget)
        self.width_l.setObjectName(u"width_l")

        self.balance_lyt.addWidget(self.width_l)

        self.width_dsb = QDoubleSpinBox(self.centralwidget)
        self.width_dsb.setObjectName(u"width_dsb")
        self.width_dsb.setMinimumSize(QSize(48, 0))
        self.width_dsb.setFrame(False)
        self.width_dsb.setAlignment(Qt.AlignCenter)
        self.width_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.width_dsb.setDecimals(3)
        self.width_dsb.setMaximum(8.000000000000000)
        self.width_dsb.setSingleStep(0.100000000000000)
        self.width_dsb.setValue(1.000000000000000)

        self.balance_lyt.addWidget(self.width_dsb)


        self.verticalLayout.addLayout(self.balance_lyt)

        self.output_path_title_l = QLabel(self.centralwidget)
        self.output_path_title_l.setObjectName(u"output_path_title_l")
        sizePolicy.setHeightForWidth(self.output_path_title_l.sizePolicy().hasHeightForWidth())
        self.output_path_title_l.setSizePolicy(sizePolicy)
        self.output_path_title_l.setMinimumSize(QSize(64, 0))
        self.output_path_title_l.setStyleSheet(u"background-color: rgb(127, 63, 127);\n"
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
        self.settings_title_l.setStyleSheet(u"background-color: rgb(127, 63, 127);\n"
"color: rgb(255, 255, 255);")

        self.verticalLayout.addWidget(self.settings_title_l)

        self.file_settings_lyt = QHBoxLayout()
        self.file_settings_lyt.setObjectName(u"file_settings_lyt")
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy3)

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
        sizePolicy1.setHeightForWidth(self.bd_l.sizePolicy().hasHeightForWidth())
        self.bd_l.setSizePolicy(sizePolicy1)

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
        sizePolicy1.setHeightForWidth(self.add_suffix_cb.sizePolicy().hasHeightForWidth())
        self.add_suffix_cb.setSizePolicy(sizePolicy1)
        self.add_suffix_cb.setChecked(True)

        self.file_settings_lyt.addWidget(self.add_suffix_cb)

        self.suffix_le = QLineEdit(self.centralwidget)
        self.suffix_le.setObjectName(u"suffix_le")
        sizePolicy1.setHeightForWidth(self.suffix_le.sizePolicy().hasHeightForWidth())
        self.suffix_le.setSizePolicy(sizePolicy1)
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
        sizePolicy1.setHeightForWidth(self.process_sel_pb.sizePolicy().hasHeightForWidth())
        self.process_sel_pb.setSizePolicy(sizePolicy1)
        self.process_sel_pb.setMinimumSize(QSize(160, 0))
        self.process_sel_pb.setFont(font)

        self.buttons_lyt.addWidget(self.process_sel_pb, 0, 1, 1, 1)

        self.preview_pb = QPushButton(self.centralwidget)
        self.preview_pb.setObjectName(u"preview_pb")
        sizePolicy1.setHeightForWidth(self.preview_pb.sizePolicy().hasHeightForWidth())
        self.preview_pb.setSizePolicy(sizePolicy1)
        self.preview_pb.setMinimumSize(QSize(160, 0))
        self.preview_pb.setFont(font)

        self.buttons_lyt.addWidget(self.preview_pb, 0, 0, 1, 1)

        self.process_pb = QPushButton(self.centralwidget)
        self.process_pb.setObjectName(u"process_pb")
        sizePolicy1.setHeightForWidth(self.process_pb.sizePolicy().hasHeightForWidth())
        self.process_pb.setSizePolicy(sizePolicy1)
        self.process_pb.setMinimumSize(QSize(160, 0))
        self.process_pb.setFont(font)
        self.process_pb.setStyleSheet(u"QPushButton{background-color: rgb(127, 63, 127);\n"
"color: rgb(255, 255, 255);}")

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

        st_tool_mw.setCentralWidget(self.centralwidget)

        self.retranslateUi(st_tool_mw)

        self.st_mode_cmb.setCurrentIndex(3)
        self.format_cmb.setCurrentIndex(0)
        self.bitdepth_cmb.setCurrentIndex(0)
        self.process_pb.setDefault(False)


        QMetaObject.connectSlotsByName(st_tool_mw)
    # setupUi

    def retranslateUi(self, st_tool_mw):
        st_tool_mw.setWindowTitle(QCoreApplication.translate("st_tool_mw", u"Stereo Tool", None))
        self.files_title_l.setText(QCoreApplication.translate("st_tool_mw", u"Input Files", None))
#if QT_CONFIG(tooltip)
        self.files_lw.setToolTip(QCoreApplication.translate("st_tool_mw", u"Drag and drop files or directories\n"
"Right click for context actions\n"
"Double-click to play", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.set_files_tb.setToolTip(QCoreApplication.translate("st_tool_mw", u"Set files to process", None))
#endif // QT_CONFIG(tooltip)
        self.set_files_tb.setText(QCoreApplication.translate("st_tool_mw", u"...", None))
        self.options_title_l.setText(QCoreApplication.translate("st_tool_mw", u"Options", None))
        self.mode_l.setText(QCoreApplication.translate("st_tool_mw", u"Mode", None))
        self.st_mode_cmb.setItemText(0, QCoreApplication.translate("st_tool_mw", u"haas", None))
        self.st_mode_cmb.setItemText(1, QCoreApplication.translate("st_tool_mw", u"velvet", None))
        self.st_mode_cmb.setItemText(2, QCoreApplication.translate("st_tool_mw", u"convolve", None))
        self.st_mode_cmb.setItemText(3, QCoreApplication.translate("st_tool_mw", u"conv_side", None))

#if QT_CONFIG(tooltip)
        self.st_mode_cmb.setToolTip(QCoreApplication.translate("st_tool_mw", u"Mode used to for pseudo-stereo effect\n"
"\n"
"'haas'	apply delay to side channel\n"
"'velvet'	convolve side channel with a velvet noise with increasing delay between impulses (typically weaker than haas)\n"
"'convolve'	apply a stereo imaging impulse response\n"
"'conv_side'	convolve the side channel using the side channel of an impulse response (default)\n"
"\n"
"NOTE: 'haas' and 'velvet' modes are inspired by Ozone Stereo Imager module and behave roughly the same", None))
#endif // QT_CONFIG(tooltip)
        self.delay_l.setText(QCoreApplication.translate("st_tool_mw", u"Delay", None))
#if QT_CONFIG(tooltip)
        self.delay_dsb.setToolTip(QCoreApplication.translate("st_tool_mw", u"Pseudo-Stereo delay in in ms", None))
#endif // QT_CONFIG(tooltip)
        self.seed_l.setText(QCoreApplication.translate("st_tool_mw", u"Seed", None))
#if QT_CONFIG(tooltip)
        self.seed_sb.setToolTip(QCoreApplication.translate("st_tool_mw", u"Velvet noise seed (-1 to randomize for each file)", None))
#endif // QT_CONFIG(tooltip)
        self.wet_l.setText(QCoreApplication.translate("st_tool_mw", u"Wet", None))
#if QT_CONFIG(tooltip)
        self.wet_dsb.setToolTip(QCoreApplication.translate("st_tool_mw", u"Convolution effect wet level\n"
"\n"
"TIP : Reduce wetness then increase (compensate) stereo width\n"
"to benefit from enhanced stereo but with reduced hall/room effect", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.mxlen_cb.setToolTip(QCoreApplication.translate("st_tool_mw", u"Extend length so convolution tail is not cut", None))
#endif // QT_CONFIG(tooltip)
        self.mxlen_cb.setText(QCoreApplication.translate("st_tool_mw", u"Max Length", None))
#if QT_CONFIG(tooltip)
        self.ir_path_l.setToolTip(QCoreApplication.translate("st_tool_mw", u"Path to IR used for 'convolve' and 'conv_side' modes\n"
"\n"
"Right click to choose between provided dolby headphones IR\n"
"DH1 is designed to sound as a home theater setup (average effect)\n"
"DH2 is designed to sound like a medium-sized room (more subtle effect)\n"
"DH3 sounds like a large theater (strongest effect)", None))
#endif // QT_CONFIG(tooltip)
        self.ir_path_l.setText("")
#if QT_CONFIG(tooltip)
        self.set_ir_path_tb.setToolTip(QCoreApplication.translate("st_tool_mw", u"Set IR to use for convolution", None))
#endif // QT_CONFIG(tooltip)
        self.set_ir_path_tb.setText(QCoreApplication.translate("st_tool_mw", u"...", None))
#if QT_CONFIG(tooltip)
        self.filter_side_cb.setToolTip(QCoreApplication.translate("st_tool_mw", u"Filter side channel\n"
"High pass to remove stereo effect in bass range (Band > 0)\n"
"Low pass to tighten stereo effect (Band < 0)", None))
#endif // QT_CONFIG(tooltip)
        self.filter_side_cb.setText(QCoreApplication.translate("st_tool_mw", u"Filter Side Channel", None))
        self.cutoff_l.setText(QCoreApplication.translate("st_tool_mw", u"Cutoff", None))
#if QT_CONFIG(tooltip)
        self.cutoff_dsb.setToolTip(QCoreApplication.translate("st_tool_mw", u"Cutoff frequency", None))
#endif // QT_CONFIG(tooltip)
        self.band_l.setText(QCoreApplication.translate("st_tool_mw", u"Band", None))
#if QT_CONFIG(tooltip)
        self.band_dsb.setToolTip(QCoreApplication.translate("st_tool_mw", u"Transition band width\n"
"Positive : High pass filter\n"
"Negative : Low pass filter", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.balance_cb.setToolTip(QCoreApplication.translate("st_tool_mw", u"Rebalance left and right channel so audio feels centered\n"
"Recommanded, especially with 'velvet' mode", None))
#endif // QT_CONFIG(tooltip)
        self.balance_cb.setText(QCoreApplication.translate("st_tool_mw", u"Auto-Balance L/R", None))
#if QT_CONFIG(tooltip)
        self.align_phase_cb.setToolTip(QCoreApplication.translate("st_tool_mw", u"Align phase of L/R channels\n"
"May improve centering\n"
"non-reversible", None))
#endif // QT_CONFIG(tooltip)
        self.align_phase_cb.setText(QCoreApplication.translate("st_tool_mw", u"Align Phase", None))
        self.width_l.setText(QCoreApplication.translate("st_tool_mw", u"Stereo Width", None))
#if QT_CONFIG(tooltip)
        self.width_dsb.setToolTip(QCoreApplication.translate("st_tool_mw", u"Stereo Width", None))
#endif // QT_CONFIG(tooltip)
        self.output_path_title_l.setText(QCoreApplication.translate("st_tool_mw", u"Output Path", None))
        self.output_path_l.setText("")
#if QT_CONFIG(tooltip)
        self.set_output_path_tb.setToolTip(QCoreApplication.translate("st_tool_mw", u"Set output path\n"
"Process files in their respective directory if empty", None))
#endif // QT_CONFIG(tooltip)
        self.set_output_path_tb.setText(QCoreApplication.translate("st_tool_mw", u"...", None))
        self.settings_title_l.setText(QCoreApplication.translate("st_tool_mw", u"File Settings", None))
        self.label_2.setText(QCoreApplication.translate("st_tool_mw", u"Format", None))
        self.format_cmb.setItemText(0, QCoreApplication.translate("st_tool_mw", u"same", None))
        self.format_cmb.setItemText(1, QCoreApplication.translate("st_tool_mw", u"wav", None))
        self.format_cmb.setItemText(2, QCoreApplication.translate("st_tool_mw", u"flac", None))

#if QT_CONFIG(tooltip)
        self.format_cmb.setToolTip(QCoreApplication.translate("st_tool_mw", u"flac only allows integer format up to 24 bits\n"
"Use wav for 32 bits float", None))
#endif // QT_CONFIG(tooltip)
        self.bd_l.setText(QCoreApplication.translate("st_tool_mw", u"Bit Depth", None))
        self.bitdepth_cmb.setItemText(0, QCoreApplication.translate("st_tool_mw", u"same", None))
        self.bitdepth_cmb.setItemText(1, QCoreApplication.translate("st_tool_mw", u"16", None))
        self.bitdepth_cmb.setItemText(2, QCoreApplication.translate("st_tool_mw", u"24", None))
        self.bitdepth_cmb.setItemText(3, QCoreApplication.translate("st_tool_mw", u"32", None))

        self.add_suffix_cb.setText(QCoreApplication.translate("st_tool_mw", u"Add suffix", None))
#if QT_CONFIG(tooltip)
        self.suffix_le.setToolTip(QCoreApplication.translate("st_tool_mw", u"Suffix added to the base name", None))
#endif // QT_CONFIG(tooltip)
        self.suffix_le.setText(QCoreApplication.translate("st_tool_mw", u"_st", None))
#if QT_CONFIG(tooltip)
        self.no_overwriting_cb.setToolTip(QCoreApplication.translate("st_tool_mw", u"Avoid overwriting original files by moving them to a backup directory", None))
#endif // QT_CONFIG(tooltip)
        self.no_overwriting_cb.setText(QCoreApplication.translate("st_tool_mw", u"Avoid Overwriting", None))
#if QT_CONFIG(tooltip)
        self.process_sel_pb.setToolTip(QCoreApplication.translate("st_tool_mw", u"Process selected item(s) with current settings", None))
#endif // QT_CONFIG(tooltip)
        self.process_sel_pb.setText(QCoreApplication.translate("st_tool_mw", u"Process Selected", None))
#if QT_CONFIG(tooltip)
        self.preview_pb.setToolTip(QCoreApplication.translate("st_tool_mw", u"Apply current settings to selected item\n"
"and play the result without writing anything", None))
#endif // QT_CONFIG(tooltip)
        self.preview_pb.setText(QCoreApplication.translate("st_tool_mw", u"Preview Selected", None))
#if QT_CONFIG(tooltip)
        self.process_pb.setToolTip(QCoreApplication.translate("st_tool_mw", u"Process all input files with current settings", None))
#endif // QT_CONFIG(tooltip)
        self.process_pb.setText(QCoreApplication.translate("st_tool_mw", u"Batch Process", None))
    # retranslateUi

