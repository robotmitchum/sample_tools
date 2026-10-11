# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'loop_tool.ui'
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

class Ui_loop_tool_mw(object):
    def setupUi(self, loop_tool_mw):
        if not loop_tool_mw.objectName():
            loop_tool_mw.setObjectName(u"loop_tool_mw")
        loop_tool_mw.resize(880, 800)
        loop_tool_mw.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        self.centralwidget = QWidget(loop_tool_mw)
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
        self.files_title_l.setStyleSheet(u"background-color: rgb(31, 159, 127);\n"
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
        self.options_title_l.setStyleSheet(u"background-color: rgb(31, 159, 127);\n"
"color: rgb(255, 255, 255);")

        self.verticalLayout.addWidget(self.options_title_l)

        self.env_shaping_lyt = QHBoxLayout()
        self.env_shaping_lyt.setObjectName(u"env_shaping_lyt")
        self.env_shaping_cb = QCheckBox(self.centralwidget)
        self.env_shaping_cb.setObjectName(u"env_shaping_cb")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.env_shaping_cb.sizePolicy().hasHeightForWidth())
        self.env_shaping_cb.setSizePolicy(sizePolicy1)
        self.env_shaping_cb.setMinimumSize(QSize(128, 0))

        self.env_shaping_lyt.addWidget(self.env_shaping_cb)

        self.env_shaping_wid = QWidget(self.centralwidget)
        self.env_shaping_wid.setObjectName(u"env_shaping_wid")
        self.env_shaping_wid.setEnabled(False)
        self.horizontalLayout = QHBoxLayout(self.env_shaping_wid)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.env_mask_l = QLabel(self.env_shaping_wid)
        self.env_mask_l.setObjectName(u"env_mask_l")
        self.env_mask_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout.addWidget(self.env_mask_l)

        self.env_mode_cmb = QComboBox(self.env_shaping_wid)
        self.env_mode_cmb.addItem("")
        self.env_mode_cmb.addItem("")
        self.env_mode_cmb.setObjectName(u"env_mode_cmb")
        sizePolicy.setHeightForWidth(self.env_mode_cmb.sizePolicy().hasHeightForWidth())
        self.env_mode_cmb.setSizePolicy(sizePolicy)
        self.env_mode_cmb.setFrame(False)

        self.horizontalLayout.addWidget(self.env_mode_cmb)

        self.env_th_l = QLabel(self.env_shaping_wid)
        self.env_th_l.setObjectName(u"env_th_l")
        self.env_th_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout.addWidget(self.env_th_l)

        self.env_th_dsb = QDoubleSpinBox(self.env_shaping_wid)
        self.env_th_dsb.setObjectName(u"env_th_dsb")
        self.env_th_dsb.setMinimumSize(QSize(48, 0))
        self.env_th_dsb.setFrame(False)
        self.env_th_dsb.setAlignment(Qt.AlignCenter)
        self.env_th_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.env_th_dsb.setMinimum(-60.000000000000000)
        self.env_th_dsb.setMaximum(0.000000000000000)
        self.env_th_dsb.setValue(-12.000000000000000)

        self.horizontalLayout.addWidget(self.env_th_dsb)

        self.env_min_l = QLabel(self.env_shaping_wid)
        self.env_min_l.setObjectName(u"env_min_l")
        self.env_min_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout.addWidget(self.env_min_l)

        self.env_min_dsb = QDoubleSpinBox(self.env_shaping_wid)
        self.env_min_dsb.setObjectName(u"env_min_dsb")
        self.env_min_dsb.setMinimumSize(QSize(48, 0))
        self.env_min_dsb.setFrame(False)
        self.env_min_dsb.setAlignment(Qt.AlignCenter)
        self.env_min_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.env_min_dsb.setMinimum(-60.000000000000000)
        self.env_min_dsb.setMaximum(0.000000000000000)
        self.env_min_dsb.setValue(-30.000000000000000)

        self.horizontalLayout.addWidget(self.env_min_dsb)

        self.env_window_wid = QWidget(self.env_shaping_wid)
        self.env_window_wid.setObjectName(u"env_window_wid")
        self.env_window_wid.setEnabled(False)
        self.horizontalLayout_4 = QHBoxLayout(self.env_window_wid)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.env_window_l = QLabel(self.env_window_wid)
        self.env_window_l.setObjectName(u"env_window_l")
        self.env_window_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_4.addWidget(self.env_window_l)

        self.env_window_dsb = QDoubleSpinBox(self.env_window_wid)
        self.env_window_dsb.setObjectName(u"env_window_dsb")
        self.env_window_dsb.setMinimumSize(QSize(48, 0))
        self.env_window_dsb.setFrame(False)
        self.env_window_dsb.setAlignment(Qt.AlignCenter)
        self.env_window_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.env_window_dsb.setDecimals(1)
        self.env_window_dsb.setMinimum(1.000000000000000)
        self.env_window_dsb.setMaximum(500.000000000000000)
        self.env_window_dsb.setValue(50.000000000000000)

        self.horizontalLayout_4.addWidget(self.env_window_dsb)


        self.horizontalLayout.addWidget(self.env_window_wid)


        self.env_shaping_lyt.addWidget(self.env_shaping_wid)


        self.verticalLayout.addLayout(self.env_shaping_lyt)

        self.line_2 = QFrame(self.centralwidget)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.Shape.HLine)
        self.line_2.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout.addWidget(self.line_2)

        self.detect_loop_lyt = QHBoxLayout()
        self.detect_loop_lyt.setObjectName(u"detect_loop_lyt")
        self.detect_loop_cb = QCheckBox(self.centralwidget)
        self.detect_loop_cb.setObjectName(u"detect_loop_cb")
        sizePolicy1.setHeightForWidth(self.detect_loop_cb.sizePolicy().hasHeightForWidth())
        self.detect_loop_cb.setSizePolicy(sizePolicy1)
        self.detect_loop_cb.setMinimumSize(QSize(128, 0))
        self.detect_loop_cb.setChecked(True)

        self.detect_loop_lyt.addWidget(self.detect_loop_cb)

        self.detect_loop_wid = QWidget(self.centralwidget)
        self.detect_loop_wid.setObjectName(u"detect_loop_wid")
        self.detect_loop_wid.setEnabled(True)
        self.horizontalLayout_2 = QHBoxLayout(self.detect_loop_wid)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.loop_min_length_l = QLabel(self.detect_loop_wid)
        self.loop_min_length_l.setObjectName(u"loop_min_length_l")
        self.loop_min_length_l.setMinimumSize(QSize(64, 0))
        self.loop_min_length_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_2.addWidget(self.loop_min_length_l)

        self.loop_min_length_dsb = QDoubleSpinBox(self.detect_loop_wid)
        self.loop_min_length_dsb.setObjectName(u"loop_min_length_dsb")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.loop_min_length_dsb.sizePolicy().hasHeightForWidth())
        self.loop_min_length_dsb.setSizePolicy(sizePolicy2)
        self.loop_min_length_dsb.setMinimumSize(QSize(64, 0))
        self.loop_min_length_dsb.setFrame(False)
        self.loop_min_length_dsb.setAlignment(Qt.AlignCenter)
        self.loop_min_length_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.loop_min_length_dsb.setDecimals(1)
        self.loop_min_length_dsb.setMinimum(0.000000000000000)
        self.loop_min_length_dsb.setMaximum(60000.000000000000000)
        self.loop_min_length_dsb.setValue(100.000000000000000)

        self.horizontalLayout_2.addWidget(self.loop_min_length_dsb)

        self.cues_l = QLabel(self.detect_loop_wid)
        self.cues_l.setObjectName(u"cues_l")
        self.cues_l.setMinimumSize(QSize(64, 0))
        self.cues_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_2.addWidget(self.cues_l)

        self.cues_sb = QSpinBox(self.detect_loop_wid)
        self.cues_sb.setObjectName(u"cues_sb")
        sizePolicy2.setHeightForWidth(self.cues_sb.sizePolicy().hasHeightForWidth())
        self.cues_sb.setSizePolicy(sizePolicy2)
        self.cues_sb.setMinimumSize(QSize(64, 0))
        self.cues_sb.setFrame(False)
        self.cues_sb.setAlignment(Qt.AlignCenter)
        self.cues_sb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.cues_sb.setMinimum(90000)
        self.cues_sb.setMaximum(4000000)
        self.cues_sb.setSingleStep(1)
        self.cues_sb.setValue(640000)

        self.horizontalLayout_2.addWidget(self.cues_sb)

        self.loop_window_l = QLabel(self.detect_loop_wid)
        self.loop_window_l.setObjectName(u"loop_window_l")
        self.loop_window_l.setMinimumSize(QSize(64, 0))
        self.loop_window_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_2.addWidget(self.loop_window_l)

        self.loop_window_dsb = QDoubleSpinBox(self.detect_loop_wid)
        self.loop_window_dsb.setObjectName(u"loop_window_dsb")
        sizePolicy2.setHeightForWidth(self.loop_window_dsb.sizePolicy().hasHeightForWidth())
        self.loop_window_dsb.setSizePolicy(sizePolicy2)
        self.loop_window_dsb.setMinimumSize(QSize(64, 0))
        self.loop_window_dsb.setFrame(False)
        self.loop_window_dsb.setAlignment(Qt.AlignCenter)
        self.loop_window_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.loop_window_dsb.setDecimals(1)
        self.loop_window_dsb.setMinimum(0.000000000000000)
        self.loop_window_dsb.setMaximum(2000.000000000000000)
        self.loop_window_dsb.setValue(10.000000000000000)

        self.horizontalLayout_2.addWidget(self.loop_window_dsb)

        self.window_offset_l = QLabel(self.detect_loop_wid)
        self.window_offset_l.setObjectName(u"window_offset_l")
        self.window_offset_l.setMinimumSize(QSize(64, 0))
        self.window_offset_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_2.addWidget(self.window_offset_l)

        self.window_offset_dsb = QDoubleSpinBox(self.detect_loop_wid)
        self.window_offset_dsb.setObjectName(u"window_offset_dsb")
        self.window_offset_dsb.setMinimumSize(QSize(64, 0))
        self.window_offset_dsb.setFrame(False)
        self.window_offset_dsb.setAlignment(Qt.AlignCenter)
        self.window_offset_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.window_offset_dsb.setDecimals(1)
        self.window_offset_dsb.setMinimum(0.000000000000000)
        self.window_offset_dsb.setMaximum(1.000000000000000)
        self.window_offset_dsb.setSingleStep(0.010000000000000)
        self.window_offset_dsb.setValue(0.500000000000000)

        self.horizontalLayout_2.addWidget(self.window_offset_dsb)


        self.detect_loop_lyt.addWidget(self.detect_loop_wid)


        self.verticalLayout.addLayout(self.detect_loop_lyt)

        self.detect_extra_wid = QWidget(self.centralwidget)
        self.detect_extra_wid.setObjectName(u"detect_extra_wid")
        self.horizontalLayout_6 = QHBoxLayout(self.detect_extra_wid)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.hash_search_cb = QCheckBox(self.detect_extra_wid)
        self.hash_search_cb.setObjectName(u"hash_search_cb")
        sizePolicy1.setHeightForWidth(self.hash_search_cb.sizePolicy().hasHeightForWidth())
        self.hash_search_cb.setSizePolicy(sizePolicy1)
        self.hash_search_cb.setMinimumSize(QSize(128, 0))
        self.hash_search_cb.setLayoutDirection(Qt.LeftToRight)

        self.horizontalLayout_6.addWidget(self.hash_search_cb)

        self.horizontalSpacer_10 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_6.addItem(self.horizontalSpacer_10)

        self.range_limits_cb = QCheckBox(self.detect_extra_wid)
        self.range_limits_cb.setObjectName(u"range_limits_cb")
        self.range_limits_cb.setChecked(True)

        self.horizontalLayout_6.addWidget(self.range_limits_cb)

        self.range_limits_wid = QWidget(self.detect_extra_wid)
        self.range_limits_wid.setObjectName(u"range_limits_wid")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.range_limits_wid.sizePolicy().hasHeightForWidth())
        self.range_limits_wid.setSizePolicy(sizePolicy3)
        self.horizontalLayout_5 = QHBoxLayout(self.range_limits_wid)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.start_range_l = QLabel(self.range_limits_wid)
        self.start_range_l.setObjectName(u"start_range_l")
        self.start_range_l.setMinimumSize(QSize(64, 0))
        self.start_range_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_5.addWidget(self.start_range_l)

        self.start_range_dsb = QDoubleSpinBox(self.range_limits_wid)
        self.start_range_dsb.setObjectName(u"start_range_dsb")
        sizePolicy2.setHeightForWidth(self.start_range_dsb.sizePolicy().hasHeightForWidth())
        self.start_range_dsb.setSizePolicy(sizePolicy2)
        self.start_range_dsb.setMinimumSize(QSize(64, 0))
        self.start_range_dsb.setFrame(False)
        self.start_range_dsb.setAlignment(Qt.AlignCenter)
        self.start_range_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.start_range_dsb.setDecimals(3)
        self.start_range_dsb.setMinimum(0.000000000000000)
        self.start_range_dsb.setMaximum(1.000000000000000)
        self.start_range_dsb.setSingleStep(0.010000000000000)
        self.start_range_dsb.setValue(1.000000000000000)

        self.horizontalLayout_5.addWidget(self.start_range_dsb)

        self.end_range_l = QLabel(self.range_limits_wid)
        self.end_range_l.setObjectName(u"end_range_l")
        self.end_range_l.setMinimumSize(QSize(64, 0))
        self.end_range_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_5.addWidget(self.end_range_l)

        self.end_range_dsb = QDoubleSpinBox(self.range_limits_wid)
        self.end_range_dsb.setObjectName(u"end_range_dsb")
        sizePolicy2.setHeightForWidth(self.end_range_dsb.sizePolicy().hasHeightForWidth())
        self.end_range_dsb.setSizePolicy(sizePolicy2)
        self.end_range_dsb.setMinimumSize(QSize(64, 0))
        self.end_range_dsb.setFrame(False)
        self.end_range_dsb.setAlignment(Qt.AlignCenter)
        self.end_range_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.end_range_dsb.setDecimals(3)
        self.end_range_dsb.setMinimum(0.000000000000000)
        self.end_range_dsb.setMaximum(1.000000000000000)
        self.end_range_dsb.setSingleStep(0.010000000000000)
        self.end_range_dsb.setValue(0.000000000000000)

        self.horizontalLayout_5.addWidget(self.end_range_dsb)


        self.horizontalLayout_6.addWidget(self.range_limits_wid)


        self.verticalLayout.addWidget(self.detect_extra_wid)

        self.line_3 = QFrame(self.centralwidget)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setFrameShape(QFrame.Shape.HLine)
        self.line_3.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout.addWidget(self.line_3)

        self.crossfade_lyt = QHBoxLayout()
        self.crossfade_lyt.setObjectName(u"crossfade_lyt")
        self.crossfade_cb = QCheckBox(self.centralwidget)
        self.crossfade_cb.setObjectName(u"crossfade_cb")
        sizePolicy1.setHeightForWidth(self.crossfade_cb.sizePolicy().hasHeightForWidth())
        self.crossfade_cb.setSizePolicy(sizePolicy1)
        self.crossfade_cb.setMinimumSize(QSize(128, 0))

        self.crossfade_lyt.addWidget(self.crossfade_cb)

        self.crossfade_wid = QWidget(self.centralwidget)
        self.crossfade_wid.setObjectName(u"crossfade_wid")
        self.crossfade_wid.setEnabled(False)
        self.horizontalLayout_3 = QHBoxLayout(self.crossfade_wid)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.fd_preview_only_cb = QCheckBox(self.crossfade_wid)
        self.fd_preview_only_cb.setObjectName(u"fd_preview_only_cb")
        sizePolicy1.setHeightForWidth(self.fd_preview_only_cb.sizePolicy().hasHeightForWidth())
        self.fd_preview_only_cb.setSizePolicy(sizePolicy1)
        self.fd_preview_only_cb.setChecked(True)

        self.horizontalLayout_3.addWidget(self.fd_preview_only_cb)

        self.fade_mode_l = QLabel(self.crossfade_wid)
        self.fade_mode_l.setObjectName(u"fade_mode_l")
        self.fade_mode_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_3.addWidget(self.fade_mode_l)

        self.fade_mode_cmb = QComboBox(self.crossfade_wid)
        self.fade_mode_cmb.addItem("")
        self.fade_mode_cmb.addItem("")
        self.fade_mode_cmb.addItem("")
        self.fade_mode_cmb.setObjectName(u"fade_mode_cmb")
        sizePolicy.setHeightForWidth(self.fade_mode_cmb.sizePolicy().hasHeightForWidth())
        self.fade_mode_cmb.setSizePolicy(sizePolicy)
        self.fade_mode_cmb.setFrame(False)

        self.horizontalLayout_3.addWidget(self.fade_mode_cmb)

        self.fade_in_l = QLabel(self.crossfade_wid)
        self.fade_in_l.setObjectName(u"fade_in_l")
        self.fade_in_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_3.addWidget(self.fade_in_l)

        self.fade_in_dsb = QDoubleSpinBox(self.crossfade_wid)
        self.fade_in_dsb.setObjectName(u"fade_in_dsb")
        self.fade_in_dsb.setMinimumSize(QSize(48, 0))
        self.fade_in_dsb.setFrame(False)
        self.fade_in_dsb.setAlignment(Qt.AlignCenter)
        self.fade_in_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.fade_in_dsb.setMinimum(0.000000000000000)
        self.fade_in_dsb.setMaximum(4.000000000000000)
        self.fade_in_dsb.setSingleStep(0.010000000000000)
        self.fade_in_dsb.setValue(0.500000000000000)

        self.horizontalLayout_3.addWidget(self.fade_in_dsb)

        self.fade_out_l = QLabel(self.crossfade_wid)
        self.fade_out_l.setObjectName(u"fade_out_l")
        self.fade_out_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_3.addWidget(self.fade_out_l)

        self.fade_out_dsb = QDoubleSpinBox(self.crossfade_wid)
        self.fade_out_dsb.setObjectName(u"fade_out_dsb")
        self.fade_out_dsb.setMinimumSize(QSize(48, 0))
        self.fade_out_dsb.setFrame(False)
        self.fade_out_dsb.setAlignment(Qt.AlignCenter)
        self.fade_out_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.fade_out_dsb.setMinimum(0.000000000000000)
        self.fade_out_dsb.setMaximum(4.000000000000000)
        self.fade_out_dsb.setSingleStep(0.010000000000000)
        self.fade_out_dsb.setValue(0.500000000000000)

        self.horizontalLayout_3.addWidget(self.fade_out_dsb)


        self.crossfade_lyt.addWidget(self.crossfade_wid)


        self.verticalLayout.addLayout(self.crossfade_lyt)

        self.line_4 = QFrame(self.centralwidget)
        self.line_4.setObjectName(u"line_4")
        self.line_4.setFrameShape(QFrame.Shape.HLine)
        self.line_4.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout.addWidget(self.line_4)

        self.resynth_lyt = QHBoxLayout()
        self.resynth_lyt.setSpacing(0)
        self.resynth_lyt.setObjectName(u"resynth_lyt")
        self.resynth_cb = QCheckBox(self.centralwidget)
        self.resynth_cb.setObjectName(u"resynth_cb")
        sizePolicy1.setHeightForWidth(self.resynth_cb.sizePolicy().hasHeightForWidth())
        self.resynth_cb.setSizePolicy(sizePolicy1)
        self.resynth_cb.setMinimumSize(QSize(128, 0))

        self.resynth_lyt.addWidget(self.resynth_cb)

        self.resynth_wid = QWidget(self.centralwidget)
        self.resynth_wid.setObjectName(u"resynth_wid")
        self.resynth_wid.setEnabled(False)
        self.horizontalLayout_7 = QHBoxLayout(self.resynth_wid)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.fft_range_l = QLabel(self.resynth_wid)
        self.fft_range_l.setObjectName(u"fft_range_l")
        self.fft_range_l.setMinimumSize(QSize(80, 0))
        self.fft_range_l.setMaximumSize(QSize(80, 16777215))
        self.fft_range_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_7.addWidget(self.fft_range_l)

        self.fft_range_cmb = QComboBox(self.resynth_wid)
        self.fft_range_cmb.addItem("")
        self.fft_range_cmb.addItem("")
        self.fft_range_cmb.setObjectName(u"fft_range_cmb")
        self.fft_range_cmb.setMinimumSize(QSize(80, 0))
        self.fft_range_cmb.setMaximumSize(QSize(80, 16777215))
        self.fft_range_cmb.setFrame(False)

        self.horizontalLayout_7.addWidget(self.fft_range_cmb)

        self.fft_start_l = QLabel(self.resynth_wid)
        self.fft_start_l.setObjectName(u"fft_start_l")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.fft_start_l.sizePolicy().hasHeightForWidth())
        self.fft_start_l.setSizePolicy(sizePolicy4)
        self.fft_start_l.setMinimumSize(QSize(96, 0))
        self.fft_start_l.setMaximumSize(QSize(96, 16777215))
        self.fft_start_l.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_7.addWidget(self.fft_start_l)

        self.fft_start_dsb = QDoubleSpinBox(self.resynth_wid)
        self.fft_start_dsb.setObjectName(u"fft_start_dsb")
        self.fft_start_dsb.setMinimumSize(QSize(64, 0))
        self.fft_start_dsb.setFrame(False)
        self.fft_start_dsb.setAlignment(Qt.AlignCenter)
        self.fft_start_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.fft_start_dsb.setDecimals(3)
        self.fft_start_dsb.setMinimum(0.000000000000000)
        self.fft_start_dsb.setMaximum(1.000000000000000)
        self.fft_start_dsb.setSingleStep(0.100000000000000)
        self.fft_start_dsb.setValue(0.250000000000000)

        self.horizontalLayout_7.addWidget(self.fft_start_dsb)

        self.fft_end_l = QLabel(self.resynth_wid)
        self.fft_end_l.setObjectName(u"fft_end_l")
        self.fft_end_l.setMinimumSize(QSize(80, 0))
        self.fft_end_l.setMaximumSize(QSize(80, 16777215))
        self.fft_end_l.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_7.addWidget(self.fft_end_l)

        self.fft_end_dsb = QDoubleSpinBox(self.resynth_wid)
        self.fft_end_dsb.setObjectName(u"fft_end_dsb")
        self.fft_end_dsb.setMinimumSize(QSize(64, 0))
        self.fft_end_dsb.setFrame(False)
        self.fft_end_dsb.setAlignment(Qt.AlignCenter)
        self.fft_end_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.fft_end_dsb.setDecimals(3)
        self.fft_end_dsb.setMinimum(0.000000000000000)
        self.fft_end_dsb.setMaximum(1.000000000000000)
        self.fft_end_dsb.setSingleStep(0.100000000000000)
        self.fft_end_dsb.setValue(1.000000000000000)

        self.horizontalLayout_7.addWidget(self.fft_end_dsb)

        self.duration_l = QLabel(self.resynth_wid)
        self.duration_l.setObjectName(u"duration_l")
        self.duration_l.setMinimumSize(QSize(96, 0))
        self.duration_l.setMaximumSize(QSize(96, 16777215))
        self.duration_l.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_7.addWidget(self.duration_l)

        self.resynth_duration_dsb = QDoubleSpinBox(self.resynth_wid)
        self.resynth_duration_dsb.setObjectName(u"resynth_duration_dsb")
        self.resynth_duration_dsb.setMinimumSize(QSize(64, 0))
        self.resynth_duration_dsb.setFrame(False)
        self.resynth_duration_dsb.setAlignment(Qt.AlignCenter)
        self.resynth_duration_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.resynth_duration_dsb.setDecimals(3)
        self.resynth_duration_dsb.setMinimum(0.000000000000000)
        self.resynth_duration_dsb.setMaximum(10.000000000000000)
        self.resynth_duration_dsb.setSingleStep(0.100000000000000)
        self.resynth_duration_dsb.setValue(2.000000000000000)

        self.horizontalLayout_7.addWidget(self.resynth_duration_dsb)


        self.resynth_lyt.addWidget(self.resynth_wid)


        self.verticalLayout.addLayout(self.resynth_lyt)

        self.resynth_tone_wid = QWidget(self.centralwidget)
        self.resynth_tone_wid.setObjectName(u"resynth_tone_wid")
        self.resynth_tone_wid.setEnabled(False)
        self.horizontalLayout_11 = QHBoxLayout(self.resynth_tone_wid)
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.horizontalLayout_11.setContentsMargins(0, 0, 0, 0)
        self.horizontalSpacer_6 = QSpacerItem(128, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_11.addItem(self.horizontalSpacer_6)

        self.resynth_freq_mode_l = QLabel(self.resynth_tone_wid)
        self.resynth_freq_mode_l.setObjectName(u"resynth_freq_mode_l")
        self.resynth_freq_mode_l.setMinimumSize(QSize(80, 0))
        self.resynth_freq_mode_l.setMaximumSize(QSize(80, 16777215))
        self.resynth_freq_mode_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_11.addWidget(self.resynth_freq_mode_l)

        self.resynth_freq_mode_cmb = QComboBox(self.resynth_tone_wid)
        self.resynth_freq_mode_cmb.addItem("")
        self.resynth_freq_mode_cmb.addItem("")
        self.resynth_freq_mode_cmb.addItem("")
        self.resynth_freq_mode_cmb.setObjectName(u"resynth_freq_mode_cmb")
        self.resynth_freq_mode_cmb.setMinimumSize(QSize(80, 0))
        self.resynth_freq_mode_cmb.setMaximumSize(QSize(80, 16777215))
        self.resynth_freq_mode_cmb.setFrame(False)

        self.horizontalLayout_11.addWidget(self.resynth_freq_mode_cmb)

        self.freqs_l = QLabel(self.resynth_tone_wid)
        self.freqs_l.setObjectName(u"freqs_l")
        self.freqs_l.setMinimumSize(QSize(96, 0))
        self.freqs_l.setMaximumSize(QSize(96, 16777215))
        self.freqs_l.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_11.addWidget(self.freqs_l)

        self.freqs_le = QLineEdit(self.resynth_tone_wid)
        self.freqs_le.setObjectName(u"freqs_le")
        sizePolicy.setHeightForWidth(self.freqs_le.sizePolicy().hasHeightForWidth())
        self.freqs_le.setSizePolicy(sizePolicy)
        self.freqs_le.setFrame(False)
        self.freqs_le.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_11.addWidget(self.freqs_le)

        self.atonal_mix_l = QLabel(self.resynth_tone_wid)
        self.atonal_mix_l.setObjectName(u"atonal_mix_l")
        self.atonal_mix_l.setMinimumSize(QSize(96, 0))
        self.atonal_mix_l.setMaximumSize(QSize(96, 16777215))
        self.atonal_mix_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_11.addWidget(self.atonal_mix_l)

        self.atonal_mix_dsb = QDoubleSpinBox(self.resynth_tone_wid)
        self.atonal_mix_dsb.setObjectName(u"atonal_mix_dsb")
        sizePolicy.setHeightForWidth(self.atonal_mix_dsb.sizePolicy().hasHeightForWidth())
        self.atonal_mix_dsb.setSizePolicy(sizePolicy)
        self.atonal_mix_dsb.setMinimumSize(QSize(96, 0))
        self.atonal_mix_dsb.setFrame(False)
        self.atonal_mix_dsb.setAlignment(Qt.AlignCenter)
        self.atonal_mix_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.atonal_mix_dsb.setDecimals(3)
        self.atonal_mix_dsb.setMinimum(0.000000000000000)
        self.atonal_mix_dsb.setMaximum(4.000000000000000)
        self.atonal_mix_dsb.setSingleStep(0.100000000000000)
        self.atonal_mix_dsb.setValue(1.000000000000000)

        self.horizontalLayout_11.addWidget(self.atonal_mix_dsb)


        self.verticalLayout.addWidget(self.resynth_tone_wid)

        self.resynth_mix_wid = QWidget(self.centralwidget)
        self.resynth_mix_wid.setObjectName(u"resynth_mix_wid")
        self.resynth_mix_wid.setEnabled(False)
        self.horizontalLayout_10 = QHBoxLayout(self.resynth_mix_wid)
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.horizontalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.horizontalSpacer_8 = QSpacerItem(128, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_10.addItem(self.horizontalSpacer_8)

        self.resynth_mix_l = QLabel(self.resynth_mix_wid)
        self.resynth_mix_l.setObjectName(u"resynth_mix_l")
        self.resynth_mix_l.setMinimumSize(QSize(80, 0))
        self.resynth_mix_l.setMaximumSize(QSize(80, 16777215))
        self.resynth_mix_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_10.addWidget(self.resynth_mix_l)

        self.resynth_mix_cmb = QComboBox(self.resynth_mix_wid)
        self.resynth_mix_cmb.addItem("")
        self.resynth_mix_cmb.addItem("")
        self.resynth_mix_cmb.addItem("")
        self.resynth_mix_cmb.setObjectName(u"resynth_mix_cmb")
        self.resynth_mix_cmb.setMinimumSize(QSize(80, 0))
        self.resynth_mix_cmb.setMaximumSize(QSize(80, 16777215))
        self.resynth_mix_cmb.setFrame(False)

        self.horizontalLayout_10.addWidget(self.resynth_mix_cmb)

        self.resynth_fade_in_l = QLabel(self.resynth_mix_wid)
        self.resynth_fade_in_l.setObjectName(u"resynth_fade_in_l")
        self.resynth_fade_in_l.setMinimumSize(QSize(96, 0))
        self.resynth_fade_in_l.setMaximumSize(QSize(96, 16777215))
        self.resynth_fade_in_l.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_10.addWidget(self.resynth_fade_in_l)

        self.resynth_fade_in_dsb = QDoubleSpinBox(self.resynth_mix_wid)
        self.resynth_fade_in_dsb.setObjectName(u"resynth_fade_in_dsb")
        self.resynth_fade_in_dsb.setMinimumSize(QSize(64, 0))
        self.resynth_fade_in_dsb.setFrame(False)
        self.resynth_fade_in_dsb.setAlignment(Qt.AlignCenter)
        self.resynth_fade_in_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.resynth_fade_in_dsb.setDecimals(3)
        self.resynth_fade_in_dsb.setMinimum(0.000000000000000)
        self.resynth_fade_in_dsb.setMaximum(1.000000000000000)
        self.resynth_fade_in_dsb.setSingleStep(0.100000000000000)
        self.resynth_fade_in_dsb.setValue(0.250000000000000)

        self.horizontalLayout_10.addWidget(self.resynth_fade_in_dsb)

        self.resynth_fade_out_l = QLabel(self.resynth_mix_wid)
        self.resynth_fade_out_l.setObjectName(u"resynth_fade_out_l")
        self.resynth_fade_out_l.setMinimumSize(QSize(80, 0))
        self.resynth_fade_out_l.setMaximumSize(QSize(80, 16777215))
        self.resynth_fade_out_l.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_10.addWidget(self.resynth_fade_out_l)

        self.resynth_fade_out_dsb = QDoubleSpinBox(self.resynth_mix_wid)
        self.resynth_fade_out_dsb.setObjectName(u"resynth_fade_out_dsb")
        self.resynth_fade_out_dsb.setMinimumSize(QSize(64, 0))
        self.resynth_fade_out_dsb.setFrame(False)
        self.resynth_fade_out_dsb.setAlignment(Qt.AlignCenter)
        self.resynth_fade_out_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.resynth_fade_out_dsb.setDecimals(3)
        self.resynth_fade_out_dsb.setMinimum(0.000000000000000)
        self.resynth_fade_out_dsb.setMaximum(1.000000000000000)
        self.resynth_fade_out_dsb.setSingleStep(0.100000000000000)
        self.resynth_fade_out_dsb.setValue(0.250000000000000)

        self.horizontalLayout_10.addWidget(self.resynth_fade_out_dsb)

        self.st_width_l = QLabel(self.resynth_mix_wid)
        self.st_width_l.setObjectName(u"st_width_l")
        self.st_width_l.setMinimumSize(QSize(96, 0))
        self.st_width_l.setMaximumSize(QSize(96, 16777215))
        self.st_width_l.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_10.addWidget(self.st_width_l)

        self.st_width_dsb = QDoubleSpinBox(self.resynth_mix_wid)
        self.st_width_dsb.setObjectName(u"st_width_dsb")
        self.st_width_dsb.setMinimumSize(QSize(64, 0))
        self.st_width_dsb.setFrame(False)
        self.st_width_dsb.setAlignment(Qt.AlignCenter)
        self.st_width_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.st_width_dsb.setDecimals(3)
        self.st_width_dsb.setMinimum(0.000000000000000)
        self.st_width_dsb.setMaximum(1.000000000000000)
        self.st_width_dsb.setSingleStep(0.100000000000000)
        self.st_width_dsb.setValue(0.500000000000000)

        self.horizontalLayout_10.addWidget(self.st_width_dsb)


        self.verticalLayout.addWidget(self.resynth_mix_wid)

        self.output_path_title_l = QLabel(self.centralwidget)
        self.output_path_title_l.setObjectName(u"output_path_title_l")
        sizePolicy.setHeightForWidth(self.output_path_title_l.sizePolicy().hasHeightForWidth())
        self.output_path_title_l.setSizePolicy(sizePolicy)
        self.output_path_title_l.setMinimumSize(QSize(64, 0))
        self.output_path_title_l.setStyleSheet(u"background-color: rgb(31, 159, 127);\n"
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
        self.settings_title_l.setStyleSheet(u"background-color: rgb(31, 159, 127);\n"
"color: rgb(255, 255, 255);")

        self.verticalLayout.addWidget(self.settings_title_l)

        self.output_file_lyt = QHBoxLayout()
        self.output_file_lyt.setObjectName(u"output_file_lyt")
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy5)

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
        sizePolicy1.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy1)

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

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.output_file_lyt.addItem(self.horizontalSpacer_2)

        self.add_suffix_cb = QCheckBox(self.centralwidget)
        self.add_suffix_cb.setObjectName(u"add_suffix_cb")
        sizePolicy1.setHeightForWidth(self.add_suffix_cb.sizePolicy().hasHeightForWidth())
        self.add_suffix_cb.setSizePolicy(sizePolicy1)
        self.add_suffix_cb.setChecked(True)

        self.output_file_lyt.addWidget(self.add_suffix_cb)

        self.suffix_le = QLineEdit(self.centralwidget)
        self.suffix_le.setObjectName(u"suffix_le")
        sizePolicy1.setHeightForWidth(self.suffix_le.sizePolicy().hasHeightForWidth())
        self.suffix_le.setSizePolicy(sizePolicy1)
        self.suffix_le.setMaxLength(16)
        self.suffix_le.setFrame(False)

        self.output_file_lyt.addWidget(self.suffix_le)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.output_file_lyt.addItem(self.horizontalSpacer_4)

        self.no_overwriting_cb = QCheckBox(self.centralwidget)
        self.no_overwriting_cb.setObjectName(u"no_overwriting_cb")
        self.no_overwriting_cb.setChecked(True)

        self.output_file_lyt.addWidget(self.no_overwriting_cb)


        self.verticalLayout.addLayout(self.output_file_lyt)

        self.trim_lyt = QHBoxLayout()
        self.trim_lyt.setSpacing(0)
        self.trim_lyt.setObjectName(u"trim_lyt")
        self.trim_cb = QCheckBox(self.centralwidget)
        self.trim_cb.setObjectName(u"trim_cb")

        self.trim_lyt.addWidget(self.trim_cb)

        self.trim_mode_cmb = QComboBox(self.centralwidget)
        self.trim_mode_cmb.addItem("")
        self.trim_mode_cmb.addItem("")
        self.trim_mode_cmb.addItem("")
        self.trim_mode_cmb.setObjectName(u"trim_mode_cmb")
        self.trim_mode_cmb.setEnabled(False)

        self.trim_lyt.addWidget(self.trim_mode_cmb)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.trim_lyt.addItem(self.horizontalSpacer)


        self.verticalLayout.addLayout(self.trim_lyt)

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
        self.process_pb.setStyleSheet(u"QPushButton{background-color: rgb(31, 159, 127);\n"
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

        loop_tool_mw.setCentralWidget(self.centralwidget)

        self.retranslateUi(loop_tool_mw)

        self.env_mode_cmb.setCurrentIndex(0)
        self.fade_mode_cmb.setCurrentIndex(0)
        self.resynth_freq_mode_cmb.setCurrentIndex(1)
        self.resynth_mix_cmb.setCurrentIndex(1)
        self.format_cmb.setCurrentIndex(0)
        self.bitdepth_cmb.setCurrentIndex(0)
        self.trim_mode_cmb.setCurrentIndex(2)


        QMetaObject.connectSlotsByName(loop_tool_mw)
    # setupUi

    def retranslateUi(self, loop_tool_mw):
        loop_tool_mw.setWindowTitle(QCoreApplication.translate("loop_tool_mw", u"Loop Tool", None))
        self.files_title_l.setText(QCoreApplication.translate("loop_tool_mw", u"Input Files", None))
#if QT_CONFIG(tooltip)
        self.files_lw.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Drag and drop files or directories\n"
"Right click for context actions\n"
"\n"
"Double-click to play selected file unlooped\n"
"\n"
"SPACE BAR to play/stop selected file LOOPED", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.set_files_tb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Set files to process", None))
#endif // QT_CONFIG(tooltip)
        self.set_files_tb.setText(QCoreApplication.translate("loop_tool_mw", u"...", None))
        self.options_title_l.setText(QCoreApplication.translate("loop_tool_mw", u"Options", None))
#if QT_CONFIG(tooltip)
        self.env_shaping_cb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Invert audio envelope, flattening the amplitude\n"
"May help  looping the sound\n"
"Enable cross-fade to use the 'mask' 'loop' setting\n"
"\n"
"Does alter audio data", None))
#endif // QT_CONFIG(tooltip)
        self.env_shaping_cb.setText(QCoreApplication.translate("loop_tool_mw", u"Envelope Shaping", None))
        self.env_mask_l.setText(QCoreApplication.translate("loop_tool_mw", u"Mask", None))
        self.env_mode_cmb.setItemText(0, QCoreApplication.translate("loop_tool_mw", u"loop", None))
        self.env_mode_cmb.setItemText(1, QCoreApplication.translate("loop_tool_mw", u"full", None))

#if QT_CONFIG(tooltip)
        self.env_mode_cmb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"'loop'	Apply effect  within loop region only\n"
"'full'	Apply everywhere", None))
#endif // QT_CONFIG(tooltip)
        self.env_th_l.setText(QCoreApplication.translate("loop_tool_mw", u"Threshold", None))
#if QT_CONFIG(tooltip)
        self.env_th_dsb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Shaping begins below this volume threshold in dB", None))
#endif // QT_CONFIG(tooltip)
        self.env_min_l.setText(QCoreApplication.translate("loop_tool_mw", u"Minimum", None))
#if QT_CONFIG(tooltip)
        self.env_min_dsb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Shaping ends below this volume threshold in dB", None))
#endif // QT_CONFIG(tooltip)
        self.env_window_l.setText(QCoreApplication.translate("loop_tool_mw", u"Envelope Window", None))
#if QT_CONFIG(tooltip)
        self.env_window_dsb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Window size of envelope transform used for envelope shaping\n"
"\n"
"Higher value, looser correction but more natural sounding\n"
"Lower walue, tighter correction but may sound more distorted", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.detect_loop_cb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Overwrites any pre-existing loop data\n"
"Using 'Detect-Loop' only does not alter audio data", None))
#endif // QT_CONFIG(tooltip)
        self.detect_loop_cb.setText(QCoreApplication.translate("loop_tool_mw", u"Detect Loop", None))
        self.loop_min_length_l.setText(QCoreApplication.translate("loop_tool_mw", u"Min Length", None))
#if QT_CONFIG(tooltip)
        self.loop_min_length_dsb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"MINIMUM loop length in ms\n"
"NOTE: if the search can find a loop which is better but much longer it will favor it", None))
#endif // QT_CONFIG(tooltip)
        self.cues_l.setText(QCoreApplication.translate("loop_tool_mw", u"Candidates", None))
#if QT_CONFIG(tooltip)
        self.cues_sb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Maximum number of cue combinations used to perform the loop search\n"
"The bigger the longer", None))
#endif // QT_CONFIG(tooltip)
        self.loop_window_l.setText(QCoreApplication.translate("loop_tool_mw", u"Window", None))
#if QT_CONFIG(tooltip)
        self.loop_window_dsb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Loop search window size in ms\n"
"\n"
"A larger window can be better for low frequency content but is longer to search\n"
"For high notes or high frequency content, try to reduce value below 10ms", None))
#endif // QT_CONFIG(tooltip)
        self.window_offset_l.setText(QCoreApplication.translate("loop_tool_mw", u"Offset", None))
#if QT_CONFIG(tooltip)
        self.window_offset_dsb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"0	backward, allow loop end very close to end of sample\n"
"0.5	centered (Default) \n"
"1	forward, allow loop start very close to start of sample", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.hash_search_cb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Detect \"perfect\" binary repeat\n"
"Useful to detect loop in preprocessed audio where a perfect loop is likely present\n"
"Add some overhead dependent on the file size", None))
#endif // QT_CONFIG(tooltip)
        self.hash_search_cb.setText(QCoreApplication.translate("loop_tool_mw", u"1st Pass Hash Search", None))
#if QT_CONFIG(tooltip)
        self.range_limits_cb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Apply constraints to loop search scope", None))
#endif // QT_CONFIG(tooltip)
        self.range_limits_cb.setText(QCoreApplication.translate("loop_tool_mw", u"Range Limits", None))
        self.start_range_l.setText(QCoreApplication.translate("loop_tool_mw", u"Start Range", None))
#if QT_CONFIG(tooltip)
        self.start_range_dsb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Restrict loop start before this point (percent)\n"
"Set to 1 to disable restriction", None))
#endif // QT_CONFIG(tooltip)
        self.end_range_l.setText(QCoreApplication.translate("loop_tool_mw", u"End Range", None))
#if QT_CONFIG(tooltip)
        self.end_range_dsb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Restrict loop end after this point (percent)\n"
"Set to 0 to disable restriction", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.crossfade_cb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Create a smoother transition by fading the sample over itself\n"
"Does alter audio data", None))
#endif // QT_CONFIG(tooltip)
        self.crossfade_cb.setText(QCoreApplication.translate("loop_tool_mw", u"Cross-Fade", None))
#if QT_CONFIG(tooltip)
        self.fd_preview_only_cb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Enable cross-fade for preview only\n"
"\n"
"Cross-fade features exist in most sampler VSTi\n"
"This option allows to preview the cross-faded result without baking it in the processed files", None))
#endif // QT_CONFIG(tooltip)
        self.fd_preview_only_cb.setText(QCoreApplication.translate("loop_tool_mw", u"Preview Only", None))
        self.fade_mode_l.setText(QCoreApplication.translate("loop_tool_mw", u"Fade Mode", None))
        self.fade_mode_cmb.setItemText(0, QCoreApplication.translate("loop_tool_mw", u"linear", None))
        self.fade_mode_cmb.setItemText(1, QCoreApplication.translate("loop_tool_mw", u"smoothstep", None))
        self.fade_mode_cmb.setItemText(2, QCoreApplication.translate("loop_tool_mw", u"exp", None))

#if QT_CONFIG(tooltip)
        self.fade_mode_cmb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Fade in/out interpolation type", None))
#endif // QT_CONFIG(tooltip)
        self.fade_in_l.setText(QCoreApplication.translate("loop_tool_mw", u"Fade In", None))
#if QT_CONFIG(tooltip)
        self.fade_in_dsb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Length in percent of loop length\n"
"Values over 1.0 may help for very short loops (a few periods)", None))
#endif // QT_CONFIG(tooltip)
        self.fade_out_l.setText(QCoreApplication.translate("loop_tool_mw", u"Fade Out", None))
#if QT_CONFIG(tooltip)
        self.fade_out_dsb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Fading of the looped region after loop end\n"
"Length in percent of loop length", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.resynth_cb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Re-synthesize audio from FFT analysis\n"
"As a side effect of the FFT re-synthesis, the synthesized part will loop seamlessly\n"
"Overrides loop points", None))
#endif // QT_CONFIG(tooltip)
        self.resynth_cb.setText(QCoreApplication.translate("loop_tool_mw", u"Re-Synthesis", None))
        self.fft_range_l.setText(QCoreApplication.translate("loop_tool_mw", u"FFT Range", None))
        self.fft_range_cmb.setItemText(0, QCoreApplication.translate("loop_tool_mw", u"custom", None))
        self.fft_range_cmb.setItemText(1, QCoreApplication.translate("loop_tool_mw", u"from_loop", None))

#if QT_CONFIG(tooltip)
        self.fft_range_cmb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"'from_loop'	Use loop points as range for FFT analysis", None))
#endif // QT_CONFIG(tooltip)
        self.fft_start_l.setText(QCoreApplication.translate("loop_tool_mw", u"FFT Start", None))
#if QT_CONFIG(tooltip)
        self.fft_start_dsb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Start of FFT analysis in percent of of input audio length", None))
#endif // QT_CONFIG(tooltip)
        self.fft_end_l.setText(QCoreApplication.translate("loop_tool_mw", u"FFT End", None))
#if QT_CONFIG(tooltip)
        self.fft_end_dsb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"FFT analysis end in percent of of input audio length\n"
"\n"
"The shorter the FFT size the coarser, more \"noise-like\" the result\n"
"The longer the smoother, more diluted the result, better for low frequencies", None))
#endif // QT_CONFIG(tooltip)
        self.duration_l.setText(QCoreApplication.translate("loop_tool_mw", u"Duration", None))
#if QT_CONFIG(tooltip)
        self.resynth_duration_dsb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Re-synthesis duration in s\n"
"This also will be the resulting loop lentgh", None))
#endif // QT_CONFIG(tooltip)
        self.resynth_freq_mode_l.setText(QCoreApplication.translate("loop_tool_mw", u"Frequency", None))
        self.resynth_freq_mode_cmb.setItemText(0, QCoreApplication.translate("loop_tool_mw", u"note", None))
        self.resynth_freq_mode_cmb.setItemText(1, QCoreApplication.translate("loop_tool_mw", u"note_pf", None))
        self.resynth_freq_mode_cmb.setItemText(2, QCoreApplication.translate("loop_tool_mw", u"custom", None))

#if QT_CONFIG(tooltip)
        self.resynth_freq_mode_cmb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Fundamental frequency source\n"
"The full harmonic series from fundamental will preserve its phase and will compose the \"tonal\" part\n"
"\n"
"'note'		Frequency from sample note\n"
"'note_pf'		note + pitch fraction (default)\n"
"'custom'		Specify arbitrary frequency with 'freqs'\n"
"\n"
"This is important when mixing synthesized part with original input so both are in phase and the transition feels more natural", None))
#endif // QT_CONFIG(tooltip)
        self.freqs_l.setText(QCoreApplication.translate("loop_tool_mw", u"Freqs", None))
#if QT_CONFIG(tooltip)
        self.freqs_le.setToolTip(QCoreApplication.translate("loop_tool_mw", u"List of note names or frequencies separated by spaces or commas", None))
#endif // QT_CONFIG(tooltip)
        self.atonal_mix_l.setText(QCoreApplication.translate("loop_tool_mw", u"Atonal Mix", None))
#if QT_CONFIG(tooltip)
        self.atonal_mix_dsb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Atonal part amplitude\n"
"Balance \"Tonal\" (Fundamental frequency) VS \"Atonal\" (Noise) part of the sound\n"
"Has no effect with no fundamental frequency set\n"
"\n"
"0: Pure tonal frequency, may sound very artificial\n"
"1: Can sound a bit 'chorusy' or 'verby'", None))
#endif // QT_CONFIG(tooltip)
        self.resynth_mix_l.setText(QCoreApplication.translate("loop_tool_mw", u"Mix", None))
        self.resynth_mix_cmb.setItemText(0, QCoreApplication.translate("loop_tool_mw", u"all", None))
        self.resynth_mix_cmb.setItemText(1, QCoreApplication.translate("loop_tool_mw", u"loop_tail", None))
        self.resynth_mix_cmb.setItemText(2, QCoreApplication.translate("loop_tool_mw", u"loop", None))

#if QT_CONFIG(tooltip)
        self.resynth_mix_cmb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Mix re-synthesized audio with original input\n"
"\n"
"'all'	Replace everything with re-synth\n"
"'loop_tail'	Keep original attack only\n"
"'loop'	Keep original attack and tail", None))
#endif // QT_CONFIG(tooltip)
        self.resynth_fade_in_l.setText(QCoreApplication.translate("loop_tool_mw", u"Fade In", None))
#if QT_CONFIG(tooltip)
        self.resynth_fade_in_dsb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Fading of the synthesized part before loop start\n"
"Length in percent of audio length", None))
#endif // QT_CONFIG(tooltip)
        self.resynth_fade_out_l.setText(QCoreApplication.translate("loop_tool_mw", u"Fade Out", None))
#if QT_CONFIG(tooltip)
        self.resynth_fade_out_dsb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Fading of the synthesized part after loop end\n"
"Length in percent of audio length", None))
#endif // QT_CONFIG(tooltip)
        self.st_width_l.setText(QCoreApplication.translate("loop_tool_mw", u"Stereo Width", None))
#if QT_CONFIG(tooltip)
        self.st_width_dsb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Stereo effect width when using Re-synth mix 'all'\n"
"With this mode left and right channels use a different random seed", None))
#endif // QT_CONFIG(tooltip)
        self.output_path_title_l.setText(QCoreApplication.translate("loop_tool_mw", u"Output Path", None))
        self.output_path_l.setText("")
#if QT_CONFIG(tooltip)
        self.set_output_path_tb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Set output path\n"
"Process files in their respective directory if empty", None))
#endif // QT_CONFIG(tooltip)
        self.set_output_path_tb.setText(QCoreApplication.translate("loop_tool_mw", u"...", None))
        self.settings_title_l.setText(QCoreApplication.translate("loop_tool_mw", u"File Settings", None))
        self.label_2.setText(QCoreApplication.translate("loop_tool_mw", u"Format", None))
        self.format_cmb.setItemText(0, QCoreApplication.translate("loop_tool_mw", u"same", None))
        self.format_cmb.setItemText(1, QCoreApplication.translate("loop_tool_mw", u"wav", None))
        self.format_cmb.setItemText(2, QCoreApplication.translate("loop_tool_mw", u"flac", None))

#if QT_CONFIG(tooltip)
        self.format_cmb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"flac only allows integer format up to 24 bits\n"
"Use wav for 32 bits float", None))
#endif // QT_CONFIG(tooltip)
        self.label.setText(QCoreApplication.translate("loop_tool_mw", u"Bit Depth", None))
        self.bitdepth_cmb.setItemText(0, QCoreApplication.translate("loop_tool_mw", u"same", None))
        self.bitdepth_cmb.setItemText(1, QCoreApplication.translate("loop_tool_mw", u"16", None))
        self.bitdepth_cmb.setItemText(2, QCoreApplication.translate("loop_tool_mw", u"24", None))
        self.bitdepth_cmb.setItemText(3, QCoreApplication.translate("loop_tool_mw", u"32", None))

        self.add_suffix_cb.setText(QCoreApplication.translate("loop_tool_mw", u"Add suffix", None))
#if QT_CONFIG(tooltip)
        self.suffix_le.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Suffix added to the base name", None))
#endif // QT_CONFIG(tooltip)
        self.suffix_le.setText(QCoreApplication.translate("loop_tool_mw", u"_looped", None))
#if QT_CONFIG(tooltip)
        self.no_overwriting_cb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Avoid overwriting original files by moving them to a backup directory", None))
#endif // QT_CONFIG(tooltip)
        self.no_overwriting_cb.setText(QCoreApplication.translate("loop_tool_mw", u"Avoid Overwriting", None))
#if QT_CONFIG(tooltip)
        self.trim_cb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Reduce sample size by discarding data outside of loop region", None))
#endif // QT_CONFIG(tooltip)
        self.trim_cb.setText(QCoreApplication.translate("loop_tool_mw", u"Trim Loop", None))
        self.trim_mode_cmb.setItemText(0, QCoreApplication.translate("loop_tool_mw", u"start", None))
        self.trim_mode_cmb.setItemText(1, QCoreApplication.translate("loop_tool_mw", u"end", None))
        self.trim_mode_cmb.setItemText(2, QCoreApplication.translate("loop_tool_mw", u"all", None))

#if QT_CONFIG(tooltip)
        self.trim_mode_cmb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"start	Trim audio before loop\n"
"end	Trim audio after loop\n"
"all	Keep only looped region", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.process_sel_pb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Process selected item(s) with current settings", None))
#endif // QT_CONFIG(tooltip)
        self.process_sel_pb.setText(QCoreApplication.translate("loop_tool_mw", u"Process Selected", None))
#if QT_CONFIG(tooltip)
        self.preview_pb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Apply current settings to selected item\n"
"and play the result without writing anything", None))
#endif // QT_CONFIG(tooltip)
        self.preview_pb.setText(QCoreApplication.translate("loop_tool_mw", u"Preview Selected", None))
#if QT_CONFIG(tooltip)
        self.process_pb.setToolTip(QCoreApplication.translate("loop_tool_mw", u"Process all input files with current settings", None))
#endif // QT_CONFIG(tooltip)
        self.process_pb.setText(QCoreApplication.translate("loop_tool_mw", u"Batch Process", None))
    # retranslateUi

