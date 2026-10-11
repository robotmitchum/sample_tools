# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'smp_to_ds.ui'
##
## Created by: Qt User Interface Compiler version 6.10.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QCheckBox, QComboBox,
    QDoubleSpinBox, QFrame, QHBoxLayout, QLabel,
    QLayout, QLineEdit, QMainWindow, QProgressBar,
    QPushButton, QSizePolicy, QSpacerItem, QSpinBox,
    QToolButton, QVBoxLayout, QWidget)

class Ui_smp_to_ds_ui(object):
    def setupUi(self, smp_to_ds_ui):
        if not smp_to_ds_ui.objectName():
            smp_to_ds_ui.setObjectName(u"smp_to_ds_ui")
        smp_to_ds_ui.resize(800, 800)
        smp_to_ds_ui.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        self.actionSettings = QAction(smp_to_ds_ui)
        self.actionSettings.setObjectName(u"actionSettings")
        self.save_settings_a = QAction(smp_to_ds_ui)
        self.save_settings_a.setObjectName(u"save_settings_a")
        self.load_settings_a = QAction(smp_to_ds_ui)
        self.load_settings_a.setObjectName(u"load_settings_a")
        self.restore_defaults_a = QAction(smp_to_ds_ui)
        self.restore_defaults_a.setObjectName(u"restore_defaults_a")
        self.centralwidget = QWidget(smp_to_ds_ui)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget_lyt = QVBoxLayout(self.centralwidget)
        self.centralwidget_lyt.setSpacing(4)
        self.centralwidget_lyt.setObjectName(u"centralwidget_lyt")
        self.centralwidget_lyt.setContentsMargins(8, 8, 8, 8)
        self.output_path_lyt = QHBoxLayout()
        self.output_path_lyt.setSpacing(4)
        self.output_path_lyt.setObjectName(u"output_path_lyt")
        self.output_path_lyt.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.output_path_l = QLabel(self.centralwidget)
        self.output_path_l.setObjectName(u"output_path_l")
        font = QFont()
        font.setBold(True)
        self.output_path_l.setFont(font)
        self.output_path_l.setStyleSheet(u"QLabel{color: rgb(30, 161, 205);}")
        self.output_path_l.setAlignment(Qt.AlignCenter)

        self.output_path_lyt.addWidget(self.output_path_l)

        self.output_path_tb = QToolButton(self.centralwidget)
        self.output_path_tb.setObjectName(u"output_path_tb")
        self.output_path_tb.setFont(font)

        self.output_path_lyt.addWidget(self.output_path_tb)


        self.centralwidget_lyt.addLayout(self.output_path_lyt)

        self.sample_title_l = QLabel(self.centralwidget)
        self.sample_title_l.setObjectName(u"sample_title_l")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sample_title_l.sizePolicy().hasHeightForWidth())
        self.sample_title_l.setSizePolicy(sizePolicy)
        self.sample_title_l.setMinimumSize(QSize(64, 0))
        self.sample_title_l.setStyleSheet(u"background-color: rgb(63, 95, 127);\n"
"color: rgb(255, 255, 255);")

        self.centralwidget_lyt.addWidget(self.sample_title_l)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.centralwidget_lyt.addItem(self.verticalSpacer)

        self.pattern_lyt = QHBoxLayout()
        self.pattern_lyt.setSpacing(4)
        self.pattern_lyt.setObjectName(u"pattern_lyt")
        self.spacer_7 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.pattern_lyt.addItem(self.spacer_7)

        self.pattern_pb = QPushButton(self.centralwidget)
        self.pattern_pb.setObjectName(u"pattern_pb")
        self.pattern_pb.setFont(font)

        self.pattern_lyt.addWidget(self.pattern_pb)

        self.pattern_le = QLineEdit(self.centralwidget)
        self.pattern_le.setObjectName(u"pattern_le")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.pattern_le.sizePolicy().hasHeightForWidth())
        self.pattern_le.setSizePolicy(sizePolicy1)
        self.pattern_le.setMinimumSize(QSize(192, 0))
        self.pattern_le.setFrame(False)
        self.pattern_le.setAlignment(Qt.AlignCenter)

        self.pattern_lyt.addWidget(self.pattern_le)

        self.grpnaming_l = QLabel(self.centralwidget)
        self.grpnaming_l.setObjectName(u"grpnaming_l")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.grpnaming_l.sizePolicy().hasHeightForWidth())
        self.grpnaming_l.setSizePolicy(sizePolicy2)

        self.pattern_lyt.addWidget(self.grpnaming_l)

        self.groupnaming_cmb = QComboBox(self.centralwidget)
        self.groupnaming_cmb.addItem("")
        self.groupnaming_cmb.addItem("")
        self.groupnaming_cmb.addItem("")
        self.groupnaming_cmb.addItem("")
        self.groupnaming_cmb.addItem("")
        self.groupnaming_cmb.setObjectName(u"groupnaming_cmb")
        self.groupnaming_cmb.setFrame(False)

        self.pattern_lyt.addWidget(self.groupnaming_cmb)

        self.spacer_12 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.pattern_lyt.addItem(self.spacer_12)

        self.override_cb = QCheckBox(self.centralwidget)
        self.override_cb.setObjectName(u"override_cb")
        sizePolicy2.setHeightForWidth(self.override_cb.sizePolicy().hasHeightForWidth())
        self.override_cb.setSizePolicy(sizePolicy2)

        self.pattern_lyt.addWidget(self.override_cb)

        self.loop_cb = QCheckBox(self.centralwidget)
        self.loop_cb.setObjectName(u"loop_cb")
        sizePolicy2.setHeightForWidth(self.loop_cb.sizePolicy().hasHeightForWidth())
        self.loop_cb.setSizePolicy(sizePolicy2)
        self.loop_cb.setChecked(True)

        self.pattern_lyt.addWidget(self.loop_cb)

        self.spacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.pattern_lyt.addItem(self.spacer_3)


        self.centralwidget_lyt.addLayout(self.pattern_lyt)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.centralwidget_lyt.addItem(self.verticalSpacer_2)

        self.env_title_l = QLabel(self.centralwidget)
        self.env_title_l.setObjectName(u"env_title_l")
        sizePolicy.setHeightForWidth(self.env_title_l.sizePolicy().hasHeightForWidth())
        self.env_title_l.setSizePolicy(sizePolicy)
        self.env_title_l.setMinimumSize(QSize(64, 0))
        self.env_title_l.setStyleSheet(u"background-color: rgb(63, 95, 127);\n"
"color: rgb(255, 255, 255);")

        self.centralwidget_lyt.addWidget(self.env_title_l)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.centralwidget_lyt.addItem(self.verticalSpacer_3)

        self.adsr_lyt = QHBoxLayout()
        self.adsr_lyt.setSpacing(4)
        self.adsr_lyt.setObjectName(u"adsr_lyt")
        self.spacer_6 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.adsr_lyt.addItem(self.spacer_6)

        self.adsr_enable_cb = QCheckBox(self.centralwidget)
        self.adsr_enable_cb.setObjectName(u"adsr_enable_cb")
        sizePolicy2.setHeightForWidth(self.adsr_enable_cb.sizePolicy().hasHeightForWidth())
        self.adsr_enable_cb.setSizePolicy(sizePolicy2)
        self.adsr_enable_cb.setChecked(True)

        self.adsr_lyt.addWidget(self.adsr_enable_cb)

        self.adsr_wid = QWidget(self.centralwidget)
        self.adsr_wid.setObjectName(u"adsr_wid")
        self.adsr_wid.setEnabled(True)
        font1 = QFont()
        self.adsr_wid.setFont(font1)
        self.adsr_wid.setStyleSheet(u"")
        self.horizontalLayout_6 = QHBoxLayout(self.adsr_wid)
        self.horizontalLayout_6.setSpacing(0)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.ADSR_pb = QPushButton(self.adsr_wid)
        self.ADSR_pb.setObjectName(u"ADSR_pb")
        self.ADSR_pb.setFont(font)

        self.horizontalLayout_6.addWidget(self.ADSR_pb)

        self.A_dsb = QDoubleSpinBox(self.adsr_wid)
        self.A_dsb.setObjectName(u"A_dsb")
        self.A_dsb.setContextMenuPolicy(Qt.NoContextMenu)
        self.A_dsb.setFrame(False)
        self.A_dsb.setAlignment(Qt.AlignCenter)
        self.A_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.A_dsb.setDecimals(3)
        self.A_dsb.setMaximum(100.000000000000000)
        self.A_dsb.setSingleStep(0.010000000000000)
        self.A_dsb.setValue(0.001000000000000)

        self.horizontalLayout_6.addWidget(self.A_dsb)

        self.D_dsb = QDoubleSpinBox(self.adsr_wid)
        self.D_dsb.setObjectName(u"D_dsb")
        self.D_dsb.setContextMenuPolicy(Qt.NoContextMenu)
        self.D_dsb.setFrame(False)
        self.D_dsb.setAlignment(Qt.AlignCenter)
        self.D_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.D_dsb.setDecimals(3)
        self.D_dsb.setMaximum(100.000000000000000)
        self.D_dsb.setSingleStep(0.010000000000000)
        self.D_dsb.setValue(0.375000000000000)

        self.horizontalLayout_6.addWidget(self.D_dsb)

        self.S_dsb = QDoubleSpinBox(self.adsr_wid)
        self.S_dsb.setObjectName(u"S_dsb")
        self.S_dsb.setContextMenuPolicy(Qt.NoContextMenu)
        self.S_dsb.setFrame(False)
        self.S_dsb.setAlignment(Qt.AlignCenter)
        self.S_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.S_dsb.setDecimals(3)
        self.S_dsb.setMaximum(100.000000000000000)
        self.S_dsb.setSingleStep(0.010000000000000)
        self.S_dsb.setValue(1.000000000000000)

        self.horizontalLayout_6.addWidget(self.S_dsb)

        self.R_dsb = QDoubleSpinBox(self.adsr_wid)
        self.R_dsb.setObjectName(u"R_dsb")
        self.R_dsb.setContextMenuPolicy(Qt.NoContextMenu)
        self.R_dsb.setFrame(False)
        self.R_dsb.setAlignment(Qt.AlignCenter)
        self.R_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.R_dsb.setDecimals(3)
        self.R_dsb.setMaximum(100.000000000000000)
        self.R_dsb.setSingleStep(0.010000000000000)
        self.R_dsb.setValue(0.375000000000000)

        self.horizontalLayout_6.addWidget(self.R_dsb)

        self.spacer_11 = QSpacerItem(37, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_6.addItem(self.spacer_11)

        self.ADRr_pb = QPushButton(self.adsr_wid)
        self.ADRr_pb.setObjectName(u"ADRr_pb")
        self.ADRr_pb.setFont(font)

        self.horizontalLayout_6.addWidget(self.ADRr_pb)

        self.Ar_dsb = QDoubleSpinBox(self.adsr_wid)
        self.Ar_dsb.setObjectName(u"Ar_dsb")
        self.Ar_dsb.setContextMenuPolicy(Qt.NoContextMenu)
        self.Ar_dsb.setFrame(False)
        self.Ar_dsb.setAlignment(Qt.AlignCenter)
        self.Ar_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.Ar_dsb.setDecimals(2)
        self.Ar_dsb.setMinimum(-100.000000000000000)
        self.Ar_dsb.setMaximum(100.000000000000000)
        self.Ar_dsb.setSingleStep(1.000000000000000)
        self.Ar_dsb.setValue(-100.000000000000000)

        self.horizontalLayout_6.addWidget(self.Ar_dsb)

        self.Dr_dsb = QDoubleSpinBox(self.adsr_wid)
        self.Dr_dsb.setObjectName(u"Dr_dsb")
        self.Dr_dsb.setContextMenuPolicy(Qt.NoContextMenu)
        self.Dr_dsb.setFrame(False)
        self.Dr_dsb.setAlignment(Qt.AlignCenter)
        self.Dr_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.Dr_dsb.setDecimals(2)
        self.Dr_dsb.setMinimum(-100.000000000000000)
        self.Dr_dsb.setMaximum(100.000000000000000)
        self.Dr_dsb.setSingleStep(1.000000000000000)
        self.Dr_dsb.setValue(100.000000000000000)

        self.horizontalLayout_6.addWidget(self.Dr_dsb)

        self.Rr_dsb = QDoubleSpinBox(self.adsr_wid)
        self.Rr_dsb.setObjectName(u"Rr_dsb")
        self.Rr_dsb.setContextMenuPolicy(Qt.NoContextMenu)
        self.Rr_dsb.setFrame(False)
        self.Rr_dsb.setAlignment(Qt.AlignCenter)
        self.Rr_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.Rr_dsb.setDecimals(2)
        self.Rr_dsb.setMinimum(-100.000000000000000)
        self.Rr_dsb.setMaximum(100.000000000000000)
        self.Rr_dsb.setSingleStep(1.000000000000000)
        self.Rr_dsb.setValue(100.000000000000000)

        self.horizontalLayout_6.addWidget(self.Rr_dsb)


        self.adsr_lyt.addWidget(self.adsr_wid)

        self.spacer_15 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.adsr_lyt.addItem(self.spacer_15)


        self.centralwidget_lyt.addLayout(self.adsr_lyt)

        self.verticalSpacer_12 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.centralwidget_lyt.addItem(self.verticalSpacer_12)

        self.fake_trigger_title_l = QLabel(self.centralwidget)
        self.fake_trigger_title_l.setObjectName(u"fake_trigger_title_l")
        sizePolicy.setHeightForWidth(self.fake_trigger_title_l.sizePolicy().hasHeightForWidth())
        self.fake_trigger_title_l.setSizePolicy(sizePolicy)
        self.fake_trigger_title_l.setMinimumSize(QSize(64, 0))
        self.fake_trigger_title_l.setStyleSheet(u"background-color: rgb(63, 95, 127);\n"
"color: rgb(255, 255, 255);")

        self.centralwidget_lyt.addWidget(self.fake_trigger_title_l)

        self.verticalSpacer_11 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.centralwidget_lyt.addItem(self.verticalSpacer_11)

        self.fake_leg_lyt = QHBoxLayout()
        self.fake_leg_lyt.setSpacing(4)
        self.fake_leg_lyt.setObjectName(u"fake_leg_lyt")
        self.horizontalSpacer_23 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.fake_leg_lyt.addItem(self.horizontalSpacer_23)

        self.fake_leg_cb = QCheckBox(self.centralwidget)
        self.fake_leg_cb.setObjectName(u"fake_leg_cb")
        sizePolicy2.setHeightForWidth(self.fake_leg_cb.sizePolicy().hasHeightForWidth())
        self.fake_leg_cb.setSizePolicy(sizePolicy2)

        self.fake_leg_lyt.addWidget(self.fake_leg_cb)

        self.fake_leg_wid = QWidget(self.centralwidget)
        self.fake_leg_wid.setObjectName(u"fake_leg_wid")
        self.fake_leg_wid.setEnabled(False)
        self.horizontalLayout_4 = QHBoxLayout(self.fake_leg_wid)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalSpacer_24 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_24)

        self.fk_leg_start_l = QLabel(self.fake_leg_wid)
        self.fk_leg_start_l.setObjectName(u"fk_leg_start_l")

        self.horizontalLayout_4.addWidget(self.fk_leg_start_l)

        self.fk_leg_start_dsb = QDoubleSpinBox(self.fake_leg_wid)
        self.fk_leg_start_dsb.setObjectName(u"fk_leg_start_dsb")
        self.fk_leg_start_dsb.setContextMenuPolicy(Qt.NoContextMenu)
        self.fk_leg_start_dsb.setFrame(False)
        self.fk_leg_start_dsb.setAlignment(Qt.AlignCenter)
        self.fk_leg_start_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.fk_leg_start_dsb.setDecimals(3)
        self.fk_leg_start_dsb.setMaximum(100.000000000000000)
        self.fk_leg_start_dsb.setSingleStep(0.010000000000000)
        self.fk_leg_start_dsb.setValue(0.100000000000000)

        self.horizontalLayout_4.addWidget(self.fk_leg_start_dsb)

        self.horizontalSpacer_25 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_25)

        self.fk_leg_a_l = QLabel(self.fake_leg_wid)
        self.fk_leg_a_l.setObjectName(u"fk_leg_a_l")

        self.horizontalLayout_4.addWidget(self.fk_leg_a_l)

        self.fk_leg_a_dsb = QDoubleSpinBox(self.fake_leg_wid)
        self.fk_leg_a_dsb.setObjectName(u"fk_leg_a_dsb")
        self.fk_leg_a_dsb.setContextMenuPolicy(Qt.NoContextMenu)
        self.fk_leg_a_dsb.setFrame(False)
        self.fk_leg_a_dsb.setAlignment(Qt.AlignCenter)
        self.fk_leg_a_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.fk_leg_a_dsb.setDecimals(3)
        self.fk_leg_a_dsb.setMaximum(100.000000000000000)
        self.fk_leg_a_dsb.setSingleStep(0.010000000000000)
        self.fk_leg_a_dsb.setValue(0.100000000000000)

        self.horizontalLayout_4.addWidget(self.fk_leg_a_dsb)

        self.horizontalSpacer_26 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_26)

        self.fk_leg_a_curve_l = QLabel(self.fake_leg_wid)
        self.fk_leg_a_curve_l.setObjectName(u"fk_leg_a_curve_l")

        self.horizontalLayout_4.addWidget(self.fk_leg_a_curve_l)

        self.fk_leg_a_curve_dsb = QDoubleSpinBox(self.fake_leg_wid)
        self.fk_leg_a_curve_dsb.setObjectName(u"fk_leg_a_curve_dsb")
        self.fk_leg_a_curve_dsb.setContextMenuPolicy(Qt.NoContextMenu)
        self.fk_leg_a_curve_dsb.setFrame(False)
        self.fk_leg_a_curve_dsb.setAlignment(Qt.AlignCenter)
        self.fk_leg_a_curve_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.fk_leg_a_curve_dsb.setDecimals(2)
        self.fk_leg_a_curve_dsb.setMinimum(-100.000000000000000)
        self.fk_leg_a_curve_dsb.setMaximum(100.000000000000000)
        self.fk_leg_a_curve_dsb.setSingleStep(1.000000000000000)
        self.fk_leg_a_curve_dsb.setValue(-100.000000000000000)

        self.horizontalLayout_4.addWidget(self.fk_leg_a_curve_dsb)

        self.horizontalSpacer_28 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_28)


        self.fake_leg_lyt.addWidget(self.fake_leg_wid)

        self.horizontalSpacer_27 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.fake_leg_lyt.addItem(self.horizontalSpacer_27)


        self.centralwidget_lyt.addLayout(self.fake_leg_lyt)

        self.fake_rls_lyt = QHBoxLayout()
        self.fake_rls_lyt.setSpacing(4)
        self.fake_rls_lyt.setObjectName(u"fake_rls_lyt")
        self.horizontalSpacer_17 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.fake_rls_lyt.addItem(self.horizontalSpacer_17)

        self.fake_rls_cb = QCheckBox(self.centralwidget)
        self.fake_rls_cb.setObjectName(u"fake_rls_cb")
        sizePolicy2.setHeightForWidth(self.fake_rls_cb.sizePolicy().hasHeightForWidth())
        self.fake_rls_cb.setSizePolicy(sizePolicy2)

        self.fake_rls_lyt.addWidget(self.fake_rls_cb)

        self.fake_options_wid = QWidget(self.centralwidget)
        self.fake_options_wid.setObjectName(u"fake_options_wid")
        self.fake_options_wid.setEnabled(False)
        self.horizontalLayout_2 = QHBoxLayout(self.fake_options_wid)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalSpacer_19 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_19)

        self.fk_rls_mode_l = QLabel(self.fake_options_wid)
        self.fk_rls_mode_l.setObjectName(u"fk_rls_mode_l")

        self.horizontalLayout_2.addWidget(self.fk_rls_mode_l)

        self.fk_rls_mode_cmb = QComboBox(self.fake_options_wid)
        self.fk_rls_mode_cmb.addItem("")
        self.fk_rls_mode_cmb.addItem("")
        self.fk_rls_mode_cmb.addItem("")
        self.fk_rls_mode_cmb.setObjectName(u"fk_rls_mode_cmb")
        sizePolicy.setHeightForWidth(self.fk_rls_mode_cmb.sizePolicy().hasHeightForWidth())
        self.fk_rls_mode_cmb.setSizePolicy(sizePolicy)
        self.fk_rls_mode_cmb.setMinimumSize(QSize(64, 0))
        self.fk_rls_mode_cmb.setFrame(False)

        self.horizontalLayout_2.addWidget(self.fk_rls_mode_cmb)

        self.horizontalSpacer_22 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_22)

        self.fk_volume_l = QLabel(self.fake_options_wid)
        self.fk_volume_l.setObjectName(u"fk_volume_l")

        self.horizontalLayout_2.addWidget(self.fk_volume_l)

        self.fk_volume_dsb = QDoubleSpinBox(self.fake_options_wid)
        self.fk_volume_dsb.setObjectName(u"fk_volume_dsb")
        self.fk_volume_dsb.setMinimumSize(QSize(64, 0))
        self.fk_volume_dsb.setFrame(False)
        self.fk_volume_dsb.setAlignment(Qt.AlignCenter)
        self.fk_volume_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.fk_volume_dsb.setDecimals(3)
        self.fk_volume_dsb.setMinimum(-96.000000000000000)
        self.fk_volume_dsb.setMaximum(0.000000000000000)
        self.fk_volume_dsb.setValue(-15.000000000000000)

        self.horizontalLayout_2.addWidget(self.fk_volume_dsb)

        self.horizontalSpacer_20 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_20)

        self.fk_rls_tweaks_wid = QWidget(self.fake_options_wid)
        self.fk_rls_tweaks_wid.setObjectName(u"fk_rls_tweaks_wid")
        self.horizontalLayout_5 = QHBoxLayout(self.fk_rls_tweaks_wid)
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.fk_tuning_l = QLabel(self.fk_rls_tweaks_wid)
        self.fk_tuning_l.setObjectName(u"fk_tuning_l")

        self.horizontalLayout_5.addWidget(self.fk_tuning_l)

        self.fk_tuning_dsb = QDoubleSpinBox(self.fk_rls_tweaks_wid)
        self.fk_tuning_dsb.setObjectName(u"fk_tuning_dsb")
        self.fk_tuning_dsb.setMinimumSize(QSize(64, 0))
        self.fk_tuning_dsb.setFrame(False)
        self.fk_tuning_dsb.setAlignment(Qt.AlignCenter)
        self.fk_tuning_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.fk_tuning_dsb.setDecimals(3)
        self.fk_tuning_dsb.setMinimum(-48.000000000000000)
        self.fk_tuning_dsb.setMaximum(48.000000000000000)
        self.fk_tuning_dsb.setValue(0.000000000000000)

        self.horizontalLayout_5.addWidget(self.fk_tuning_dsb)

        self.horizontalSpacer_21 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_21)

        self.fk_cutoff_l = QLabel(self.fk_rls_tweaks_wid)
        self.fk_cutoff_l.setObjectName(u"fk_cutoff_l")

        self.horizontalLayout_5.addWidget(self.fk_cutoff_l)

        self.fk_cutoff_dsb = QDoubleSpinBox(self.fk_rls_tweaks_wid)
        self.fk_cutoff_dsb.setObjectName(u"fk_cutoff_dsb")
        self.fk_cutoff_dsb.setMinimumSize(QSize(64, 0))
        self.fk_cutoff_dsb.setFrame(False)
        self.fk_cutoff_dsb.setAlignment(Qt.AlignCenter)
        self.fk_cutoff_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.fk_cutoff_dsb.setDecimals(0)
        self.fk_cutoff_dsb.setMinimum(0.000000000000000)
        self.fk_cutoff_dsb.setMaximum(22000.000000000000000)
        self.fk_cutoff_dsb.setValue(1000.000000000000000)

        self.horizontalLayout_5.addWidget(self.fk_cutoff_dsb)


        self.horizontalLayout_2.addWidget(self.fk_rls_tweaks_wid)


        self.fake_rls_lyt.addWidget(self.fake_options_wid)

        self.horizontalSpacer_18 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.fake_rls_lyt.addItem(self.horizontalSpacer_18)


        self.centralwidget_lyt.addLayout(self.fake_rls_lyt)

        self.fk_adsr_wid = QWidget(self.centralwidget)
        self.fk_adsr_wid.setObjectName(u"fk_adsr_wid")
        self.fk_adsr_wid.setEnabled(False)
        self.horizontalLayout_3 = QHBoxLayout(self.fk_adsr_wid)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.spacer_18 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.spacer_18)

        self.fk_ADSR_pb = QPushButton(self.fk_adsr_wid)
        self.fk_ADSR_pb.setObjectName(u"fk_ADSR_pb")
        self.fk_ADSR_pb.setFont(font)

        self.horizontalLayout_3.addWidget(self.fk_ADSR_pb)

        self.fk_A_dsb = QDoubleSpinBox(self.fk_adsr_wid)
        self.fk_A_dsb.setObjectName(u"fk_A_dsb")
        self.fk_A_dsb.setContextMenuPolicy(Qt.NoContextMenu)
        self.fk_A_dsb.setFrame(False)
        self.fk_A_dsb.setAlignment(Qt.AlignCenter)
        self.fk_A_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.fk_A_dsb.setDecimals(3)
        self.fk_A_dsb.setMaximum(100.000000000000000)
        self.fk_A_dsb.setSingleStep(0.010000000000000)
        self.fk_A_dsb.setValue(0.001000000000000)

        self.horizontalLayout_3.addWidget(self.fk_A_dsb)

        self.fk_D_dsb = QDoubleSpinBox(self.fk_adsr_wid)
        self.fk_D_dsb.setObjectName(u"fk_D_dsb")
        self.fk_D_dsb.setContextMenuPolicy(Qt.NoContextMenu)
        self.fk_D_dsb.setFrame(False)
        self.fk_D_dsb.setAlignment(Qt.AlignCenter)
        self.fk_D_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.fk_D_dsb.setDecimals(3)
        self.fk_D_dsb.setMaximum(100.000000000000000)
        self.fk_D_dsb.setSingleStep(0.010000000000000)
        self.fk_D_dsb.setValue(0.050000000000000)

        self.horizontalLayout_3.addWidget(self.fk_D_dsb)

        self.fk_S_dsb = QDoubleSpinBox(self.fk_adsr_wid)
        self.fk_S_dsb.setObjectName(u"fk_S_dsb")
        self.fk_S_dsb.setContextMenuPolicy(Qt.NoContextMenu)
        self.fk_S_dsb.setFrame(False)
        self.fk_S_dsb.setAlignment(Qt.AlignCenter)
        self.fk_S_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.fk_S_dsb.setDecimals(3)
        self.fk_S_dsb.setMaximum(100.000000000000000)
        self.fk_S_dsb.setSingleStep(0.010000000000000)
        self.fk_S_dsb.setValue(0.000000000000000)

        self.horizontalLayout_3.addWidget(self.fk_S_dsb)

        self.fk_R_dsb = QDoubleSpinBox(self.fk_adsr_wid)
        self.fk_R_dsb.setObjectName(u"fk_R_dsb")
        self.fk_R_dsb.setContextMenuPolicy(Qt.NoContextMenu)
        self.fk_R_dsb.setFrame(False)
        self.fk_R_dsb.setAlignment(Qt.AlignCenter)
        self.fk_R_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.fk_R_dsb.setDecimals(3)
        self.fk_R_dsb.setMaximum(100.000000000000000)
        self.fk_R_dsb.setSingleStep(0.010000000000000)
        self.fk_R_dsb.setValue(0.050000000000000)

        self.horizontalLayout_3.addWidget(self.fk_R_dsb)

        self.spacer_21 = QSpacerItem(40, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.spacer_21)

        self.fk_ADRr_pb = QPushButton(self.fk_adsr_wid)
        self.fk_ADRr_pb.setObjectName(u"fk_ADRr_pb")
        self.fk_ADRr_pb.setFont(font)

        self.horizontalLayout_3.addWidget(self.fk_ADRr_pb)

        self.fk_Ar_dsb = QDoubleSpinBox(self.fk_adsr_wid)
        self.fk_Ar_dsb.setObjectName(u"fk_Ar_dsb")
        self.fk_Ar_dsb.setContextMenuPolicy(Qt.NoContextMenu)
        self.fk_Ar_dsb.setFrame(False)
        self.fk_Ar_dsb.setAlignment(Qt.AlignCenter)
        self.fk_Ar_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.fk_Ar_dsb.setDecimals(2)
        self.fk_Ar_dsb.setMinimum(-100.000000000000000)
        self.fk_Ar_dsb.setMaximum(100.000000000000000)
        self.fk_Ar_dsb.setSingleStep(1.000000000000000)
        self.fk_Ar_dsb.setValue(-100.000000000000000)

        self.horizontalLayout_3.addWidget(self.fk_Ar_dsb)

        self.fk_Dr_dsb = QDoubleSpinBox(self.fk_adsr_wid)
        self.fk_Dr_dsb.setObjectName(u"fk_Dr_dsb")
        self.fk_Dr_dsb.setContextMenuPolicy(Qt.NoContextMenu)
        self.fk_Dr_dsb.setFrame(False)
        self.fk_Dr_dsb.setAlignment(Qt.AlignCenter)
        self.fk_Dr_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.fk_Dr_dsb.setDecimals(2)
        self.fk_Dr_dsb.setMinimum(-100.000000000000000)
        self.fk_Dr_dsb.setMaximum(100.000000000000000)
        self.fk_Dr_dsb.setSingleStep(1.000000000000000)
        self.fk_Dr_dsb.setValue(100.000000000000000)

        self.horizontalLayout_3.addWidget(self.fk_Dr_dsb)

        self.fk_Rr_dsb = QDoubleSpinBox(self.fk_adsr_wid)
        self.fk_Rr_dsb.setObjectName(u"fk_Rr_dsb")
        self.fk_Rr_dsb.setContextMenuPolicy(Qt.NoContextMenu)
        self.fk_Rr_dsb.setFrame(False)
        self.fk_Rr_dsb.setAlignment(Qt.AlignCenter)
        self.fk_Rr_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.fk_Rr_dsb.setDecimals(2)
        self.fk_Rr_dsb.setMinimum(-100.000000000000000)
        self.fk_Rr_dsb.setMaximum(100.000000000000000)
        self.fk_Rr_dsb.setSingleStep(1.000000000000000)
        self.fk_Rr_dsb.setValue(100.000000000000000)

        self.horizontalLayout_3.addWidget(self.fk_Rr_dsb)

        self.spacer_22 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.spacer_22)


        self.centralwidget_lyt.addWidget(self.fk_adsr_wid)

        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.centralwidget_lyt.addItem(self.verticalSpacer_4)

        self.mapping_l = QLabel(self.centralwidget)
        self.mapping_l.setObjectName(u"mapping_l")
        sizePolicy.setHeightForWidth(self.mapping_l.sizePolicy().hasHeightForWidth())
        self.mapping_l.setSizePolicy(sizePolicy)
        self.mapping_l.setMinimumSize(QSize(64, 0))
        self.mapping_l.setStyleSheet(u"background-color: rgb(63, 95, 127);\n"
"color: rgb(255, 255, 255);")

        self.centralwidget_lyt.addWidget(self.mapping_l)

        self.verticalSpacer_5 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.centralwidget_lyt.addItem(self.verticalSpacer_5)

        self.spread_seq_lyt = QHBoxLayout()
        self.spread_seq_lyt.setSpacing(4)
        self.spread_seq_lyt.setObjectName(u"spread_seq_lyt")
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.spread_seq_lyt.addItem(self.horizontalSpacer)

        self.notespread_l = QLabel(self.centralwidget)
        self.notespread_l.setObjectName(u"notespread_l")
        self.notespread_l.setMinimumSize(QSize(96, 0))

        self.spread_seq_lyt.addWidget(self.notespread_l)

        self.spread_cmb = QComboBox(self.centralwidget)
        self.spread_cmb.addItem("")
        self.spread_cmb.addItem("")
        self.spread_cmb.addItem("")
        self.spread_cmb.addItem("")
        self.spread_cmb.setObjectName(u"spread_cmb")
        self.spread_cmb.setFrame(False)

        self.spread_seq_lyt.addWidget(self.spread_cmb)

        self.horizontalSpacer_16 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.spread_seq_lyt.addItem(self.horizontalSpacer_16)

        self.pad_vel_cb = QCheckBox(self.centralwidget)
        self.pad_vel_cb.setObjectName(u"pad_vel_cb")

        self.spread_seq_lyt.addWidget(self.pad_vel_cb)

        self.horizontalSpacer_13 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.spread_seq_lyt.addItem(self.horizontalSpacer_13)

        self.seqmode_l = QLabel(self.centralwidget)
        self.seqmode_l.setObjectName(u"seqmode_l")

        self.spread_seq_lyt.addWidget(self.seqmode_l)

        self.seq_mode_cmb = QComboBox(self.centralwidget)
        self.seq_mode_cmb.addItem("")
        self.seq_mode_cmb.addItem("")
        self.seq_mode_cmb.addItem("")
        self.seq_mode_cmb.addItem("")
        self.seq_mode_cmb.setObjectName(u"seq_mode_cmb")
        self.seq_mode_cmb.setFrame(False)

        self.spread_seq_lyt.addWidget(self.seq_mode_cmb)

        self.horizontalSpacer_14 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.spread_seq_lyt.addItem(self.horizontalSpacer_14)


        self.centralwidget_lyt.addLayout(self.spread_seq_lyt)

        self.mapping_lyt = QHBoxLayout()
        self.mapping_lyt.setSpacing(4)
        self.mapping_lyt.setObjectName(u"mapping_lyt")
        self.spacer_9 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.mapping_lyt.addItem(self.spacer_9)

        self.limit_pb = QPushButton(self.centralwidget)
        self.limit_pb.setObjectName(u"limit_pb")
        self.limit_pb.setFont(font)

        self.mapping_lyt.addWidget(self.limit_pb)

        self.limit_le = QLineEdit(self.centralwidget)
        self.limit_le.setObjectName(u"limit_le")
        sizePolicy2.setHeightForWidth(self.limit_le.sizePolicy().hasHeightForWidth())
        self.limit_le.setSizePolicy(sizePolicy2)
        self.limit_le.setFrame(False)
        self.limit_le.setAlignment(Qt.AlignCenter)

        self.mapping_lyt.addWidget(self.limit_le)

        self.note_limit_cmb = QComboBox(self.centralwidget)
        self.note_limit_cmb.addItem("")
        self.note_limit_cmb.addItem("")
        self.note_limit_cmb.setObjectName(u"note_limit_cmb")
        self.note_limit_cmb.setFrame(False)

        self.mapping_lyt.addWidget(self.note_limit_cmb)

        self.spacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.mapping_lyt.addItem(self.spacer_2)

        self.fake_rr_pb = QPushButton(self.centralwidget)
        self.fake_rr_pb.setObjectName(u"fake_rr_pb")
        self.fake_rr_pb.setFont(font)

        self.mapping_lyt.addWidget(self.fake_rr_pb)

        self.rrofs_le = QLineEdit(self.centralwidget)
        self.rrofs_le.setObjectName(u"rrofs_le")
        sizePolicy2.setHeightForWidth(self.rrofs_le.sizePolicy().hasHeightForWidth())
        self.rrofs_le.setSizePolicy(sizePolicy2)
        self.rrofs_le.setFrame(False)
        self.rrofs_le.setAlignment(Qt.AlignCenter)

        self.mapping_lyt.addWidget(self.rrofs_le)

        self.rr_bounds_cb = QCheckBox(self.centralwidget)
        self.rr_bounds_cb.setObjectName(u"rr_bounds_cb")
        self.rr_bounds_cb.setChecked(True)

        self.mapping_lyt.addWidget(self.rr_bounds_cb)

        self.spacer_16 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.mapping_lyt.addItem(self.spacer_16)


        self.centralwidget_lyt.addLayout(self.mapping_lyt)

        self.transpose_lyt = QHBoxLayout()
        self.transpose_lyt.setSpacing(4)
        self.transpose_lyt.setObjectName(u"transpose_lyt")
        self.spacer_19 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.transpose_lyt.addItem(self.spacer_19)

        self.transpose_l = QLabel(self.centralwidget)
        self.transpose_l.setObjectName(u"transpose_l")
        self.transpose_l.setMinimumSize(QSize(96, 0))

        self.transpose_lyt.addWidget(self.transpose_l)

        self.transpose_sb = QSpinBox(self.centralwidget)
        self.transpose_sb.setObjectName(u"transpose_sb")
        sizePolicy2.setHeightForWidth(self.transpose_sb.sizePolicy().hasHeightForWidth())
        self.transpose_sb.setSizePolicy(sizePolicy2)
        self.transpose_sb.setMinimumSize(QSize(64, 0))
        self.transpose_sb.setContextMenuPolicy(Qt.NoContextMenu)
        self.transpose_sb.setFrame(False)
        self.transpose_sb.setAlignment(Qt.AlignCenter)
        self.transpose_sb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.transpose_sb.setMinimum(-64)
        self.transpose_sb.setMaximum(64)

        self.transpose_lyt.addWidget(self.transpose_sb)

        self.horizontalSpacer_15 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.transpose_lyt.addItem(self.horizontalSpacer_15)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setSpacing(4)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.pf_mode_l = QLabel(self.centralwidget)
        self.pf_mode_l.setObjectName(u"pf_mode_l")

        self.horizontalLayout.addWidget(self.pf_mode_l)

        self.pf_mode_cmb = QComboBox(self.centralwidget)
        self.pf_mode_cmb.addItem("")
        self.pf_mode_cmb.addItem("")
        self.pf_mode_cmb.addItem("")
        self.pf_mode_cmb.addItem("")
        self.pf_mode_cmb.addItem("")
        self.pf_mode_cmb.addItem("")
        self.pf_mode_cmb.addItem("")
        self.pf_mode_cmb.addItem("")
        self.pf_mode_cmb.setObjectName(u"pf_mode_cmb")
        self.pf_mode_cmb.setFrame(False)

        self.horizontalLayout.addWidget(self.pf_mode_cmb)

        self.pf_th_dsb = QDoubleSpinBox(self.centralwidget)
        self.pf_th_dsb.setObjectName(u"pf_th_dsb")
        self.pf_th_dsb.setEnabled(False)
        self.pf_th_dsb.setFrame(False)
        self.pf_th_dsb.setAlignment(Qt.AlignCenter)
        self.pf_th_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.pf_th_dsb.setDecimals(3)
        self.pf_th_dsb.setMaximum(100.000000000000000)
        self.pf_th_dsb.setValue(5.000000000000000)

        self.horizontalLayout.addWidget(self.pf_th_dsb)


        self.transpose_lyt.addLayout(self.horizontalLayout)

        self.spacer_5 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.transpose_lyt.addItem(self.spacer_5)

        self.tuning_l = QLabel(self.centralwidget)
        self.tuning_l.setObjectName(u"tuning_l")
        sizePolicy2.setHeightForWidth(self.tuning_l.sizePolicy().hasHeightForWidth())
        self.tuning_l.setSizePolicy(sizePolicy2)

        self.transpose_lyt.addWidget(self.tuning_l)

        self.tuning_dsb = QDoubleSpinBox(self.centralwidget)
        self.tuning_dsb.setObjectName(u"tuning_dsb")
        sizePolicy2.setHeightForWidth(self.tuning_dsb.sizePolicy().hasHeightForWidth())
        self.tuning_dsb.setSizePolicy(sizePolicy2)
        self.tuning_dsb.setMinimumSize(QSize(64, 0))
        self.tuning_dsb.setContextMenuPolicy(Qt.NoContextMenu)
        self.tuning_dsb.setFrame(False)
        self.tuning_dsb.setAlignment(Qt.AlignCenter)
        self.tuning_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.tuning_dsb.setDecimals(3)
        self.tuning_dsb.setMinimum(-1.000000000000000)
        self.tuning_dsb.setMaximum(1.000000000000000)
        self.tuning_dsb.setSingleStep(0.010000000000000)

        self.transpose_lyt.addWidget(self.tuning_dsb)

        self.spacer_10 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.transpose_lyt.addItem(self.spacer_10)


        self.centralwidget_lyt.addLayout(self.transpose_lyt)

        self.verticalSpacer_6 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.centralwidget_lyt.addItem(self.verticalSpacer_6)

        self.other_title_l = QLabel(self.centralwidget)
        self.other_title_l.setObjectName(u"other_title_l")
        sizePolicy.setHeightForWidth(self.other_title_l.sizePolicy().hasHeightForWidth())
        self.other_title_l.setSizePolicy(sizePolicy)
        self.other_title_l.setMinimumSize(QSize(64, 0))
        self.other_title_l.setStyleSheet(u"background-color: rgb(63, 95, 127);\n"
"color: rgb(255, 255, 255);")

        self.centralwidget_lyt.addWidget(self.other_title_l)

        self.verticalSpacer_7 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.centralwidget_lyt.addItem(self.verticalSpacer_7)

        self.options_lyt = QHBoxLayout()
        self.options_lyt.setSpacing(4)
        self.options_lyt.setObjectName(u"options_lyt")
        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.options_lyt.addItem(self.horizontalSpacer_2)

        self.attenuation_l = QLabel(self.centralwidget)
        self.attenuation_l.setObjectName(u"attenuation_l")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.attenuation_l.sizePolicy().hasHeightForWidth())
        self.attenuation_l.setSizePolicy(sizePolicy3)
        self.attenuation_l.setMinimumSize(QSize(128, 0))

        self.options_lyt.addWidget(self.attenuation_l)

        self.attenuation_dsb = QDoubleSpinBox(self.centralwidget)
        self.attenuation_dsb.setObjectName(u"attenuation_dsb")
        sizePolicy2.setHeightForWidth(self.attenuation_dsb.sizePolicy().hasHeightForWidth())
        self.attenuation_dsb.setSizePolicy(sizePolicy2)
        self.attenuation_dsb.setMinimumSize(QSize(64, 0))
        self.attenuation_dsb.setFrame(False)
        self.attenuation_dsb.setAlignment(Qt.AlignCenter)
        self.attenuation_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.attenuation_dsb.setDecimals(3)
        self.attenuation_dsb.setMinimum(-12.000000000000000)
        self.attenuation_dsb.setMaximum(0.000000000000000)
        self.attenuation_dsb.setSingleStep(1.000000000000000)
        self.attenuation_dsb.setValue(-6.000000000000000)

        self.options_lyt.addWidget(self.attenuation_dsb)

        self.spacer_13 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.options_lyt.addItem(self.spacer_13)

        self.veltrk_l = QLabel(self.centralwidget)
        self.veltrk_l.setObjectName(u"veltrk_l")
        sizePolicy2.setHeightForWidth(self.veltrk_l.sizePolicy().hasHeightForWidth())
        self.veltrk_l.setSizePolicy(sizePolicy2)

        self.options_lyt.addWidget(self.veltrk_l)

        self.ampveltrk_dsb = QDoubleSpinBox(self.centralwidget)
        self.ampveltrk_dsb.setObjectName(u"ampveltrk_dsb")
        sizePolicy2.setHeightForWidth(self.ampveltrk_dsb.sizePolicy().hasHeightForWidth())
        self.ampveltrk_dsb.setSizePolicy(sizePolicy2)
        self.ampveltrk_dsb.setMinimumSize(QSize(64, 0))
        self.ampveltrk_dsb.setFrame(False)
        self.ampveltrk_dsb.setAlignment(Qt.AlignCenter)
        self.ampveltrk_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.ampveltrk_dsb.setDecimals(3)
        self.ampveltrk_dsb.setMinimum(0.000000000000000)
        self.ampveltrk_dsb.setMaximum(1.000000000000000)
        self.ampveltrk_dsb.setSingleStep(0.100000000000000)
        self.ampveltrk_dsb.setValue(1.000000000000000)

        self.options_lyt.addWidget(self.ampveltrk_dsb)

        self.spacer_8 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.options_lyt.addItem(self.spacer_8)

        self.notepan_l = QLabel(self.centralwidget)
        self.notepan_l.setObjectName(u"notepan_l")
        sizePolicy2.setHeightForWidth(self.notepan_l.sizePolicy().hasHeightForWidth())
        self.notepan_l.setSizePolicy(sizePolicy2)

        self.options_lyt.addWidget(self.notepan_l)

        self.notepan_dsb = QDoubleSpinBox(self.centralwidget)
        self.notepan_dsb.setObjectName(u"notepan_dsb")
        sizePolicy2.setHeightForWidth(self.notepan_dsb.sizePolicy().hasHeightForWidth())
        self.notepan_dsb.setSizePolicy(sizePolicy2)
        self.notepan_dsb.setMinimumSize(QSize(64, 0))
        self.notepan_dsb.setFrame(False)
        self.notepan_dsb.setAlignment(Qt.AlignCenter)
        self.notepan_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.notepan_dsb.setDecimals(3)
        self.notepan_dsb.setMinimum(-1.000000000000000)
        self.notepan_dsb.setMaximum(1.000000000000000)
        self.notepan_dsb.setSingleStep(0.100000000000000)
        self.notepan_dsb.setValue(0.000000000000000)

        self.options_lyt.addWidget(self.notepan_dsb)

        self.spacer = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.options_lyt.addItem(self.spacer)


        self.centralwidget_lyt.addLayout(self.options_lyt)

        self.crossfade_lyt = QHBoxLayout()
        self.crossfade_lyt.setSpacing(4)
        self.crossfade_lyt.setObjectName(u"crossfade_lyt")
        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.crossfade_lyt.addItem(self.horizontalSpacer_6)

        self.crossfade_l = QLabel(self.centralwidget)
        self.crossfade_l.setObjectName(u"crossfade_l")
        sizePolicy3.setHeightForWidth(self.crossfade_l.sizePolicy().hasHeightForWidth())
        self.crossfade_l.setSizePolicy(sizePolicy3)
        self.crossfade_l.setMinimumSize(QSize(128, 0))

        self.crossfade_lyt.addWidget(self.crossfade_l)

        self.crossfade_cmb = QComboBox(self.centralwidget)
        self.crossfade_cmb.addItem("")
        self.crossfade_cmb.addItem("")
        self.crossfade_cmb.addItem("")
        self.crossfade_cmb.setObjectName(u"crossfade_cmb")
        self.crossfade_cmb.setFrame(False)

        self.crossfade_lyt.addWidget(self.crossfade_cmb)

        self.crossfade_dsb = QDoubleSpinBox(self.centralwidget)
        self.crossfade_dsb.setObjectName(u"crossfade_dsb")
        sizePolicy2.setHeightForWidth(self.crossfade_dsb.sizePolicy().hasHeightForWidth())
        self.crossfade_dsb.setSizePolicy(sizePolicy2)
        self.crossfade_dsb.setMinimumSize(QSize(64, 0))
        self.crossfade_dsb.setFrame(False)
        self.crossfade_dsb.setAlignment(Qt.AlignCenter)
        self.crossfade_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.crossfade_dsb.setDecimals(3)
        self.crossfade_dsb.setMinimum(0.000000000000000)
        self.crossfade_dsb.setMaximum(1.000000000000000)
        self.crossfade_dsb.setSingleStep(0.100000000000000)
        self.crossfade_dsb.setValue(0.050000000000000)

        self.crossfade_lyt.addWidget(self.crossfade_dsb)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.crossfade_lyt.addItem(self.horizontalSpacer_3)

        self.monophonic_cb = QCheckBox(self.centralwidget)
        self.monophonic_cb.setObjectName(u"monophonic_cb")

        self.crossfade_lyt.addWidget(self.monophonic_cb)

        self.horizontalSpacer_12 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.crossfade_lyt.addItem(self.horizontalSpacer_12)


        self.centralwidget_lyt.addLayout(self.crossfade_lyt)

        self.verticalSpacer_8 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.centralwidget_lyt.addItem(self.verticalSpacer_8)

        self.instr_ui_title_l = QLabel(self.centralwidget)
        self.instr_ui_title_l.setObjectName(u"instr_ui_title_l")
        sizePolicy.setHeightForWidth(self.instr_ui_title_l.sizePolicy().hasHeightForWidth())
        self.instr_ui_title_l.setSizePolicy(sizePolicy)
        self.instr_ui_title_l.setMinimumSize(QSize(64, 0))
        self.instr_ui_title_l.setStyleSheet(u"background-color: rgb(63, 95, 127);\n"
"color: rgb(255, 255, 255);")

        self.centralwidget_lyt.addWidget(self.instr_ui_title_l)

        self.verticalSpacer_9 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.centralwidget_lyt.addItem(self.verticalSpacer_9)

        self.bg_text_lyt = QHBoxLayout()
        self.bg_text_lyt.setSpacing(0)
        self.bg_text_lyt.setObjectName(u"bg_text_lyt")
        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.bg_text_lyt.addItem(self.horizontalSpacer_5)

        self.bg_text_l = QLabel(self.centralwidget)
        self.bg_text_l.setObjectName(u"bg_text_l")
        self.bg_text_l.setMinimumSize(QSize(128, 0))

        self.bg_text_lyt.addWidget(self.bg_text_l)

        self.bg_text_cmb = QComboBox(self.centralwidget)
        self.bg_text_cmb.addItem("")
        self.bg_text_cmb.addItem("")
        self.bg_text_cmb.addItem("")
        self.bg_text_cmb.setObjectName(u"bg_text_cmb")
        self.bg_text_cmb.setFrame(False)

        self.bg_text_lyt.addWidget(self.bg_text_cmb)

        self.bg_text_le = QLineEdit(self.centralwidget)
        self.bg_text_le.setObjectName(u"bg_text_le")
        self.bg_text_le.setEnabled(False)
        self.bg_text_le.setFrame(False)
        self.bg_text_le.setAlignment(Qt.AlignCenter)

        self.bg_text_lyt.addWidget(self.bg_text_le)

        self.horizontalSpacer_11 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.bg_text_lyt.addItem(self.horizontalSpacer_11)


        self.centralwidget_lyt.addLayout(self.bg_text_lyt)

        self.palette_lyt = QHBoxLayout()
        self.palette_lyt.setSpacing(0)
        self.palette_lyt.setObjectName(u"palette_lyt")
        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.palette_lyt.addItem(self.horizontalSpacer_4)

        self.palette_l = QLabel(self.centralwidget)
        self.palette_l.setObjectName(u"palette_l")
        self.palette_l.setMinimumSize(QSize(128, 0))

        self.palette_lyt.addWidget(self.palette_l)

        self.palette_cmb = QComboBox(self.centralwidget)
        self.palette_cmb.setObjectName(u"palette_cmb")
        self.palette_cmb.setMinimumSize(QSize(160, 0))
        self.palette_cmb.setMaximumSize(QSize(160, 16777215))
        self.palette_cmb.setFrame(False)

        self.palette_lyt.addWidget(self.palette_cmb)

        self.spacer_14 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.palette_lyt.addItem(self.spacer_14)

        self.hsv_pb = QPushButton(self.centralwidget)
        self.hsv_pb.setObjectName(u"hsv_pb")
        self.hsv_pb.setFont(font)

        self.palette_lyt.addWidget(self.hsv_pb)

        self.hue_dsb = QDoubleSpinBox(self.centralwidget)
        self.hue_dsb.setObjectName(u"hue_dsb")
        self.hue_dsb.setMinimumSize(QSize(64, 0))
        self.hue_dsb.setContextMenuPolicy(Qt.NoContextMenu)
        self.hue_dsb.setFrame(False)
        self.hue_dsb.setAlignment(Qt.AlignCenter)
        self.hue_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.hue_dsb.setDecimals(3)
        self.hue_dsb.setMaximum(1.000000000000000)
        self.hue_dsb.setSingleStep(0.100000000000000)
        self.hue_dsb.setValue(0.000000000000000)

        self.palette_lyt.addWidget(self.hue_dsb)

        self.sat_dsb = QDoubleSpinBox(self.centralwidget)
        self.sat_dsb.setObjectName(u"sat_dsb")
        self.sat_dsb.setMinimumSize(QSize(64, 0))
        self.sat_dsb.setContextMenuPolicy(Qt.NoContextMenu)
        self.sat_dsb.setFrame(False)
        self.sat_dsb.setAlignment(Qt.AlignCenter)
        self.sat_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.sat_dsb.setDecimals(3)
        self.sat_dsb.setMaximum(2.000000000000000)
        self.sat_dsb.setSingleStep(0.100000000000000)
        self.sat_dsb.setValue(1.000000000000000)

        self.palette_lyt.addWidget(self.sat_dsb)

        self.val_dsb = QDoubleSpinBox(self.centralwidget)
        self.val_dsb.setObjectName(u"val_dsb")
        self.val_dsb.setMinimumSize(QSize(64, 0))
        self.val_dsb.setContextMenuPolicy(Qt.NoContextMenu)
        self.val_dsb.setFrame(False)
        self.val_dsb.setAlignment(Qt.AlignCenter)
        self.val_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.val_dsb.setDecimals(3)
        self.val_dsb.setMaximum(2.000000000000000)
        self.val_dsb.setSingleStep(0.100000000000000)
        self.val_dsb.setValue(1.000000000000000)

        self.palette_lyt.addWidget(self.val_dsb)

        self.spacer_17 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.palette_lyt.addItem(self.spacer_17)


        self.centralwidget_lyt.addLayout(self.palette_lyt)

        self.grp_ui_lyt = QHBoxLayout()
        self.grp_ui_lyt.setSpacing(0)
        self.grp_ui_lyt.setObjectName(u"grp_ui_lyt")
        self.horizontalSpacer_8 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.grp_ui_lyt.addItem(self.horizontalSpacer_8)

        self.grp_knob_rows_l = QLabel(self.centralwidget)
        self.grp_knob_rows_l.setObjectName(u"grp_knob_rows_l")
        self.grp_knob_rows_l.setMinimumSize(QSize(128, 0))

        self.grp_ui_lyt.addWidget(self.grp_knob_rows_l)

        self.grp_knob_rows_sb = QSpinBox(self.centralwidget)
        self.grp_knob_rows_sb.setObjectName(u"grp_knob_rows_sb")
        self.grp_knob_rows_sb.setMinimumSize(QSize(48, 0))
        self.grp_knob_rows_sb.setContextMenuPolicy(Qt.NoContextMenu)
        self.grp_knob_rows_sb.setFrame(False)
        self.grp_knob_rows_sb.setAlignment(Qt.AlignCenter)
        self.grp_knob_rows_sb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.grp_knob_rows_sb.setMinimum(0)
        self.grp_knob_rows_sb.setMaximum(4)
        self.grp_knob_rows_sb.setValue(1)

        self.grp_ui_lyt.addWidget(self.grp_knob_rows_sb)

        self.horizontalSpacer_34 = QSpacerItem(8, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.grp_ui_lyt.addItem(self.horizontalSpacer_34)

        self.group_mute_wid = QWidget(self.centralwidget)
        self.group_mute_wid.setObjectName(u"group_mute_wid")
        self.horizontalLayout_9 = QHBoxLayout(self.group_mute_wid)
        self.horizontalLayout_9.setSpacing(0)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.group_mute_l = QLabel(self.group_mute_wid)
        self.group_mute_l.setObjectName(u"group_mute_l")

        self.horizontalLayout_9.addWidget(self.group_mute_l)

        self.group_mute_le = QLineEdit(self.group_mute_wid)
        self.group_mute_le.setObjectName(u"group_mute_le")
        self.group_mute_le.setFrame(False)
        self.group_mute_le.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_9.addWidget(self.group_mute_le)


        self.grp_ui_lyt.addWidget(self.group_mute_wid)

        self.horizontalSpacer_35 = QSpacerItem(8, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.grp_ui_lyt.addItem(self.horizontalSpacer_35)

        self.no_solo_grp_knob_cb = QCheckBox(self.centralwidget)
        self.no_solo_grp_knob_cb.setObjectName(u"no_solo_grp_knob_cb")
        self.no_solo_grp_knob_cb.setChecked(True)

        self.grp_ui_lyt.addWidget(self.no_solo_grp_knob_cb)

        self.horizontalSpacer_10 = QSpacerItem(8, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.grp_ui_lyt.addItem(self.horizontalSpacer_10)

        self.adsr_knobs_cb = QCheckBox(self.centralwidget)
        self.adsr_knobs_cb.setObjectName(u"adsr_knobs_cb")
        self.adsr_knobs_cb.setChecked(True)

        self.grp_ui_lyt.addWidget(self.adsr_knobs_cb)

        self.max_adsr_value_l = QLabel(self.centralwidget)
        self.max_adsr_value_l.setObjectName(u"max_adsr_value_l")

        self.grp_ui_lyt.addWidget(self.max_adsr_value_l)

        self.max_adsr_dsb = QDoubleSpinBox(self.centralwidget)
        self.max_adsr_dsb.setObjectName(u"max_adsr_dsb")
        self.max_adsr_dsb.setMinimumSize(QSize(64, 0))
        self.max_adsr_dsb.setContextMenuPolicy(Qt.NoContextMenu)
        self.max_adsr_dsb.setFrame(False)
        self.max_adsr_dsb.setAlignment(Qt.AlignCenter)
        self.max_adsr_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.max_adsr_dsb.setDecimals(3)
        self.max_adsr_dsb.setMaximum(100.000000000000000)
        self.max_adsr_dsb.setSingleStep(0.100000000000000)
        self.max_adsr_dsb.setValue(20.000000000000000)

        self.grp_ui_lyt.addWidget(self.max_adsr_dsb)

        self.horizontalSpacer_9 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.grp_ui_lyt.addItem(self.horizontalSpacer_9)


        self.centralwidget_lyt.addLayout(self.grp_ui_lyt)

        self.reverb_lyt = QHBoxLayout()
        self.reverb_lyt.setSpacing(0)
        self.reverb_lyt.setObjectName(u"reverb_lyt")
        self.spacer_20 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.reverb_lyt.addItem(self.spacer_20)

        self.use_cutoff_cb = QCheckBox(self.centralwidget)
        self.use_cutoff_cb.setObjectName(u"use_cutoff_cb")
        self.use_cutoff_cb.setMinimumSize(QSize(128, 0))
        self.use_cutoff_cb.setChecked(True)

        self.reverb_lyt.addWidget(self.use_cutoff_cb)

        self.use_reverb_cb = QCheckBox(self.centralwidget)
        self.use_reverb_cb.setObjectName(u"use_reverb_cb")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.use_reverb_cb.sizePolicy().hasHeightForWidth())
        self.use_reverb_cb.setSizePolicy(sizePolicy4)
        self.use_reverb_cb.setChecked(True)

        self.reverb_lyt.addWidget(self.use_reverb_cb)

        self.reverb_wet_dsb = QDoubleSpinBox(self.centralwidget)
        self.reverb_wet_dsb.setObjectName(u"reverb_wet_dsb")
        sizePolicy2.setHeightForWidth(self.reverb_wet_dsb.sizePolicy().hasHeightForWidth())
        self.reverb_wet_dsb.setSizePolicy(sizePolicy2)
        self.reverb_wet_dsb.setMinimumSize(QSize(64, 0))
        self.reverb_wet_dsb.setFrame(False)
        self.reverb_wet_dsb.setAlignment(Qt.AlignCenter)
        self.reverb_wet_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.reverb_wet_dsb.setDecimals(3)
        self.reverb_wet_dsb.setMinimum(0.000000000000000)
        self.reverb_wet_dsb.setMaximum(1.000000000000000)
        self.reverb_wet_dsb.setSingleStep(0.100000000000000)
        self.reverb_wet_dsb.setValue(0.000000000000000)

        self.reverb_lyt.addWidget(self.reverb_wet_dsb)

        self.horizontalSpacer_33 = QSpacerItem(8, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.reverb_lyt.addItem(self.horizontalSpacer_33)

        self.use_ir_cb = QCheckBox(self.centralwidget)
        self.use_ir_cb.setObjectName(u"use_ir_cb")

        self.reverb_lyt.addWidget(self.use_ir_cb)

        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.reverb_lyt.addItem(self.horizontalSpacer_7)

        self.multi_out_cb = QCheckBox(self.centralwidget)
        self.multi_out_cb.setObjectName(u"multi_out_cb")

        self.reverb_lyt.addWidget(self.multi_out_cb)

        self.horizontalSpacer_32 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.reverb_lyt.addItem(self.horizontalSpacer_32)

        self.hd_knobs_cb = QCheckBox(self.centralwidget)
        self.hd_knobs_cb.setObjectName(u"hd_knobs_cb")

        self.reverb_lyt.addWidget(self.hd_knobs_cb)

        self.estimate_delay_cb = QCheckBox(self.centralwidget)
        self.estimate_delay_cb.setObjectName(u"estimate_delay_cb")

        self.reverb_lyt.addWidget(self.estimate_delay_cb)

        self.horizontalSpacer_31 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.reverb_lyt.addItem(self.horizontalSpacer_31)


        self.centralwidget_lyt.addLayout(self.reverb_lyt)

        self.verticalSpacer_10 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.centralwidget_lyt.addItem(self.verticalSpacer_10)

        self.line = QFrame(self.centralwidget)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)

        self.centralwidget_lyt.addWidget(self.line)

        self.create_button_lyt = QHBoxLayout()
        self.create_button_lyt.setSpacing(8)
        self.create_button_lyt.setObjectName(u"create_button_lyt")
        self.horizontalSpacer_29 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.create_button_lyt.addItem(self.horizontalSpacer_29)

        self.dsp_lyt = QVBoxLayout()
        self.dsp_lyt.setSpacing(4)
        self.dsp_lyt.setObjectName(u"dsp_lyt")
        self.suffix_lyt = QHBoxLayout()
        self.suffix_lyt.setObjectName(u"suffix_lyt")
        self.add_suffix_cb = QCheckBox(self.centralwidget)
        self.add_suffix_cb.setObjectName(u"add_suffix_cb")

        self.suffix_lyt.addWidget(self.add_suffix_cb)

        self.suffix_le = QLineEdit(self.centralwidget)
        self.suffix_le.setObjectName(u"suffix_le")
        self.suffix_le.setEnabled(False)
        sizePolicy.setHeightForWidth(self.suffix_le.sizePolicy().hasHeightForWidth())
        self.suffix_le.setSizePolicy(sizePolicy)
        self.suffix_le.setFrame(False)

        self.suffix_lyt.addWidget(self.suffix_le)


        self.dsp_lyt.addLayout(self.suffix_lyt)

        self.auto_incr_cb = QCheckBox(self.centralwidget)
        self.auto_incr_cb.setObjectName(u"auto_incr_cb")

        self.dsp_lyt.addWidget(self.auto_incr_cb)

        self.preset_fmt_lyt = QHBoxLayout()
        self.preset_fmt_lyt.setSpacing(0)
        self.preset_fmt_lyt.setObjectName(u"preset_fmt_lyt")
        self.preset_fmt_l = QLabel(self.centralwidget)
        self.preset_fmt_l.setObjectName(u"preset_fmt_l")
        self.preset_fmt_l.setMinimumSize(QSize(96, 0))
        self.preset_fmt_l.setMaximumSize(QSize(96, 16777215))

        self.preset_fmt_lyt.addWidget(self.preset_fmt_l)

        self.preset_fmt_cmb = QComboBox(self.centralwidget)
        self.preset_fmt_cmb.addItem("")
        self.preset_fmt_cmb.addItem("")
        self.preset_fmt_cmb.addItem("")
        self.preset_fmt_cmb.setObjectName(u"preset_fmt_cmb")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.preset_fmt_cmb.sizePolicy().hasHeightForWidth())
        self.preset_fmt_cmb.setSizePolicy(sizePolicy5)
        self.preset_fmt_cmb.setMinimumSize(QSize(80, 0))
        self.preset_fmt_cmb.setMaximumSize(QSize(80, 16777215))
        self.preset_fmt_cmb.setFrame(False)

        self.preset_fmt_lyt.addWidget(self.preset_fmt_cmb)

        self.sfz_engine_wid = QWidget(self.centralwidget)
        self.sfz_engine_wid.setObjectName(u"sfz_engine_wid")
        self.horizontalLayout_7 = QHBoxLayout(self.sfz_engine_wid)
        self.horizontalLayout_7.setSpacing(0)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.sfz_engine_l = QLabel(self.sfz_engine_wid)
        self.sfz_engine_l.setObjectName(u"sfz_engine_l")
        self.sfz_engine_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_7.addWidget(self.sfz_engine_l)

        self.sfz_engine_cmb = QComboBox(self.sfz_engine_wid)
        self.sfz_engine_cmb.addItem("")
        self.sfz_engine_cmb.addItem("")
        self.sfz_engine_cmb.addItem("")
        self.sfz_engine_cmb.setObjectName(u"sfz_engine_cmb")
        sizePolicy5.setHeightForWidth(self.sfz_engine_cmb.sizePolicy().hasHeightForWidth())
        self.sfz_engine_cmb.setSizePolicy(sizePolicy5)
        self.sfz_engine_cmb.setFrame(False)

        self.horizontalLayout_7.addWidget(self.sfz_engine_cmb)


        self.preset_fmt_lyt.addWidget(self.sfz_engine_wid)


        self.dsp_lyt.addLayout(self.preset_fmt_lyt)

        self.sfz_options_wid = QWidget(self.centralwidget)
        self.sfz_options_wid.setObjectName(u"sfz_options_wid")
        self.horizontalLayout_8 = QHBoxLayout(self.sfz_options_wid)
        self.horizontalLayout_8.setSpacing(0)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.use_eg_l = QLabel(self.sfz_options_wid)
        self.use_eg_l.setObjectName(u"use_eg_l")
        self.use_eg_l.setMinimumSize(QSize(96, 0))
        self.use_eg_l.setMaximumSize(QSize(96, 16777215))

        self.horizontalLayout_8.addWidget(self.use_eg_l)

        self.use_eg_cmb = QComboBox(self.sfz_options_wid)
        self.use_eg_cmb.addItem("")
        self.use_eg_cmb.addItem("")
        self.use_eg_cmb.addItem("")
        self.use_eg_cmb.setObjectName(u"use_eg_cmb")
        sizePolicy5.setHeightForWidth(self.use_eg_cmb.sizePolicy().hasHeightForWidth())
        self.use_eg_cmb.setSizePolicy(sizePolicy5)
        self.use_eg_cmb.setMinimumSize(QSize(80, 0))
        self.use_eg_cmb.setMaximumSize(QSize(80, 16777215))
        self.use_eg_cmb.setFrame(False)

        self.horizontalLayout_8.addWidget(self.use_eg_cmb)

        self.release_off_by_attack_cb = QCheckBox(self.sfz_options_wid)
        self.release_off_by_attack_cb.setObjectName(u"release_off_by_attack_cb")
        self.release_off_by_attack_cb.setLayoutDirection(Qt.RightToLeft)

        self.horizontalLayout_8.addWidget(self.release_off_by_attack_cb)


        self.dsp_lyt.addWidget(self.sfz_options_wid)

        self.create_preset_pb = QPushButton(self.centralwidget)
        self.create_preset_pb.setObjectName(u"create_preset_pb")
        sizePolicy.setHeightForWidth(self.create_preset_pb.sizePolicy().hasHeightForWidth())
        self.create_preset_pb.setSizePolicy(sizePolicy)
        self.create_preset_pb.setFont(font)
        self.create_preset_pb.setStyleSheet(u"QPushButton{color: rgb(0, 0, 0);\n"
"background-color: rgb(128, 220, 213);}")

        self.dsp_lyt.addWidget(self.create_preset_pb)


        self.create_button_lyt.addLayout(self.dsp_lyt)

        self.dslib_lyt = QVBoxLayout()
        self.dslib_lyt.setSpacing(4)
        self.dslib_lyt.setObjectName(u"dslib_lyt")
        self.verticalSpacer_13 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.dslib_lyt.addItem(self.verticalSpacer_13)

        self.lossy_flac_lyt = QHBoxLayout()
        self.lossy_flac_lyt.setObjectName(u"lossy_flac_lyt")
        self.use_lossy_flac_cb = QCheckBox(self.centralwidget)
        self.use_lossy_flac_cb.setObjectName(u"use_lossy_flac_cb")
        self.use_lossy_flac_cb.setLayoutDirection(Qt.RightToLeft)

        self.lossy_flac_lyt.addWidget(self.use_lossy_flac_cb)

        self.lossy_flac_mode_cmb = QComboBox(self.centralwidget)
        self.lossy_flac_mode_cmb.addItem("")
        self.lossy_flac_mode_cmb.addItem("")
        self.lossy_flac_mode_cmb.addItem("")
        self.lossy_flac_mode_cmb.setObjectName(u"lossy_flac_mode_cmb")
        self.lossy_flac_mode_cmb.setEnabled(False)
        self.lossy_flac_mode_cmb.setFrame(False)

        self.lossy_flac_lyt.addWidget(self.lossy_flac_mode_cmb)


        self.dslib_lyt.addLayout(self.lossy_flac_lyt)

        self.create_dslib_pb = QPushButton(self.centralwidget)
        self.create_dslib_pb.setObjectName(u"create_dslib_pb")
        sizePolicy.setHeightForWidth(self.create_dslib_pb.sizePolicy().hasHeightForWidth())
        self.create_dslib_pb.setSizePolicy(sizePolicy)
        self.create_dslib_pb.setFont(font)

        self.dslib_lyt.addWidget(self.create_dslib_pb)


        self.create_button_lyt.addLayout(self.dslib_lyt)

        self.horizontalSpacer_30 = QSpacerItem(40, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.create_button_lyt.addItem(self.horizontalSpacer_30)


        self.centralwidget_lyt.addLayout(self.create_button_lyt)

        self.progress_pb = QProgressBar(self.centralwidget)
        self.progress_pb.setObjectName(u"progress_pb")
        self.progress_pb.setStyleSheet(u"QProgressBar{border: none;}")
        self.progress_pb.setValue(0)
        self.progress_pb.setTextVisible(False)
        self.progress_pb.setFormat(u"")

        self.centralwidget_lyt.addWidget(self.progress_pb)

        smp_to_ds_ui.setCentralWidget(self.centralwidget)

        self.retranslateUi(smp_to_ds_ui)

        self.groupnaming_cmb.setCurrentIndex(1)
        self.spread_cmb.setCurrentIndex(1)
        self.seq_mode_cmb.setCurrentIndex(1)
        self.note_limit_cmb.setCurrentIndex(0)
        self.pf_mode_cmb.setCurrentIndex(0)
        self.crossfade_cmb.setCurrentIndex(1)
        self.bg_text_cmb.setCurrentIndex(1)
        self.preset_fmt_cmb.setCurrentIndex(2)
        self.use_eg_cmb.setCurrentIndex(2)
        self.lossy_flac_mode_cmb.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(smp_to_ds_ui)
    # setupUi

    def retranslateUi(self, smp_to_ds_ui):
        smp_to_ds_ui.setWindowTitle(QCoreApplication.translate("smp_to_ds_ui", u"SMP2ds", None))
        self.actionSettings.setText(QCoreApplication.translate("smp_to_ds_ui", u"Settings", None))
        self.save_settings_a.setText(QCoreApplication.translate("smp_to_ds_ui", u"Save settings", None))
        self.load_settings_a.setText(QCoreApplication.translate("smp_to_ds_ui", u"Load settings", None))
        self.restore_defaults_a.setText(QCoreApplication.translate("smp_to_ds_ui", u"Restore defaults", None))
#if QT_CONFIG(tooltip)
        self.output_path_l.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Preset ROOT directory path\n"
"\n"
"Samples must be located in a subdirectory called 'Samples' (case sensitive)\n"
"Samples are retrieved recursively so they may be located in subdirectories inside 'Samples'\n"
"\n"
"IR samples must be located in a separate subdirectory called 'IR'", None))
#endif // QT_CONFIG(tooltip)
        self.output_path_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Drag and drop a preset ROOT directory on the window", None))
#if QT_CONFIG(tooltip)
        self.output_path_tb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Browse preset root directory", None))
#endif // QT_CONFIG(tooltip)
        self.output_path_tb.setText(QCoreApplication.translate("smp_to_ds_ui", u"...", None))
        self.sample_title_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Sample info", None))
#if QT_CONFIG(tooltip)
        self.pattern_pb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Pattern used to convert sample names to attribute values\n"
"Click for context menu and some pattern examples", None))
#endif // QT_CONFIG(tooltip)
        self.pattern_pb.setText(QCoreApplication.translate("smp_to_ds_ui", u"Pattern", None))
#if QT_CONFIG(tooltip)
        self.pattern_le.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Pattern used to convert sample names to attribute values\n"
"\n"
"Attribute names must be enclosed with curly braces {}\n"
"Files have to be named accordingly\n"
"\n"
"Tags matching Decent Sampler specifiation are also supported for flac\n"
"when file naming feels impractical", None))
#endif // QT_CONFIG(tooltip)
        self.pattern_le.setText("")
        self.pattern_le.setPlaceholderText(QCoreApplication.translate("smp_to_ds_ui", u"Click Pattern for examples, Right click to insert an attribute", None))
        self.grpnaming_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Group Naming", None))
        self.groupnaming_cmb.setItemText(0, QCoreApplication.translate("smp_to_ds_ui", u"keep", None))
        self.groupnaming_cmb.setItemText(1, QCoreApplication.translate("smp_to_ds_ui", u"beautify", None))
        self.groupnaming_cmb.setItemText(2, QCoreApplication.translate("smp_to_ds_ui", u"upper", None))
        self.groupnaming_cmb.setItemText(3, QCoreApplication.translate("smp_to_ds_ui", u"lower", None))
        self.groupnaming_cmb.setItemText(4, QCoreApplication.translate("smp_to_ds_ui", u"shorten", None))

#if QT_CONFIG(tooltip)
        self.groupnaming_cmb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Naming style for group sliders\n"
"\n"
"keep	directly use {group} attribute as is\n"
"beautify	Capitalize the names and insert spaces by detecting CamelCase and digits\n"
"lower	lowercase\n"
"upper	ALL CAPS :)\n"
"shorten	Abreviate the names keeping only first letters by detecting CamelCase and digits", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.override_cb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Override embedded note metadata with info from sample name\n"
"Ignores per sample tuning/pitch fraction", None))
#endif // QT_CONFIG(tooltip)
        self.override_cb.setText(QCoreApplication.translate("smp_to_ds_ui", u"Override", None))
#if QT_CONFIG(tooltip)
        self.loop_cb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Enable sample loop reading if present (default) or explicitely ignore it", None))
#endif // QT_CONFIG(tooltip)
        self.loop_cb.setText(QCoreApplication.translate("smp_to_ds_ui", u"Loop", None))
        self.env_title_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Envelope", None))
#if QT_CONFIG(tooltip)
        self.adsr_enable_cb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Enable amplitude envelope (default)\n"
"Samples are played entirely if unchecked\n"
"\n"
"NOTE: envelope won't be turned off for looped samples so they don't play forever", None))
#endif // QT_CONFIG(tooltip)
        self.adsr_enable_cb.setText(QCoreApplication.translate("smp_to_ds_ui", u"Enable", None))
#if QT_CONFIG(tooltip)
        self.ADSR_pb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"ADSR envelope (in s, except sustain)\n"
"Click for context menu", None))
#endif // QT_CONFIG(tooltip)
        self.ADSR_pb.setText(QCoreApplication.translate("smp_to_ds_ui", u"ADSR", None))
#if QT_CONFIG(tooltip)
        self.A_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Attack in s", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.D_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Decay in s", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.S_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Sustain amplitude", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.R_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Release in s", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.ADRr_pb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"ADR Curve (-100 log, 0 lin, 100 exp)\n"
"Click for context menu", None))
#endif // QT_CONFIG(tooltip)
        self.ADRr_pb.setText(QCoreApplication.translate("smp_to_ds_ui", u"ADR Release", None))
#if QT_CONFIG(tooltip)
        self.Ar_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Attack Curve\n"
"(-100 log, 0 lin, 100 exp)", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.Dr_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Decay Curve\n"
"(-100 log, 0 lin, 100 exp)", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.Rr_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Release Curve\n"
"(-100 log, 0 lin, 100 exp)", None))
#endif // QT_CONFIG(tooltip)
        self.fake_trigger_title_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Fake Trigger", None))
#if QT_CONFIG(tooltip)
        self.fake_leg_cb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Create a fake legato trigger by re-using attack samples with a trimmed start and a different attack envelope\n"
"\n"
"Typical result: when playing overlapping notes, the first note will mute and the next note will have its attack cut to blend it with the previous one simulating a legato articulation\n"
"This will make the instrument monophonic and unable to play chords but, it is adequate for a solo part like a melody\n"
"Great for wind or bowed instruments...", None))
#endif // QT_CONFIG(tooltip)
        self.fake_leg_cb.setText(QCoreApplication.translate("smp_to_ds_ui", u"Add Fake Legato", None))
        self.fk_leg_start_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Start", None))
#if QT_CONFIG(tooltip)
        self.fk_leg_start_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Start of legato samples in s\n"
"\n"
"This is the lenth cut from the attack", None))
#endif // QT_CONFIG(tooltip)
        self.fk_leg_a_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Attack", None))
#if QT_CONFIG(tooltip)
        self.fk_leg_a_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Attack in s\n"
"\n"
"This is the fade-in length for the legato, you should also modify the release of the attack ADSR settings so both work together", None))
#endif // QT_CONFIG(tooltip)
        self.fk_leg_a_curve_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Attack Curve", None))
#if QT_CONFIG(tooltip)
        self.fk_leg_a_curve_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Attack Curve\n"
"(-100 log, 0 lin, 100 exp)", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.fake_rls_cb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Create fake release trigger by re-using attack samples with a different envelope and some tweaks\n"
"\n"
"Typical result: a much shorter attenuated version of the samples with a heavy low pass filter will play when releasing keys\n"
"Great for keys, guitars...", None))
#endif // QT_CONFIG(tooltip)
        self.fake_rls_cb.setText(QCoreApplication.translate("smp_to_ds_ui", u"Add Fake Release", None))
        self.fk_rls_mode_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Mode", None))
        self.fk_rls_mode_cmb.setItemText(0, QCoreApplication.translate("smp_to_ds_ui", u"start", None))
        self.fk_rls_mode_cmb.setItemText(1, QCoreApplication.translate("smp_to_ds_ui", u"loop_end", None))
        self.fk_rls_mode_cmb.setItemText(2, QCoreApplication.translate("smp_to_ds_ui", u"cue", None))

#if QT_CONFIG(tooltip)
        self.fk_rls_mode_cmb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Fake release mode\n"
"\n"
"start	Direct reuse of attack sample (default)\n"
"loop_end	Set start at loop_end (samples with tail after loop)\n"
"cue	Set start at cue marker", None))
#endif // QT_CONFIG(tooltip)
        self.fk_volume_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Volume", None))
#if QT_CONFIG(tooltip)
        self.fk_volume_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Fake Release Attenuation (dB)\n"
"\n"
"In many cases, the release sounds should be attenuated", None))
#endif // QT_CONFIG(tooltip)
        self.fk_tuning_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Tuning", None))
#if QT_CONFIG(tooltip)
        self.fk_tuning_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Fake Release Tuning (semitones)\n"
"\n"
"Detune the release sounds from their original pitch", None))
#endif // QT_CONFIG(tooltip)
        self.fk_cutoff_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Cutoff", None))
#if QT_CONFIG(tooltip)
        self.fk_cutoff_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Fake Release Low Pass Filter (Hz)\n"
"\n"
"In many case, release sounds sound \"muffled\", this can be simulated here with a simple filter", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.fk_ADSR_pb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Fake release ADSR envelope (in s)\n"
"Click for context menu", None))
#endif // QT_CONFIG(tooltip)
        self.fk_ADSR_pb.setText(QCoreApplication.translate("smp_to_ds_ui", u"Fake ADSR", None))
#if QT_CONFIG(tooltip)
        self.fk_A_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Attack in s", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.fk_D_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Decay in s", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.fk_S_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Sustain amplitude", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.fk_R_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Release in s", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.fk_ADRr_pb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Fake release ADR Curve (-100 log, 0 lin, 100 exp)\n"
"Click for context menu", None))
#endif // QT_CONFIG(tooltip)
        self.fk_ADRr_pb.setText(QCoreApplication.translate("smp_to_ds_ui", u"Fake ADR Release", None))
#if QT_CONFIG(tooltip)
        self.fk_Ar_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Attack Curve\n"
"(-100 log, 0 lin, 100 exp)", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.fk_Dr_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Decay Curve\n"
"(-100 log, 0 lin, 100 exp)", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.fk_Rr_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Release Curve\n"
"(-100 log, 0 lin, 100 exp)", None))
#endif // QT_CONFIG(tooltip)
        self.mapping_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Mapping", None))
        self.notespread_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Note Spread", None))
        self.spread_cmb.setItemText(0, QCoreApplication.translate("smp_to_ds_ui", u"up", None))
        self.spread_cmb.setItemText(1, QCoreApplication.translate("smp_to_ds_ui", u"mid", None))
        self.spread_cmb.setItemText(2, QCoreApplication.translate("smp_to_ds_ui", u"down", None))
        self.spread_cmb.setItemText(3, QCoreApplication.translate("smp_to_ds_ui", u"none", None))

#if QT_CONFIG(tooltip)
        self.spread_cmb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Note extension mode in relation with other notes\n"
"\n"
"up	extend to higher note\n"
"mid	(default) extend both ways\n"
"down	extend to lower note\n"
"none	 No extension (drums,sfx)", None))
#endif // QT_CONFIG(tooltip)
        self.spread_cmb.setCurrentText(QCoreApplication.translate("smp_to_ds_ui", u"mid", None))
#if QT_CONFIG(tooltip)
        self.pad_vel_cb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Re-use / clone samples from higher velocity layer to velocity layers with fewer notes\n"
"\n"
"Typical usage: instruments with velocity layers having a very different number of note samples\n"
"\n"
"- Avoid over-stretching zones in some circumstances\n"
"- May help with fake round-robins on lower velocities", None))
#endif // QT_CONFIG(tooltip)
        self.pad_vel_cb.setText(QCoreApplication.translate("smp_to_ds_ui", u"Pad Vel Layers", None))
        self.seqmode_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Seq Mode", None))
        self.seq_mode_cmb.setItemText(0, QCoreApplication.translate("smp_to_ds_ui", u"round_robin", None))
        self.seq_mode_cmb.setItemText(1, QCoreApplication.translate("smp_to_ds_ui", u"random", None))
        self.seq_mode_cmb.setItemText(2, QCoreApplication.translate("smp_to_ds_ui", u"true_random", None))
        self.seq_mode_cmb.setItemText(3, QCoreApplication.translate("smp_to_ds_ui", u"always", None))

#if QT_CONFIG(tooltip)
        self.seq_mode_cmb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Round-Robin mode", None))
#endif // QT_CONFIG(tooltip)
        self.seq_mode_cmb.setCurrentText(QCoreApplication.translate("smp_to_ds_ui", u"random", None))
#if QT_CONFIG(tooltip)
        self.limit_pb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Limit note range to bounding samples or extend bounds\n"
"Click for context menu and some examples", None))
#endif // QT_CONFIG(tooltip)
        self.limit_pb.setText(QCoreApplication.translate("smp_to_ds_ui", u"Limit", None))
#if QT_CONFIG(tooltip)
        self.limit_le.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Limit note range to bounding samples or extend bounds\n"
"\n"
"Enter 2 integer values separated by a space\n"
"Use - or + to offset from bounding notes\n"
"\n"
"'auto' sets bounds from average note gap\n"
"'autox2' doubles this gap", None))
#endif // QT_CONFIG(tooltip)
        self.limit_le.setText("")
        self.limit_le.setPlaceholderText(QCoreApplication.translate("smp_to_ds_ui", u"MIDI note range", None))
        self.note_limit_cmb.setItemText(0, QCoreApplication.translate("smp_to_ds_ui", u"shared", None))
        self.note_limit_cmb.setItemText(1, QCoreApplication.translate("smp_to_ds_ui", u"group", None))

#if QT_CONFIG(tooltip)
        self.note_limit_cmb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Note limit mode\n"
"\n"
"shared	use all notes among all groups (default)\n"
"group	consider only notes within the same group (typically organs with one group per stop)", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.fake_rr_pb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Fake Round-Robin offsets\n"
"Click for context menu and some examples", None))
#endif // QT_CONFIG(tooltip)
        self.fake_rr_pb.setText(QCoreApplication.translate("smp_to_ds_ui", u"Fake RR", None))
#if QT_CONFIG(tooltip)
        self.rrofs_le.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Fake Round-Robin offsets\n"
"\n"
"Avoid the \"machine-gun effect\" plaguing a lot of sampled instruments\n"
"\n"
"Borrow samples from neighboring zones to emulate Round-Robins\n"
"Enter integer values (sample index offset) separated by a space\n"
"A sufficient number of note samples is required for better results\n"
"\n"
"Leave a blank space to disable\n"
"\n"
"'x3' and 'x5' for respectively 3 and 5 fake Round-Robins", None))
#endif // QT_CONFIG(tooltip)
        self.rrofs_le.setText("")
        self.rrofs_le.setPlaceholderText(QCoreApplication.translate("smp_to_ds_ui", u"xNUMBER", None))
#if QT_CONFIG(tooltip)
        self.rr_bounds_cb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Make fake RR robin work even with bounding notes\n"
"by stealing notes further in reverse when reaching bounds\n"
"\n"
"If bounding notes sound too unnatural/pitched, disable the option", None))
#endif // QT_CONFIG(tooltip)
        self.rr_bounds_cb.setText(QCoreApplication.translate("smp_to_ds_ui", u"RR Bounds", None))
        self.transpose_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Transpose", None))
#if QT_CONFIG(tooltip)
        self.transpose_sb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Transpose the whole sample mapping in semitones", None))
#endif // QT_CONFIG(tooltip)
        self.pf_mode_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Pitch Fraction Mode", None))
        self.pf_mode_cmb.setItemText(0, QCoreApplication.translate("smp_to_ds_ui", u"off", None))
        self.pf_mode_cmb.setItemText(1, QCoreApplication.translate("smp_to_ds_ui", u"on", None))
        self.pf_mode_cmb.setItemText(2, QCoreApplication.translate("smp_to_ds_ui", u"mean_scale", None))
        self.pf_mode_cmb.setItemText(3, QCoreApplication.translate("smp_to_ds_ui", u"mean_blend", None))
        self.pf_mode_cmb.setItemText(4, QCoreApplication.translate("smp_to_ds_ui", u"mean_threshold", None))
        self.pf_mode_cmb.setItemText(5, QCoreApplication.translate("smp_to_ds_ui", u"on_rand", None))
        self.pf_mode_cmb.setItemText(6, QCoreApplication.translate("smp_to_ds_ui", u"on_threshold", None))
        self.pf_mode_cmb.setItemText(7, QCoreApplication.translate("smp_to_ds_ui", u"on_threshold_rand", None))

#if QT_CONFIG(tooltip)
        self.pf_mode_cmb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Control pitch fraction usage\n"
"Keep pitch fraction \"in check\" \n"
"\n"
"off	Ignore pitch fraction\n"
"on	Use pitch fraction\n"
"mean_scale	Scale deviation from mean to given value\n"
"mean_blend	Blend between mean and pitch fraction (0-100)\n"
"mean_threshold	Use  pitch fraction over threshold and mean under threshold\n"
"on_rand	Use and add given random\n"
"on_threshold	Use over given threshold\n"
"on_threshold_rand	Use over given threshold and add random", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.pf_th_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Pitch fraction threshold/random factor in semitone cents", None))
#endif // QT_CONFIG(tooltip)
        self.tuning_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Tuning", None))
#if QT_CONFIG(tooltip)
        self.tuning_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Fine tuning in semitones", None))
#endif // QT_CONFIG(tooltip)
        self.other_title_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Other options", None))
        self.attenuation_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Attenuation", None))
#if QT_CONFIG(tooltip)
        self.attenuation_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Volume attenuation to prevent saturation/clipping", None))
#endif // QT_CONFIG(tooltip)
        self.veltrk_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Velocity Track", None))
#if QT_CONFIG(tooltip)
        self.ampveltrk_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Velocity Tracking Amplitude\n"
"Right click for context menu", None))
#endif // QT_CONFIG(tooltip)
        self.notepan_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Note Pan", None))
#if QT_CONFIG(tooltip)
        self.notepan_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Auto pan sample from left to right depending on note number\n"
"Use negative value to invert the effect (right to left)\n"
"Right click for context menu", None))
#endif // QT_CONFIG(tooltip)
        self.crossfade_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Crossfade", None))
        self.crossfade_cmb.setItemText(0, QCoreApplication.translate("smp_to_ds_ui", u"off", None))
        self.crossfade_cmb.setItemText(1, QCoreApplication.translate("smp_to_ds_ui", u"linear", None))
        self.crossfade_cmb.setItemText(2, QCoreApplication.translate("smp_to_ds_ui", u"equal_power", None))

#if QT_CONFIG(tooltip)
        self.crossfade_cmb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Crossfade Mode", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.crossfade_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Crossfade length as a percentage of loop length", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.monophonic_cb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Make instrument monophonic at group level\n"
"(Bass, Wind instrument...)\n"
"When using legatos, the instrument will already be de facto monophonic", None))
#endif // QT_CONFIG(tooltip)
        self.monophonic_cb.setText(QCoreApplication.translate("smp_to_ds_ui", u"Monophonic", None))
        self.instr_ui_title_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Instrument UI", None))
        self.bg_text_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Background Text", None))
        self.bg_text_cmb.setItemText(0, QCoreApplication.translate("smp_to_ds_ui", u"none", None))
        self.bg_text_cmb.setItemText(1, QCoreApplication.translate("smp_to_ds_ui", u"root_dir", None))
        self.bg_text_cmb.setItemText(2, QCoreApplication.translate("smp_to_ds_ui", u"custom", None))

#if QT_CONFIG(tooltip)
        self.bg_text_cmb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Write text to the background image\n"
"none	No text\n"
"rootdir	Use root directory name as text (default)\n"
"custom	Type a custom text", None))
#endif // QT_CONFIG(tooltip)
        self.bg_text_le.setPlaceholderText(QCoreApplication.translate("smp_to_ds_ui", u"Instrument Name", None))
        self.palette_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Palette", None))
#if QT_CONFIG(tooltip)
        self.palette_cmb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Color Palette used to generate the DS Preset UI and its background image", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.hsv_pb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Globally adjust chosen color palette", None))
#endif // QT_CONFIG(tooltip)
        self.hsv_pb.setText(QCoreApplication.translate("smp_to_ds_ui", u"HSV Adjust", None))
#if QT_CONFIG(tooltip)
        self.hue_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Hue Shift", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.sat_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Saturation multiplier", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.val_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Value multiplier", None))
#endif // QT_CONFIG(tooltip)
        self.grp_knob_rows_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Group Knob Rows", None))
#if QT_CONFIG(tooltip)
        self.grp_knob_rows_sb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Number of group knob rows for generated UI\n"
"0 willl disable all group knobs", None))
#endif // QT_CONFIG(tooltip)
        self.group_mute_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Mute", None))
#if QT_CONFIG(tooltip)
        self.group_mute_le.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Mute buttons default state\n"
"Enter a binary sequence of 0 (muted) and 1 (enabled)\n"
"\n"
"The last value is repeated for the remaining buttons\n"
"\n"
"Right-click for examples", None))
#endif // QT_CONFIG(tooltip)
        self.group_mute_le.setText("")
        self.group_mute_le.setPlaceholderText(QCoreApplication.translate("smp_to_ds_ui", u"Binary Sequence", None))
#if QT_CONFIG(tooltip)
        self.no_solo_grp_knob_cb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Don't create volume knob for instrument with only one group", None))
#endif // QT_CONFIG(tooltip)
        self.no_solo_grp_knob_cb.setText(QCoreApplication.translate("smp_to_ds_ui", u"No Solo Group Knob", None))
#if QT_CONFIG(tooltip)
        self.adsr_knobs_cb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Create ADSR knobs in UI", None))
#endif // QT_CONFIG(tooltip)
        self.adsr_knobs_cb.setText(QCoreApplication.translate("smp_to_ds_ui", u"ADSR Knobs", None))
        self.max_adsr_value_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Max Value", None))
#if QT_CONFIG(tooltip)
        self.max_adsr_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Max ADR value for ADSR knobs in UI", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.use_cutoff_cb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Create cutoff filter (LPF) as a modultation knob (CC1)", None))
#endif // QT_CONFIG(tooltip)
        self.use_cutoff_cb.setText(QCoreApplication.translate("smp_to_ds_ui", u"Cutoff", None))
#if QT_CONFIG(tooltip)
        self.use_reverb_cb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Create reverb knob (CC91)", None))
#endif // QT_CONFIG(tooltip)
        self.use_reverb_cb.setText(QCoreApplication.translate("smp_to_ds_ui", u" Reverb", None))
#if QT_CONFIG(tooltip)
        self.reverb_wet_dsb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Default wet value for reverb", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.use_ir_cb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Use or ignore IR samples\n"
"\n"
"IR samples must be located in a separate subdirectory called 'IR'\n"
"\n"
"NOTE: Decent Sampler applies a -12 dB volume attenuation to IR\n"
"'IR Tool' also available on my github can perform accurate volume compensation for IRs\n"
"so their volume match with each other", None))
#endif // QT_CONFIG(tooltip)
        self.use_ir_cb.setText(QCoreApplication.translate("smp_to_ds_ui", u"Use IR", None))
#if QT_CONFIG(tooltip)
        self.multi_out_cb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Enable separate group output to AUX channels\n"
"\n"
"NOTE: instrument level effects such as reverb and LPF are not present in the AUX channels", None))
#endif // QT_CONFIG(tooltip)
        self.multi_out_cb.setText(QCoreApplication.translate("smp_to_ds_ui", u"Multi Output", None))
#if QT_CONFIG(tooltip)
        self.hd_knobs_cb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Render knob atlases with double definition\n"
"A bit longer to render and leads to image strips up to 8K", None))
#endif // QT_CONFIG(tooltip)
        self.hd_knobs_cb.setText(QCoreApplication.translate("smp_to_ds_ui", u"HD Knobs", None))
#if QT_CONFIG(tooltip)
        self.estimate_delay_cb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Estimate sample set delay and write it in the info tool tip (circled 'i' in the corner)\n"
"\n"
"This does not affect the preset generation in any way\n"
"However, this is a useful information for instruments with a slower attack (typically brass and winds)\n"
"You can then use this value as an offset in your DAW to compensate the perceived delay in the samples\n"
"\n"
"Adds some overhead when generating the preset\n"
"(performs an estimate for each samples then average the results)\n"
"\n"
"Not required for tightly trimmed samples with sharp percussive attacks", None))
#endif // QT_CONFIG(tooltip)
        self.estimate_delay_cb.setText(QCoreApplication.translate("smp_to_ds_ui", u"Estimate Delay", None))
        self.add_suffix_cb.setText(QCoreApplication.translate("smp_to_ds_ui", u"Add Suffix", None))
#if QT_CONFIG(tooltip)
        self.suffix_le.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Add custom suffix to disambiguate a preset variant", None))
#endif // QT_CONFIG(tooltip)
        self.suffix_le.setPlaceholderText(QCoreApplication.translate("smp_to_ds_ui", u"_suffix", None))
#if QT_CONFIG(tooltip)
        self.auto_incr_cb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Auto increment to avoid overwriting", None))
#endif // QT_CONFIG(tooltip)
        self.auto_incr_cb.setText(QCoreApplication.translate("smp_to_ds_ui", u"Auto Increment", None))
        self.preset_fmt_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Preset Format", None))
        self.preset_fmt_cmb.setItemText(0, QCoreApplication.translate("smp_to_ds_ui", u"DS", None))
        self.preset_fmt_cmb.setItemText(1, QCoreApplication.translate("smp_to_ds_ui", u"SFZ", None))
        self.preset_fmt_cmb.setItemText(2, QCoreApplication.translate("smp_to_ds_ui", u"DS+SFZ", None))

#if QT_CONFIG(tooltip)
        self.preset_fmt_cmb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Preset format(s) to generate", None))
#endif // QT_CONFIG(tooltip)
        self.sfz_engine_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"SFZ Engine", None))
        self.sfz_engine_cmb.setItemText(0, QCoreApplication.translate("smp_to_ds_ui", u"generic", None))
        self.sfz_engine_cmb.setItemText(1, QCoreApplication.translate("smp_to_ds_ui", u"sforzando", None))
        self.sfz_engine_cmb.setItemText(2, QCoreApplication.translate("smp_to_ds_ui", u"sfizz", None))

#if QT_CONFIG(tooltip)
        self.sfz_engine_cmb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Enable or disable opcodes or hacks depending on target engine\n"
"\n"
"generic	Recommended default\n"
"sforzando	disable unsupported opcodes such as 'output', 'loop_crossfade' and 'sample_fadeout'\n"
"sfizz	EG last point hack to terminate release and avoid polyphony crash, should not be necessary", None))
#endif // QT_CONFIG(tooltip)
        self.use_eg_l.setText(QCoreApplication.translate("smp_to_ds_ui", u"Use EG", None))
        self.use_eg_cmb.setItemText(0, QCoreApplication.translate("smp_to_ds_ui", u"off", None))
        self.use_eg_cmb.setItemText(1, QCoreApplication.translate("smp_to_ds_ui", u"attack", None))
        self.use_eg_cmb.setItemText(2, QCoreApplication.translate("smp_to_ds_ui", u"on", None))

#if QT_CONFIG(tooltip)
        self.use_eg_cmb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Envelope Generatosr (EG) allow to better emulate DS envelopes with curve shaping\n"
"Also limit decay/release durations and thus spare polyphony a bit more (fake exp/log fades)\n"
"\n"
"off	use SFZ1 default envelopes (linear attack, exp decay/release)\n"
"attack	use envelope generator for attack trigger only\n"
"on	use envelope generator for all trigger mode", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.release_off_by_attack_cb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Spare polyphony by choking a release group by its corresponding attack group", None))
#endif // QT_CONFIG(tooltip)
        self.release_off_by_attack_cb.setText(QCoreApplication.translate("smp_to_ds_ui", u"Release Off By Attack", None))
#if QT_CONFIG(tooltip)
        self.create_preset_pb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Create a dspreset / SFZ file from input directory and current settings\n"
"\n"
"NOTE: SFZ is converted from dspreset data", None))
#endif // QT_CONFIG(tooltip)
        self.create_preset_pb.setText(QCoreApplication.translate("smp_to_ds_ui", u"Create Preset", None))
#if QT_CONFIG(tooltip)
        self.use_lossy_flac_cb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Convert FLAC to LossyFLAC\n"
"Pre-process allowing near-lossless compression to squeeze the files size a bit more\n"
"\n"
"NOTE: this only affects audio files copies inside the dslibrary file", None))
#endif // QT_CONFIG(tooltip)
        self.use_lossy_flac_cb.setText(QCoreApplication.translate("smp_to_ds_ui", u"LossyFLAC", None))
        self.lossy_flac_mode_cmb.setItemText(0, QCoreApplication.translate("smp_to_ds_ui", u"auto", None))
        self.lossy_flac_mode_cmb.setItemText(1, QCoreApplication.translate("smp_to_ds_ui", u"auto_ir", None))
        self.lossy_flac_mode_cmb.setItemText(2, QCoreApplication.translate("smp_to_ds_ui", u"16", None))

#if QT_CONFIG(tooltip)
        self.lossy_flac_mode_cmb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"LossyFLAC mode\n"
"auto	keep input bit-depth, ignore IR samples\n"
"auto_ir	keep input bit-depth, also process IR samples\n"
"16	reduce bit-depth to 16 bits except for IR", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.create_dslib_pb.setToolTip(QCoreApplication.translate("smp_to_ds_ui", u"Create dslibrary (which is a renamed zip file) from root directory by archiving only required files\n"
"At least one valid dspreset must exist in the directory\n"
"\n"
"NOTE: txt, sfz and settings files are also included in the archive", None))
#endif // QT_CONFIG(tooltip)
        self.create_dslib_pb.setText(QCoreApplication.translate("smp_to_ds_ui", u"Create dslibrary", None))
    # retranslateUi

