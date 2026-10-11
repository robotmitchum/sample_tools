# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mutate_tool.ui'
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

class Ui_mutate_tool_mw(object):
    def setupUi(self, mutate_tool_mw):
        if not mutate_tool_mw.objectName():
            mutate_tool_mw.setObjectName(u"mutate_tool_mw")
        mutate_tool_mw.resize(640, 768)
        mutate_tool_mw.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        self.centralwidget = QWidget(mutate_tool_mw)
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
        self.files_title_l.setStyleSheet(u"background-color: rgb(95, 144, 47);\n"
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
        self.options_title_l.setStyleSheet(u"background-color: rgb(95, 144, 47);\n"
"color: rgb(255, 255, 255);")

        self.verticalLayout.addWidget(self.options_title_l)

        self.variants_lyt = QHBoxLayout()
        self.variants_lyt.setObjectName(u"variants_lyt")
        self.variants_l = QLabel(self.centralwidget)
        self.variants_l.setObjectName(u"variants_l")

        self.variants_lyt.addWidget(self.variants_l)

        self.count_l = QLabel(self.centralwidget)
        self.count_l.setObjectName(u"count_l")
        self.count_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.variants_lyt.addWidget(self.count_l)

        self.count_sb = QSpinBox(self.centralwidget)
        self.count_sb.setObjectName(u"count_sb")
        self.count_sb.setFrame(False)
        self.count_sb.setAlignment(Qt.AlignCenter)
        self.count_sb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.count_sb.setMinimum(1)
        self.count_sb.setMaximum(16)
        self.count_sb.setValue(5)

        self.variants_lyt.addWidget(self.count_sb)

        self.seed_l = QLabel(self.centralwidget)
        self.seed_l.setObjectName(u"seed_l")
        self.seed_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.variants_lyt.addWidget(self.seed_l)

        self.seed_mode_cmb = QComboBox(self.centralwidget)
        self.seed_mode_cmb.addItem("")
        self.seed_mode_cmb.addItem("")
        self.seed_mode_cmb.setObjectName(u"seed_mode_cmb")

        self.variants_lyt.addWidget(self.seed_mode_cmb)

        self.seed_sb = QSpinBox(self.centralwidget)
        self.seed_sb.setObjectName(u"seed_sb")
        self.seed_sb.setEnabled(False)
        self.seed_sb.setFrame(False)
        self.seed_sb.setAlignment(Qt.AlignCenter)
        self.seed_sb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.seed_sb.setMinimum(1)
        self.seed_sb.setMaximum(99999)
        self.seed_sb.setValue(1)

        self.variants_lyt.addWidget(self.seed_sb)

        self.spc_wid_3 = QWidget(self.centralwidget)
        self.spc_wid_3.setObjectName(u"spc_wid_3")

        self.variants_lyt.addWidget(self.spc_wid_3)


        self.verticalLayout.addLayout(self.variants_lyt)

        self.line_5 = QFrame(self.centralwidget)
        self.line_5.setObjectName(u"line_5")
        self.line_5.setFrameShape(QFrame.Shape.HLine)
        self.line_5.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout.addWidget(self.line_5)

        self.input_lyt = QHBoxLayout()
        self.input_lyt.setObjectName(u"input_lyt")
        self.input_l = QLabel(self.centralwidget)
        self.input_l.setObjectName(u"input_l")

        self.input_lyt.addWidget(self.input_l)

        self.multi_chn_cb = QCheckBox(self.centralwidget)
        self.multi_chn_cb.setObjectName(u"multi_chn_cb")
        self.multi_chn_cb.setChecked(True)

        self.input_lyt.addWidget(self.multi_chn_cb)

        self.trim_db_l = QLabel(self.centralwidget)
        self.trim_db_l.setObjectName(u"trim_db_l")
        self.trim_db_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.input_lyt.addWidget(self.trim_db_l)

        self.input_wid = QWidget(self.centralwidget)
        self.input_wid.setObjectName(u"input_wid")
        self.horizontalLayout_2 = QHBoxLayout(self.input_wid)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.trim_db_dsb = QDoubleSpinBox(self.input_wid)
        self.trim_db_dsb.setObjectName(u"trim_db_dsb")
        self.trim_db_dsb.setFrame(False)
        self.trim_db_dsb.setAlignment(Qt.AlignCenter)
        self.trim_db_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.trim_db_dsb.setMinimum(-120.000000000000000)
        self.trim_db_dsb.setMaximum(0.000000000000000)
        self.trim_db_dsb.setSingleStep(0.100000000000000)
        self.trim_db_dsb.setValue(-60.000000000000000)

        self.horizontalLayout_2.addWidget(self.trim_db_dsb)


        self.input_lyt.addWidget(self.input_wid)


        self.verticalLayout.addLayout(self.input_lyt)

        self.transient_cues_lyt = QHBoxLayout()
        self.transient_cues_lyt.setObjectName(u"transient_cues_lyt")
        self.transient_cues_l = QLabel(self.centralwidget)
        self.transient_cues_l.setObjectName(u"transient_cues_l")

        self.transient_cues_lyt.addWidget(self.transient_cues_l)

        self.spc_wid = QWidget(self.centralwidget)
        self.spc_wid.setObjectName(u"spc_wid")

        self.transient_cues_lyt.addWidget(self.spc_wid)

        self.min_dur_l = QLabel(self.centralwidget)
        self.min_dur_l.setObjectName(u"min_dur_l")
        self.min_dur_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.transient_cues_lyt.addWidget(self.min_dur_l)

        self.min_dur_dsb = QDoubleSpinBox(self.centralwidget)
        self.min_dur_dsb.setObjectName(u"min_dur_dsb")
        self.min_dur_dsb.setMinimumSize(QSize(64, 0))
        self.min_dur_dsb.setFrame(False)
        self.min_dur_dsb.setAlignment(Qt.AlignCenter)
        self.min_dur_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.min_dur_dsb.setDecimals(3)
        self.min_dur_dsb.setMinimum(0.000000000000000)
        self.min_dur_dsb.setMaximum(4.000000000000000)
        self.min_dur_dsb.setSingleStep(0.100000000000000)
        self.min_dur_dsb.setValue(0.100000000000000)

        self.transient_cues_lyt.addWidget(self.min_dur_dsb)


        self.verticalLayout.addLayout(self.transient_cues_lyt)

        self.noise_tonal_split_lyt = QHBoxLayout()
        self.noise_tonal_split_lyt.setObjectName(u"noise_tonal_split_lyt")
        self.noise_tonal_l = QLabel(self.centralwidget)
        self.noise_tonal_l.setObjectName(u"noise_tonal_l")

        self.noise_tonal_split_lyt.addWidget(self.noise_tonal_l)

        self.stft_size_l = QLabel(self.centralwidget)
        self.stft_size_l.setObjectName(u"stft_size_l")
        self.stft_size_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.noise_tonal_split_lyt.addWidget(self.stft_size_l)

        self.split_stft_sb = QSpinBox(self.centralwidget)
        self.split_stft_sb.setObjectName(u"split_stft_sb")
        self.split_stft_sb.setFrame(False)
        self.split_stft_sb.setAlignment(Qt.AlignCenter)
        self.split_stft_sb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.split_stft_sb.setMinimum(256)
        self.split_stft_sb.setMaximum(16384)
        self.split_stft_sb.setValue(8192)

        self.noise_tonal_split_lyt.addWidget(self.split_stft_sb)

        self.iterations_l = QLabel(self.centralwidget)
        self.iterations_l.setObjectName(u"iterations_l")
        self.iterations_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.noise_tonal_split_lyt.addWidget(self.iterations_l)

        self.iterations_sb = QSpinBox(self.centralwidget)
        self.iterations_sb.setObjectName(u"iterations_sb")
        self.iterations_sb.setFrame(False)
        self.iterations_sb.setAlignment(Qt.AlignCenter)
        self.iterations_sb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.iterations_sb.setMinimum(1)
        self.iterations_sb.setMaximum(16)
        self.iterations_sb.setValue(1)

        self.noise_tonal_split_lyt.addWidget(self.iterations_sb)

        self.end_mode_l = QLabel(self.centralwidget)
        self.end_mode_l.setObjectName(u"end_mode_l")
        self.end_mode_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.noise_tonal_split_lyt.addWidget(self.end_mode_l)

        self.comboBox = QComboBox(self.centralwidget)
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.setObjectName(u"comboBox")

        self.noise_tonal_split_lyt.addWidget(self.comboBox)


        self.verticalLayout.addLayout(self.noise_tonal_split_lyt)

        self.line_4 = QFrame(self.centralwidget)
        self.line_4.setObjectName(u"line_4")
        self.line_4.setFrameShape(QFrame.Shape.HLine)
        self.line_4.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout.addWidget(self.line_4)

        self.params_grid_lyt = QGridLayout()
        self.params_grid_lyt.setObjectName(u"params_grid_lyt")
        self.params_grid_lyt.setHorizontalSpacing(16)
        self.params_grid_lyt.setVerticalSpacing(4)
        self.noise_lyt = QHBoxLayout()
        self.noise_lyt.setSpacing(0)
        self.noise_lyt.setObjectName(u"noise_lyt")
        self.noise_amp_dsb = QDoubleSpinBox(self.centralwidget)
        self.noise_amp_dsb.setObjectName(u"noise_amp_dsb")
        self.noise_amp_dsb.setMinimumSize(QSize(64, 0))
        self.noise_amp_dsb.setFrame(False)
        self.noise_amp_dsb.setAlignment(Qt.AlignCenter)
        self.noise_amp_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.noise_amp_dsb.setDecimals(3)
        self.noise_amp_dsb.setMinimum(0.000000000000000)
        self.noise_amp_dsb.setMaximum(4.000000000000000)
        self.noise_amp_dsb.setSingleStep(0.100000000000000)
        self.noise_amp_dsb.setValue(1.000000000000000)

        self.noise_lyt.addWidget(self.noise_amp_dsb)

        self.noise_rate_dsb = QDoubleSpinBox(self.centralwidget)
        self.noise_rate_dsb.setObjectName(u"noise_rate_dsb")
        self.noise_rate_dsb.setMinimumSize(QSize(64, 0))
        self.noise_rate_dsb.setFrame(False)
        self.noise_rate_dsb.setAlignment(Qt.AlignCenter)
        self.noise_rate_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.noise_rate_dsb.setDecimals(3)
        self.noise_rate_dsb.setMinimum(0.000000000000000)
        self.noise_rate_dsb.setMaximum(8.000000000000000)
        self.noise_rate_dsb.setSingleStep(0.100000000000000)
        self.noise_rate_dsb.setValue(1.000000000000000)

        self.noise_lyt.addWidget(self.noise_rate_dsb)

        self.noise_pitch_dsb = QDoubleSpinBox(self.centralwidget)
        self.noise_pitch_dsb.setObjectName(u"noise_pitch_dsb")
        self.noise_pitch_dsb.setMinimumSize(QSize(64, 0))
        self.noise_pitch_dsb.setFrame(False)
        self.noise_pitch_dsb.setAlignment(Qt.AlignCenter)
        self.noise_pitch_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.noise_pitch_dsb.setDecimals(3)
        self.noise_pitch_dsb.setMinimum(-36.000000000000000)
        self.noise_pitch_dsb.setMaximum(36.000000000000000)
        self.noise_pitch_dsb.setSingleStep(0.100000000000000)
        self.noise_pitch_dsb.setValue(0.000000000000000)

        self.noise_lyt.addWidget(self.noise_pitch_dsb)


        self.params_grid_lyt.addLayout(self.noise_lyt, 2, 2, 1, 1)

        self.noise_l = QLabel(self.centralwidget)
        self.noise_l.setObjectName(u"noise_l")
        self.noise_l.setFont(font)
        self.noise_l.setAlignment(Qt.AlignCenter)

        self.params_grid_lyt.addWidget(self.noise_l, 0, 2, 1, 1)

        self.tonal_lyt = QHBoxLayout()
        self.tonal_lyt.setSpacing(0)
        self.tonal_lyt.setObjectName(u"tonal_lyt")
        self.tonal_amp_dsb = QDoubleSpinBox(self.centralwidget)
        self.tonal_amp_dsb.setObjectName(u"tonal_amp_dsb")
        self.tonal_amp_dsb.setMinimumSize(QSize(64, 0))
        self.tonal_amp_dsb.setFrame(False)
        self.tonal_amp_dsb.setAlignment(Qt.AlignCenter)
        self.tonal_amp_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.tonal_amp_dsb.setDecimals(3)
        self.tonal_amp_dsb.setMinimum(0.000000000000000)
        self.tonal_amp_dsb.setMaximum(4.000000000000000)
        self.tonal_amp_dsb.setSingleStep(0.100000000000000)
        self.tonal_amp_dsb.setValue(1.000000000000000)

        self.tonal_lyt.addWidget(self.tonal_amp_dsb)

        self.tonal_rate_dsb = QDoubleSpinBox(self.centralwidget)
        self.tonal_rate_dsb.setObjectName(u"tonal_rate_dsb")
        self.tonal_rate_dsb.setMinimumSize(QSize(64, 0))
        self.tonal_rate_dsb.setFrame(False)
        self.tonal_rate_dsb.setAlignment(Qt.AlignCenter)
        self.tonal_rate_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.tonal_rate_dsb.setDecimals(3)
        self.tonal_rate_dsb.setMinimum(0.000000000000000)
        self.tonal_rate_dsb.setMaximum(8.000000000000000)
        self.tonal_rate_dsb.setSingleStep(0.100000000000000)
        self.tonal_rate_dsb.setValue(1.000000000000000)

        self.tonal_lyt.addWidget(self.tonal_rate_dsb)

        self.tonal_pitch_dsb = QDoubleSpinBox(self.centralwidget)
        self.tonal_pitch_dsb.setObjectName(u"tonal_pitch_dsb")
        self.tonal_pitch_dsb.setMinimumSize(QSize(64, 0))
        self.tonal_pitch_dsb.setFrame(False)
        self.tonal_pitch_dsb.setAlignment(Qt.AlignCenter)
        self.tonal_pitch_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.tonal_pitch_dsb.setDecimals(3)
        self.tonal_pitch_dsb.setMinimum(-36.000000000000000)
        self.tonal_pitch_dsb.setMaximum(36.000000000000000)
        self.tonal_pitch_dsb.setSingleStep(0.100000000000000)
        self.tonal_pitch_dsb.setValue(0.000000000000000)

        self.tonal_lyt.addWidget(self.tonal_pitch_dsb)


        self.params_grid_lyt.addLayout(self.tonal_lyt, 2, 3, 1, 1)

        self.noise_lbl_lyt = QHBoxLayout()
        self.noise_lbl_lyt.setSpacing(0)
        self.noise_lbl_lyt.setObjectName(u"noise_lbl_lyt")
        self.noise_amp_l = QLabel(self.centralwidget)
        self.noise_amp_l.setObjectName(u"noise_amp_l")
        self.noise_amp_l.setAlignment(Qt.AlignCenter)

        self.noise_lbl_lyt.addWidget(self.noise_amp_l)

        self.noise_rate_l = QLabel(self.centralwidget)
        self.noise_rate_l.setObjectName(u"noise_rate_l")
        self.noise_rate_l.setAlignment(Qt.AlignCenter)

        self.noise_lbl_lyt.addWidget(self.noise_rate_l)

        self.noise_pitch_l = QLabel(self.centralwidget)
        self.noise_pitch_l.setObjectName(u"noise_pitch_l")
        self.noise_pitch_l.setAlignment(Qt.AlignCenter)

        self.noise_lbl_lyt.addWidget(self.noise_pitch_l)


        self.params_grid_lyt.addLayout(self.noise_lbl_lyt, 1, 2, 1, 1)

        self.params_pb = QPushButton(self.centralwidget)
        self.params_pb.setObjectName(u"params_pb")
        self.params_pb.setFont(font)

        self.params_grid_lyt.addWidget(self.params_pb, 2, 1, 1, 1)

        self.random_pb = QPushButton(self.centralwidget)
        self.random_pb.setObjectName(u"random_pb")
        self.random_pb.setFont(font)

        self.params_grid_lyt.addWidget(self.random_pb, 3, 1, 1, 1)

        self.tonal_l = QLabel(self.centralwidget)
        self.tonal_l.setObjectName(u"tonal_l")
        self.tonal_l.setFont(font)
        self.tonal_l.setAlignment(Qt.AlignCenter)

        self.params_grid_lyt.addWidget(self.tonal_l, 0, 3, 1, 1)

        self.tonal_lbl_lyt = QHBoxLayout()
        self.tonal_lbl_lyt.setSpacing(0)
        self.tonal_lbl_lyt.setObjectName(u"tonal_lbl_lyt")
        self.tonal_amp_l = QLabel(self.centralwidget)
        self.tonal_amp_l.setObjectName(u"tonal_amp_l")
        self.tonal_amp_l.setAlignment(Qt.AlignCenter)

        self.tonal_lbl_lyt.addWidget(self.tonal_amp_l)

        self.tonal_rate_l = QLabel(self.centralwidget)
        self.tonal_rate_l.setObjectName(u"tonal_rate_l")
        self.tonal_rate_l.setAlignment(Qt.AlignCenter)

        self.tonal_lbl_lyt.addWidget(self.tonal_rate_l)

        self.tonal_pitch_l = QLabel(self.centralwidget)
        self.tonal_pitch_l.setObjectName(u"tonal_pitch_l")
        self.tonal_pitch_l.setAlignment(Qt.AlignCenter)

        self.tonal_lbl_lyt.addWidget(self.tonal_pitch_l)


        self.params_grid_lyt.addLayout(self.tonal_lbl_lyt, 1, 3, 1, 1)

        self.noise_rand_lyt = QHBoxLayout()
        self.noise_rand_lyt.setSpacing(0)
        self.noise_rand_lyt.setObjectName(u"noise_rand_lyt")
        self.noise_rand_amp_dsb = QDoubleSpinBox(self.centralwidget)
        self.noise_rand_amp_dsb.setObjectName(u"noise_rand_amp_dsb")
        self.noise_rand_amp_dsb.setMinimumSize(QSize(64, 0))
        self.noise_rand_amp_dsb.setFrame(False)
        self.noise_rand_amp_dsb.setAlignment(Qt.AlignCenter)
        self.noise_rand_amp_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.noise_rand_amp_dsb.setDecimals(3)
        self.noise_rand_amp_dsb.setMinimum(0.000000000000000)
        self.noise_rand_amp_dsb.setMaximum(4.000000000000000)
        self.noise_rand_amp_dsb.setSingleStep(0.100000000000000)
        self.noise_rand_amp_dsb.setValue(0.500000000000000)

        self.noise_rand_lyt.addWidget(self.noise_rand_amp_dsb)

        self.noise_rand_rate_dsb = QDoubleSpinBox(self.centralwidget)
        self.noise_rand_rate_dsb.setObjectName(u"noise_rand_rate_dsb")
        self.noise_rand_rate_dsb.setMinimumSize(QSize(64, 0))
        self.noise_rand_rate_dsb.setFrame(False)
        self.noise_rand_rate_dsb.setAlignment(Qt.AlignCenter)
        self.noise_rand_rate_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.noise_rand_rate_dsb.setDecimals(3)
        self.noise_rand_rate_dsb.setMinimum(0.000000000000000)
        self.noise_rand_rate_dsb.setMaximum(8.000000000000000)
        self.noise_rand_rate_dsb.setSingleStep(0.100000000000000)
        self.noise_rand_rate_dsb.setValue(0.333000000000000)

        self.noise_rand_lyt.addWidget(self.noise_rand_rate_dsb)

        self.noise_rand_pitch_dsb = QDoubleSpinBox(self.centralwidget)
        self.noise_rand_pitch_dsb.setObjectName(u"noise_rand_pitch_dsb")
        self.noise_rand_pitch_dsb.setMinimumSize(QSize(64, 0))
        self.noise_rand_pitch_dsb.setFrame(False)
        self.noise_rand_pitch_dsb.setAlignment(Qt.AlignCenter)
        self.noise_rand_pitch_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.noise_rand_pitch_dsb.setDecimals(3)
        self.noise_rand_pitch_dsb.setMinimum(0.000000000000000)
        self.noise_rand_pitch_dsb.setMaximum(36.000000000000000)
        self.noise_rand_pitch_dsb.setSingleStep(0.100000000000000)
        self.noise_rand_pitch_dsb.setValue(2.000000000000000)

        self.noise_rand_lyt.addWidget(self.noise_rand_pitch_dsb)


        self.params_grid_lyt.addLayout(self.noise_rand_lyt, 3, 2, 1, 1)

        self.tonal_rand_lyt = QHBoxLayout()
        self.tonal_rand_lyt.setSpacing(0)
        self.tonal_rand_lyt.setObjectName(u"tonal_rand_lyt")
        self.tonal_rand_amp_dsb = QDoubleSpinBox(self.centralwidget)
        self.tonal_rand_amp_dsb.setObjectName(u"tonal_rand_amp_dsb")
        self.tonal_rand_amp_dsb.setMinimumSize(QSize(64, 0))
        self.tonal_rand_amp_dsb.setFrame(False)
        self.tonal_rand_amp_dsb.setAlignment(Qt.AlignCenter)
        self.tonal_rand_amp_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.tonal_rand_amp_dsb.setDecimals(3)
        self.tonal_rand_amp_dsb.setMinimum(0.000000000000000)
        self.tonal_rand_amp_dsb.setMaximum(4.000000000000000)
        self.tonal_rand_amp_dsb.setSingleStep(0.100000000000000)
        self.tonal_rand_amp_dsb.setValue(0.500000000000000)

        self.tonal_rand_lyt.addWidget(self.tonal_rand_amp_dsb)

        self.tonal_rand_rate_dsb = QDoubleSpinBox(self.centralwidget)
        self.tonal_rand_rate_dsb.setObjectName(u"tonal_rand_rate_dsb")
        self.tonal_rand_rate_dsb.setMinimumSize(QSize(64, 0))
        self.tonal_rand_rate_dsb.setFrame(False)
        self.tonal_rand_rate_dsb.setAlignment(Qt.AlignCenter)
        self.tonal_rand_rate_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.tonal_rand_rate_dsb.setDecimals(3)
        self.tonal_rand_rate_dsb.setMinimum(0.000000000000000)
        self.tonal_rand_rate_dsb.setMaximum(8.000000000000000)
        self.tonal_rand_rate_dsb.setSingleStep(0.100000000000000)
        self.tonal_rand_rate_dsb.setValue(0.333000000000000)

        self.tonal_rand_lyt.addWidget(self.tonal_rand_rate_dsb)

        self.tonal_rand_pitch_dsb = QDoubleSpinBox(self.centralwidget)
        self.tonal_rand_pitch_dsb.setObjectName(u"tonal_rand_pitch_dsb")
        self.tonal_rand_pitch_dsb.setMinimumSize(QSize(64, 0))
        self.tonal_rand_pitch_dsb.setFrame(False)
        self.tonal_rand_pitch_dsb.setAlignment(Qt.AlignCenter)
        self.tonal_rand_pitch_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.tonal_rand_pitch_dsb.setDecimals(3)
        self.tonal_rand_pitch_dsb.setMinimum(0.000000000000000)
        self.tonal_rand_pitch_dsb.setMaximum(36.000000000000000)
        self.tonal_rand_pitch_dsb.setSingleStep(0.100000000000000)
        self.tonal_rand_pitch_dsb.setValue(0.050000000000000)

        self.tonal_rand_lyt.addWidget(self.tonal_rand_pitch_dsb)


        self.params_grid_lyt.addLayout(self.tonal_rand_lyt, 3, 3, 1, 1)

        self.match_cues_cb = QCheckBox(self.centralwidget)
        self.match_cues_cb.setObjectName(u"match_cues_cb")
        self.match_cues_cb.setChecked(True)

        self.params_grid_lyt.addWidget(self.match_cues_cb, 3, 4, 1, 1)

        self.spc_wid_2 = QWidget(self.centralwidget)
        self.spc_wid_2.setObjectName(u"spc_wid_2")

        self.params_grid_lyt.addWidget(self.spc_wid_2, 0, 4, 1, 1)


        self.verticalLayout.addLayout(self.params_grid_lyt)

        self.amp_interp_lyt = QHBoxLayout()
        self.amp_interp_lyt.setSpacing(8)
        self.amp_interp_lyt.setObjectName(u"amp_interp_lyt")
        self.spc_wid_6 = QWidget(self.centralwidget)
        self.spc_wid_6.setObjectName(u"spc_wid_6")
        self.spc_wid_5 = QWidget(self.spc_wid_6)
        self.spc_wid_5.setObjectName(u"spc_wid_5")
        self.spc_wid_5.setGeometry(QRect(140, 0, 149, 20))

        self.amp_interp_lyt.addWidget(self.spc_wid_6)

        self.amp_interp_l = QLabel(self.centralwidget)
        self.amp_interp_l.setObjectName(u"amp_interp_l")
        self.amp_interp_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.amp_interp_lyt.addWidget(self.amp_interp_l)

        self.amp_interp_cmb = QComboBox(self.centralwidget)
        self.amp_interp_cmb.addItem("")
        self.amp_interp_cmb.addItem("")
        self.amp_interp_cmb.addItem("")
        self.amp_interp_cmb.setObjectName(u"amp_interp_cmb")

        self.amp_interp_lyt.addWidget(self.amp_interp_cmb)

        self.spc_wid_7 = QWidget(self.centralwidget)
        self.spc_wid_7.setObjectName(u"spc_wid_7")
        self.spc_wid_8 = QWidget(self.spc_wid_7)
        self.spc_wid_8.setObjectName(u"spc_wid_8")
        self.spc_wid_8.setGeometry(QRect(140, 0, 149, 20))

        self.amp_interp_lyt.addWidget(self.spc_wid_7)


        self.verticalLayout.addLayout(self.amp_interp_lyt)

        self.line_2 = QFrame(self.centralwidget)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.Shape.HLine)
        self.line_2.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout.addWidget(self.line_2)

        self.sampling_lyt = QHBoxLayout()
        self.sampling_lyt.setObjectName(u"sampling_lyt")
        self.sampling_wid = QWidget(self.centralwidget)
        self.sampling_wid.setObjectName(u"sampling_wid")
        self.horizontalLayout_4 = QHBoxLayout(self.sampling_wid)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.oversample_cb = QCheckBox(self.sampling_wid)
        self.oversample_cb.setObjectName(u"oversample_cb")

        self.horizontalLayout_4.addWidget(self.oversample_cb)

        self.oversample_sb = QSpinBox(self.sampling_wid)
        self.oversample_sb.setObjectName(u"oversample_sb")
        self.oversample_sb.setEnabled(False)
        self.oversample_sb.setFrame(False)
        self.oversample_sb.setAlignment(Qt.AlignCenter)
        self.oversample_sb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.oversample_sb.setMinimum(2)
        self.oversample_sb.setMaximum(4)

        self.horizontalLayout_4.addWidget(self.oversample_sb)

        self.target_sr_l = QLabel(self.sampling_wid)
        self.target_sr_l.setObjectName(u"target_sr_l")
        self.target_sr_l.setMinimumSize(QSize(48, 0))
        self.target_sr_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_4.addWidget(self.target_sr_l)

        self.target_sr_cmb = QComboBox(self.sampling_wid)
        self.target_sr_cmb.addItem("")
        self.target_sr_cmb.addItem("")
        self.target_sr_cmb.setObjectName(u"target_sr_cmb")

        self.horizontalLayout_4.addWidget(self.target_sr_cmb)

        self.target_sr_sb = QSpinBox(self.sampling_wid)
        self.target_sr_sb.setObjectName(u"target_sr_sb")
        self.target_sr_sb.setEnabled(False)
        self.target_sr_sb.setMinimumSize(QSize(48, 0))
        self.target_sr_sb.setFrame(False)
        self.target_sr_sb.setAlignment(Qt.AlignCenter)
        self.target_sr_sb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.target_sr_sb.setMinimum(8000)
        self.target_sr_sb.setMaximum(96000)
        self.target_sr_sb.setValue(48000)

        self.horizontalLayout_4.addWidget(self.target_sr_sb)


        self.sampling_lyt.addWidget(self.sampling_wid)


        self.verticalLayout.addLayout(self.sampling_lyt)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.transient_cues_l_2 = QLabel(self.centralwidget)
        self.transient_cues_l_2.setObjectName(u"transient_cues_l_2")

        self.horizontalLayout.addWidget(self.transient_cues_l_2)

        self.spc_wid_4 = QWidget(self.centralwidget)
        self.spc_wid_4.setObjectName(u"spc_wid_4")

        self.horizontalLayout.addWidget(self.spc_wid_4)

        self.fade_db_l = QLabel(self.centralwidget)
        self.fade_db_l.setObjectName(u"fade_db_l")
        self.fade_db_l.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout.addWidget(self.fade_db_l)

        self.fade_db_dsb = QDoubleSpinBox(self.centralwidget)
        self.fade_db_dsb.setObjectName(u"fade_db_dsb")
        self.fade_db_dsb.setFrame(False)
        self.fade_db_dsb.setAlignment(Qt.AlignCenter)
        self.fade_db_dsb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.fade_db_dsb.setMinimum(-120.000000000000000)
        self.fade_db_dsb.setMaximum(0.000000000000000)
        self.fade_db_dsb.setSingleStep(0.100000000000000)
        self.fade_db_dsb.setValue(-40.000000000000000)

        self.horizontalLayout.addWidget(self.fade_db_dsb)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.output_path_title_l = QLabel(self.centralwidget)
        self.output_path_title_l.setObjectName(u"output_path_title_l")
        sizePolicy.setHeightForWidth(self.output_path_title_l.sizePolicy().hasHeightForWidth())
        self.output_path_title_l.setSizePolicy(sizePolicy)
        self.output_path_title_l.setMinimumSize(QSize(64, 0))
        self.output_path_title_l.setStyleSheet(u"background-color: rgb(95, 144, 47);\n"
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
        self.settings_title_l.setStyleSheet(u"background-color: rgb(95, 144, 47);\n"
"color: rgb(255, 255, 255);")

        self.verticalLayout.addWidget(self.settings_title_l)

        self.file_settings_lyt = QHBoxLayout()
        self.file_settings_lyt.setObjectName(u"file_settings_lyt")
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
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
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.bd_l.sizePolicy().hasHeightForWidth())
        self.bd_l.setSizePolicy(sizePolicy2)

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
        sizePolicy2.setHeightForWidth(self.add_suffix_cb.sizePolicy().hasHeightForWidth())
        self.add_suffix_cb.setSizePolicy(sizePolicy2)
        self.add_suffix_cb.setChecked(True)

        self.file_settings_lyt.addWidget(self.add_suffix_cb)

        self.suffix_le = QLineEdit(self.centralwidget)
        self.suffix_le.setObjectName(u"suffix_le")
        sizePolicy2.setHeightForWidth(self.suffix_le.sizePolicy().hasHeightForWidth())
        self.suffix_le.setSizePolicy(sizePolicy2)
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
        sizePolicy2.setHeightForWidth(self.process_sel_pb.sizePolicy().hasHeightForWidth())
        self.process_sel_pb.setSizePolicy(sizePolicy2)
        self.process_sel_pb.setMinimumSize(QSize(160, 0))
        self.process_sel_pb.setFont(font)

        self.buttons_lyt.addWidget(self.process_sel_pb, 0, 1, 1, 1)

        self.preview_pb = QPushButton(self.centralwidget)
        self.preview_pb.setObjectName(u"preview_pb")
        sizePolicy2.setHeightForWidth(self.preview_pb.sizePolicy().hasHeightForWidth())
        self.preview_pb.setSizePolicy(sizePolicy2)
        self.preview_pb.setMinimumSize(QSize(160, 0))
        self.preview_pb.setFont(font)

        self.buttons_lyt.addWidget(self.preview_pb, 0, 0, 1, 1)

        self.process_pb = QPushButton(self.centralwidget)
        self.process_pb.setObjectName(u"process_pb")
        sizePolicy2.setHeightForWidth(self.process_pb.sizePolicy().hasHeightForWidth())
        self.process_pb.setSizePolicy(sizePolicy2)
        self.process_pb.setMinimumSize(QSize(160, 0))
        self.process_pb.setFont(font)
        self.process_pb.setStyleSheet(u"QPushButton{background-color: rgb(95, 144, 47);color: rgb(255, 255, 255);}")

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

        mutate_tool_mw.setCentralWidget(self.centralwidget)

        self.retranslateUi(mutate_tool_mw)

        self.amp_interp_cmb.setCurrentIndex(0)
        self.format_cmb.setCurrentIndex(0)
        self.bitdepth_cmb.setCurrentIndex(0)
        self.process_pb.setDefault(False)


        QMetaObject.connectSlotsByName(mutate_tool_mw)
    # setupUi

    def retranslateUi(self, mutate_tool_mw):
        mutate_tool_mw.setWindowTitle(QCoreApplication.translate("mutate_tool_mw", u"Mutate Tool", None))
        self.files_title_l.setText(QCoreApplication.translate("mutate_tool_mw", u"Input Files", None))
#if QT_CONFIG(tooltip)
        self.files_lw.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Drag and drop files or directories\n"
"Right click for context actions\n"
"Double-click to play", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.set_files_tb.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Set files to process", None))
#endif // QT_CONFIG(tooltip)
        self.set_files_tb.setText(QCoreApplication.translate("mutate_tool_mw", u"...", None))
        self.options_title_l.setText(QCoreApplication.translate("mutate_tool_mw", u"Options", None))
#if QT_CONFIG(tooltip)
        self.variants_l.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Input preprocessing", None))
#endif // QT_CONFIG(tooltip)
        self.variants_l.setText(QCoreApplication.translate("mutate_tool_mw", u"Variations", None))
        self.count_l.setText(QCoreApplication.translate("mutate_tool_mw", u"Count", None))
#if QT_CONFIG(tooltip)
        self.count_sb.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Number of generated variants", None))
#endif // QT_CONFIG(tooltip)
        self.seed_l.setText(QCoreApplication.translate("mutate_tool_mw", u"Seed", None))
        self.seed_mode_cmb.setItemText(0, QCoreApplication.translate("mutate_tool_mw", u"name", None))
        self.seed_mode_cmb.setItemText(1, QCoreApplication.translate("mutate_tool_mw", u"value", None))

#if QT_CONFIG(tooltip)
        self.seed_mode_cmb.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"name	use the file name as seed for the random generator\n"
"	each file will have a distinct randomization\n"
"value	specify a manual seed value\n"
"	all files will be randomized similarly", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.seed_sb.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Seed of the random number generator\n"
"The seed is incremented for each variation", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.input_l.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Input preprocessing", None))
#endif // QT_CONFIG(tooltip)
        self.input_l.setText(QCoreApplication.translate("mutate_tool_mw", u"Input", None))
#if QT_CONFIG(tooltip)
        self.multi_chn_cb.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Process in mono when unchecked", None))
#endif // QT_CONFIG(tooltip)
        self.multi_chn_cb.setText(QCoreApplication.translate("mutate_tool_mw", u"Mutichannel Processing", None))
        self.trim_db_l.setText(QCoreApplication.translate("mutate_tool_mw", u"Trim dB", None))
#if QT_CONFIG(tooltip)
        self.trim_db_dsb.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Trim threshold in dB\n"
"Trim starting and ending silences from both input and output results", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.transient_cues_l.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Transient cues detection settings\n"
"Randomization is then performed on each of these detected cues", None))
#endif // QT_CONFIG(tooltip)
        self.transient_cues_l.setText(QCoreApplication.translate("mutate_tool_mw", u"Transient Cues", None))
#if QT_CONFIG(tooltip)
        self.spc_wid.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Transient cues detection settings\n"
"Randomization is then performed on each of these detected cues", None))
#endif // QT_CONFIG(tooltip)
        self.min_dur_l.setText(QCoreApplication.translate("mutate_tool_mw", u"Min Duration", None))
#if QT_CONFIG(tooltip)
        self.min_dur_dsb.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Minimum duration between transient cues in seconds\n"
"A larger value decimates more cues leading to a smoother less chaotic result", None))
#endif // QT_CONFIG(tooltip)
        self.noise_tonal_l.setText(QCoreApplication.translate("mutate_tool_mw", u"Noise|Tonal Splitting", None))
        self.stft_size_l.setText(QCoreApplication.translate("mutate_tool_mw", u"STFT size", None))
#if QT_CONFIG(tooltip)
        self.split_stft_sb.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Transient (lower) vs frequency (higher) precision trade-off", None))
#endif // QT_CONFIG(tooltip)
        self.iterations_l.setText(QCoreApplication.translate("mutate_tool_mw", u"Iterations", None))
#if QT_CONFIG(tooltip)
        self.iterations_sb.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Number of times the noise|tonal separation is repeated over itself\n"
"Separates noise from tonal content even more though it may sound more artificial", None))
#endif // QT_CONFIG(tooltip)
        self.end_mode_l.setText(QCoreApplication.translate("mutate_tool_mw", u"End Mode", None))
        self.comboBox.setItemText(0, QCoreApplication.translate("mutate_tool_mw", u"fade", None))
        self.comboBox.setItemText(1, QCoreApplication.translate("mutate_tool_mw", u"loop", None))
        self.comboBox.setItemText(2, QCoreApplication.translate("mutate_tool_mw", u"extend", None))

#if QT_CONFIG(tooltip)
        self.comboBox.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Define how the end boundary is padded to avoid STFT wrap-around\n"
"fade	fade end\n"
"loop	use sample loop information if present\n"
"extend	repeat last STFT time frame", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.noise_l.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Noise content parameters\n"
"This deals with the transients of the sound\n"
"Noise can typically bear with much more abuse than tonal content", None))
#endif // QT_CONFIG(tooltip)
        self.noise_l.setText(QCoreApplication.translate("mutate_tool_mw", u"Noise", None))
#if QT_CONFIG(tooltip)
        self.noise_amp_l.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Volume scaling", None))
#endif // QT_CONFIG(tooltip)
        self.noise_amp_l.setText(QCoreApplication.translate("mutate_tool_mw", u"Amp", None))
#if QT_CONFIG(tooltip)
        self.noise_rate_l.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Rate of the time stretching", None))
#endif // QT_CONFIG(tooltip)
        self.noise_rate_l.setText(QCoreApplication.translate("mutate_tool_mw", u"Rate", None))
#if QT_CONFIG(tooltip)
        self.noise_pitch_l.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Pitch shift (in semitones)", None))
#endif // QT_CONFIG(tooltip)
        self.noise_pitch_l.setText(QCoreApplication.translate("mutate_tool_mw", u"Pitch", None))
#if QT_CONFIG(tooltip)
        self.params_pb.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Constant parameters\n"
"Amp and Rate are multipliers\n"
"Pitch is an offset (in semitones)", None))
#endif // QT_CONFIG(tooltip)
        self.params_pb.setText(QCoreApplication.translate("mutate_tool_mw", u"Params", None))
#if QT_CONFIG(tooltip)
        self.random_pb.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Randomized parameters\n"
"Values represent the scale of the random generator", None))
#endif // QT_CONFIG(tooltip)
        self.random_pb.setText(QCoreApplication.translate("mutate_tool_mw", u"Random", None))
#if QT_CONFIG(tooltip)
        self.tonal_l.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Tonal content parameters\n"
"This deals with the \"body\" of the sound\n"
"Tonal content should typically not bear too much pitch variation", None))
#endif // QT_CONFIG(tooltip)
        self.tonal_l.setText(QCoreApplication.translate("mutate_tool_mw", u"Tonal", None))
#if QT_CONFIG(tooltip)
        self.tonal_amp_l.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Volume scaling", None))
#endif // QT_CONFIG(tooltip)
        self.tonal_amp_l.setText(QCoreApplication.translate("mutate_tool_mw", u"Amp", None))
#if QT_CONFIG(tooltip)
        self.tonal_rate_l.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Rate of the time stretching", None))
#endif // QT_CONFIG(tooltip)
        self.tonal_rate_l.setText(QCoreApplication.translate("mutate_tool_mw", u"Rate", None))
#if QT_CONFIG(tooltip)
        self.tonal_pitch_l.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Pitch shift (in semitones)", None))
#endif // QT_CONFIG(tooltip)
        self.tonal_pitch_l.setText(QCoreApplication.translate("mutate_tool_mw", u"Pitch", None))
#if QT_CONFIG(tooltip)
        self.match_cues_cb.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Scale tonal cues so tonal and noise end cues match, keeping overall length\n"
"When unchecked, noise and tonal content are more free to drift away from each other", None))
#endif // QT_CONFIG(tooltip)
        self.match_cues_cb.setText(QCoreApplication.translate("mutate_tool_mw", u"Match End Cues", None))
#if QT_CONFIG(tooltip)
        self.amp_interp_l.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.amp_interp_l.setText(QCoreApplication.translate("mutate_tool_mw", u"Amplitude Interpolation", None))
        self.amp_interp_cmb.setItemText(0, QCoreApplication.translate("mutate_tool_mw", u"linear", None))
        self.amp_interp_cmb.setItemText(1, QCoreApplication.translate("mutate_tool_mw", u"quadratic", None))
        self.amp_interp_cmb.setItemText(2, QCoreApplication.translate("mutate_tool_mw", u"cubic", None))

#if QT_CONFIG(tooltip)
        self.amp_interp_cmb.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Define which interpolation to use for the random amplitude cues\n"
"lnear	tame interpolation without overshoots\n"
"quadratic	smoother but with overshoots\n"
"cubic	even more continuous but with crazier overshoots", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.oversample_cb.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Enable oversampling for rate/pitch processing", None))
#endif // QT_CONFIG(tooltip)
        self.oversample_cb.setText(QCoreApplication.translate("mutate_tool_mw", u"Oversampling", None))
#if QT_CONFIG(tooltip)
        self.oversample_sb.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Oversampling Factor", None))
#endif // QT_CONFIG(tooltip)
        self.target_sr_l.setText(QCoreApplication.translate("mutate_tool_mw", u"Target Sampling Rate", None))
        self.target_sr_cmb.setItemText(0, QCoreApplication.translate("mutate_tool_mw", u"same", None))
        self.target_sr_cmb.setItemText(1, QCoreApplication.translate("mutate_tool_mw", u"custom", None))

#if QT_CONFIG(tooltip)
        self.target_sr_cmb.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"same	same sampling rate as input\n"
"custom	custom sampling rate", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.target_sr_sb.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Custom sampling rate used for resulting files", None))
#endif // QT_CONFIG(tooltip)
        self.transient_cues_l_2.setText(QCoreApplication.translate("mutate_tool_mw", u"Fade In/Out", None))
        self.fade_db_l.setText(QCoreApplication.translate("mutate_tool_mw", u"Fade dB", None))
#if QT_CONFIG(tooltip)
        self.fade_db_dsb.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Fade threshold in dB\n"
"Apply micro log fades at start and end of resulting samples\n"
"when the volume falls under the given threshold, preventing clicks/pops", None))
#endif // QT_CONFIG(tooltip)
        self.output_path_title_l.setText(QCoreApplication.translate("mutate_tool_mw", u"Output Path", None))
#if QT_CONFIG(tooltip)
        self.output_path_l.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Output path", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.set_output_path_tb.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Set output path\n"
"Process files in their respective directory if empty", None))
#endif // QT_CONFIG(tooltip)
        self.set_output_path_tb.setText(QCoreApplication.translate("mutate_tool_mw", u"...", None))
        self.settings_title_l.setText(QCoreApplication.translate("mutate_tool_mw", u"File Settings", None))
        self.label_2.setText(QCoreApplication.translate("mutate_tool_mw", u"Format", None))
        self.format_cmb.setItemText(0, QCoreApplication.translate("mutate_tool_mw", u"same", None))
        self.format_cmb.setItemText(1, QCoreApplication.translate("mutate_tool_mw", u"wav", None))
        self.format_cmb.setItemText(2, QCoreApplication.translate("mutate_tool_mw", u"flac", None))

#if QT_CONFIG(tooltip)
        self.format_cmb.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"flac only allows integer format up to 24 bits\n"
"Use wav for 32 bits float", None))
#endif // QT_CONFIG(tooltip)
        self.bd_l.setText(QCoreApplication.translate("mutate_tool_mw", u"Bit Depth", None))
        self.bitdepth_cmb.setItemText(0, QCoreApplication.translate("mutate_tool_mw", u"same", None))
        self.bitdepth_cmb.setItemText(1, QCoreApplication.translate("mutate_tool_mw", u"16", None))
        self.bitdepth_cmb.setItemText(2, QCoreApplication.translate("mutate_tool_mw", u"24", None))
        self.bitdepth_cmb.setItemText(3, QCoreApplication.translate("mutate_tool_mw", u"32", None))

        self.add_suffix_cb.setText(QCoreApplication.translate("mutate_tool_mw", u"Add suffix", None))
#if QT_CONFIG(tooltip)
        self.suffix_le.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Suffix added to the base name", None))
#endif // QT_CONFIG(tooltip)
        self.suffix_le.setText(QCoreApplication.translate("mutate_tool_mw", u"_rr", None))
#if QT_CONFIG(tooltip)
        self.no_overwriting_cb.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Avoid overwriting original files by moving them to a backup directory", None))
#endif // QT_CONFIG(tooltip)
        self.no_overwriting_cb.setText(QCoreApplication.translate("mutate_tool_mw", u"Avoid Overwriting", None))
#if QT_CONFIG(tooltip)
        self.process_sel_pb.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Process selected item(s) with current settings", None))
#endif // QT_CONFIG(tooltip)
        self.process_sel_pb.setText(QCoreApplication.translate("mutate_tool_mw", u"Process Selected", None))
#if QT_CONFIG(tooltip)
        self.preview_pb.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Apply current settings to selected item\n"
"and play the result without writing anything", None))
#endif // QT_CONFIG(tooltip)
        self.preview_pb.setText(QCoreApplication.translate("mutate_tool_mw", u"Preview Selected", None))
#if QT_CONFIG(tooltip)
        self.process_pb.setToolTip(QCoreApplication.translate("mutate_tool_mw", u"Process all input files with current settings", None))
#endif // QT_CONFIG(tooltip)
        self.process_pb.setText(QCoreApplication.translate("mutate_tool_mw", u"Batch Process", None))
    # retranslateUi

