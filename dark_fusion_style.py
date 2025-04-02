# coding:utf-8
"""
    :module: dark_fusion_style.py
    :description: Minimalistic dark theme
    :author: Michel 'Mitch' Pecqueur
    :date: 2025.03
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QStyleFactory, QProxyStyle, QToolTip


class DarkFusionStyle(QProxyStyle):
    def __init__(self):
        super().__init__(QStyleFactory.create('Fusion'))  # Based on Fusion style

    def standardPalette(self):
        palette = QPalette()

        # Background
        palette.setColor(QPalette.Background, QColor(39, 39, 39))
        palette.setColor(QPalette.Window, QColor(63, 63, 63))
        palette.setColor(QPalette.WindowText, QColor(223, 223, 223))
        palette.setColor(QPalette.Base, QColor(39, 39, 39))
        palette.setColor(QPalette.AlternateBase, QColor(47, 47, 47))

        # Tool tip
        palette.setColor(QPalette.ToolTipBase, QColor(39, 39, 39))
        palette.setColor(QPalette.ToolTipText, Qt.white)

        # Text
        palette.setColor(QPalette.Button, QColor(71, 71, 71))
        palette.setColor(QPalette.ButtonText, QColor(223, 223, 223))
        palette.setColor(QPalette.Text, QColor(223, 223, 223))
        palette.setColor(QPalette.BrightText, Qt.white)

        # Highlight
        palette.setColor(QPalette.Link, QColor(31, 127, 159))
        palette.setColor(QPalette.Highlight, QColor(31, 127, 159))
        palette.setColor(QPalette.HighlightedText, Qt.black)

        # Disabled state
        palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(127, 127, 127))
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(127, 127, 127))
        palette.setColor(QPalette.Disabled, QPalette.Text, QColor(127, 127, 127))
        palette.setColor(QPalette.Disabled, QPalette.Link, QColor(127, 127, 127))
        palette.setColor(QPalette.Disabled, QPalette.Highlight, QColor(63, 63, 63))
        palette.setColor(QPalette.Disabled, QPalette.HighlightedText, QColor(127, 127, 127))

        return palette


def apply_dark_theme(widget):
    dark_style = DarkFusionStyle()
    dark_plt = dark_style.standardPalette()
    widget.setStyle(dark_style)
    widget.setPalette(dark_plt)

    QToolTip.setPalette(dark_plt)
