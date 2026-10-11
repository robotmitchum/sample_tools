# coding:utf-8
"""
    :module: dark_fusion_style.py
    :description: Minimalistic dark theme
    :author: Michel 'Mitch' Pecqueur
    :date: 2025.03
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QStyleFactory, QProxyStyle, QToolTip


class DarkFusionStyle(QProxyStyle):
    def __init__(self):
        super().__init__(QStyleFactory.create('Fusion'))  # Based on Fusion style

    def standardPalette(self):
        palette = QPalette()

        # Background
        # palette.setColor(QPalette.ColorRole.Background, QColor(39, 39, 39))  # PySide6 Equivalent?
        palette.setColor(QPalette.ColorRole.Window, QColor(63, 63, 63))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(223, 223, 223))
        palette.setColor(QPalette.ColorRole.Base, QColor(39, 39, 39))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(47, 47, 47))

        # Tool tip
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(39, 39, 39))
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)

        # Text
        palette.setColor(QPalette.ColorRole.Button, QColor(71, 71, 71))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(223, 223, 223))
        palette.setColor(QPalette.ColorRole.Text, QColor(223, 223, 223))
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.white)

        # Highlight
        palette.setColor(QPalette.ColorRole.Link, QColor(31, 127, 159))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(31, 127, 159))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)

        # Disabled state
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(127, 127, 127))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(127, 127, 127))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(127, 127, 127))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Link, QColor(127, 127, 127))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, QColor(63, 63, 63))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.HighlightedText, QColor(127, 127, 127))

        return palette


def apply_dark_theme(widget):
    dark_style = DarkFusionStyle()
    dark_plt = dark_style.standardPalette()
    widget.setStyle(dark_style)
    widget.setPalette(dark_plt)

    QToolTip.setPalette(dark_plt)

    # Also style QMessageBox and QFileDialog buttons
    style = ''
    for wid_type in ('QMessageBox', 'QFileDialog'):
        style += f"""
            {wid_type} QPushButton {{
                background-color: #474747;
                color: #dfdfdf;
                border: 1px solid #5c5c5c;
                padding: 5px 10px;
                border-radius: 4px;
                min-width: 80px;
            }}
            {wid_type} QPushButton:hover {{
                background-color: #5e5e5e;
            }}
            {wid_type} QPushButton:pressed {{
                background-color: #3d3d3d;
            }}
        """
    widget.setStyleSheet(style)
