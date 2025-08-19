# coding:utf-8
"""
    :module: sfz_bg.py
    :description:
    :author: Michel 'Mitch' Pecqueur
    :date: 2025.08
"""

import os
from pathlib import Path

from color_utils import adjust_palette, hex_to_rgba, plt_to_rgba
from color_utils import basic_background, write_text, apply_symbol, limit_font_size, write_pil_image
from jsonFile import read_json


def sfz_bg(filepath: Path | str, w: int = 775, h: int = 335,
           color_plt_cfg: str = 'plt_cfg/Dark_plt_cfg.json',
           plt_adjust: tuple[float, float, float] = (0, 1, 1),

           top_text: str | None = None,
           center_text: str | None = None,
           bottom_text: str | None = None,

           text_font: tuple[str, int] = ('HelveticaNeueThin.otf', 32),

           scl=1.0,
           ) -> Path:
    """
    Write a background image for Sfizz-ui
    Stripped down version of dspreset background image

    :param filepath:
    :param w:
    :param h:
    :param color_plt_cfg:
    :param plt_adjust:

    :param top_text: Instrument name
    :param center_text: Info text such as credits
    :param bottom_text: Extra info

    :param text_font:
    :param scl:
    :return:
    """
    current_dir = Path(__file__).parent

    # Color Palette Config
    plt_data = read_json(color_plt_cfg) or {}
    if plt_adjust:
        plt_data = adjust_palette(plt_data=plt_data, adjust=plt_adjust)

    text_plt = list(plt_data['textColor'].values())
    bg_plt = plt_data['backgroundRamp']

    if 'backgroundText' in plt_data:
        bg_text_plt = list(plt_data['backgroundText'].values())
    else:
        bg_text_plt = [text_plt[0], '80' + text_plt[0][2:]]

    # Create background image
    bg_fmt = ('png', 'jpg')[len(bg_plt.values()) > 1]  # Use jpg only for gradients
    bg_path = Path(filepath).with_suffix(f'.{bg_fmt}')

    resources_dir = bg_path.parent
    if not resources_dir.exists():
        os.makedirs(resources_dir, exist_ok=True)

    bg_colors = [hex_to_rgba(h)[1:] for h in bg_plt.values()]
    bg_img = basic_background(filepath=None, w=w, h=h, scl=scl, overwrite=True, colors=bg_colors, gamma=1.0,
                              text_font=text_font, text_color=plt_to_rgba(bg_text_plt[0]))

    if top_text:
        text_xy = (w / 2, 15)
        bg_img = write_text(im=bg_img, text=top_text, text_xy=text_xy, align='center', anchor='mt',
                            text_font=text_font, text_color=plt_to_rgba(bg_text_plt[0]), scl=scl)

    if center_text:
        info_lines = [line for line in center_text.split('\n') if line]

        max_length = w / 2

        text_length = limit_font_size(texts=info_lines, text_font=text_font, max_length=None)
        center_font_size = limit_font_size(texts=info_lines, text_font=text_font, max_length=max_length)

        center_font = (text_font[0], center_font_size)

        info_w = int(center_font_size)
        symbol_path = current_dir / 'symbols/info.png'

        text_xy = (w / 2, h / 2)
        symbol_xy = tuple(int(round(a - b - c)) for a, b, c in
                          zip(text_xy, (min(max_length, text_length) / 2, 0), (info_w * 1.5, info_w / 2)))

        bg_img = apply_symbol(im=bg_img, symbol_path=symbol_path, pos_xy=symbol_xy, size=(info_w, info_w),
                              color=plt_to_rgba(bg_text_plt[-1], alpha=.5), scl=scl)

        bg_img = write_text(im=bg_img, text=center_text.rstrip('\n'), text_xy=text_xy, align='center', anchor='mm',
                            text_font=center_font, text_color=plt_to_rgba(bg_text_plt[-1], alpha=1), scl=scl)

    if bottom_text:
        text_xy = (w / 2, h - 15)
        bottom_font = (text_font[0], text_font[1] * .75)
        bg_img = write_text(im=bg_img, text=bottom_text, text_xy=text_xy, align='center', anchor='mb',
                            text_font=bottom_font, text_color=plt_to_rgba(bg_text_plt[0], alpha=1), scl=scl)

    write_pil_image(bg_path, im=bg_img, quality=95)

    return bg_path

# sfz_bg(filepath='sfz_bg_test', info_text="Samples - Synclavier Sampler Library\nUI - Michel 'Mitch' Pecqueur",
#        bg_text='Harp Sus', color_plt_cfg='../plt_cfg/Dark_plt_cfg.json')
