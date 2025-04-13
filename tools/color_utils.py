# coding:utf-8
"""
    :module: color_utils
    :description: Color functions

    :author: Michel 'Mitch' Pecqueur
    :date: 2024.07
"""
import colorsys
import os
from pathlib import Path

import numpy as np
import webcolors
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import interp1d

from common_math_utils import lerp, clamp, linstep


def basic_background(filepath, w=812, h=375, colors=([.1] * 3, [.3] * 3), gamma=2.2,
                     text=None, text_xy=(0, 0),
                     text_font=('HelveticaNeueThin.otf', 16), text_color=(1, 1, 1, 1), scl=1,
                     overwrite=True):
    """
    Create basic background image (vertical gradient)

    :param str or Path or None filepath: Return PIL image if no path provided
    :param int w: Width
    :param int h: Height

    :param tuple or list colors: list of RGB colors as 3 floats
    :param float gamma: Gamma to compensate when interpolating colors

    :param str or None text: Title to write in the top-left corner of the image
    :param tuple[float, float] text_xy: Text coordinates

    :param tuple text_font: Font name, Font size
    :param list or tuple text_color: RGBA color
    :param float scl: Font scaling

    :param bool overwrite: Overwrite background if present

    :return: Created image path or PIL Image object
    :rtype: str or Image
    """
    if filepath is not None:
        if not overwrite and Path(filepath).is_file():
            return filepath
        elif overwrite and Path(filepath).is_file():
            os.remove(filepath)

    num = len(colors)

    ramp = np.array(colors).reshape(-1, 1).reshape(num, 1, 3) ** gamma

    w, h = int(w * scl), int(h * scl)

    # Interpolate colors
    if num > 1:
        x = np.linspace(0, 1, num)
        x_new = np.linspace(0, 1, h)
        ramp = interp1d(x, ramp, kind='linear', axis=0)(x_new) ** (1 / gamma)
    else:
        ramp = np.repeat(ramp, h, axis=0)

    ramp = np.repeat(ramp, w, axis=1)  # Repeat horizontally
    ramp *= 255

    # Apply dithering
    if num > 1:
        dither = np.random.triangular(-1, 0, 1, (h, w, 3))
        ramp += dither * 2

    ramp = np.clip(ramp, 0, 255)

    im = Image.fromarray(np.round(ramp).astype(np.uint8), mode='RGB')
    im = write_text(im, text=text, text_xy=text_xy, text_font=text_font, text_color=text_color, scl=scl)

    if filepath:
        im.save(str(filepath), subsampling=0, quality=95)
        return filepath
    else:
        return im


def write_text(im, text=None, text_xy=(0, 0),
               text_font=('HelveticaNeueThin.otf', 16), text_color=(1, 1, 1, 1), max_length=None,
               align='left', anchor=None, stroke_width=0, scl=1):
    """
    Write text to a PIL Image

    :param Image im: PIL Image
    :param str or None text: Title to write in the top-left corner of the image
    :param tuple[float, float] text_xy: Text coordinates
    :param str align: 'left', 'right', 'center'
    :param str or None anchor: To use in combination of align
    https://pillow.readthedocs.io/en/stable/handbook/text-anchors.html
    :param float stroke_width:

    :param tuple text_font: Font name, Font size
    :param list or tuple text_color: RGBA color
    :param float or None max_length:
    :param float scl: Overall scaling
    :return:
    """
    if text:
        bg_color = list(text_color)[:-1] + [0]
        fill = tuple(int(v * 255) for v in text_color)
        bg_fill = tuple(int(v * 255) for v in bg_color)
        text_img = Image.new('RGBA', im.size, bg_fill)
        draw = ImageDraw.Draw(text_img)

        try:
            font = ImageFont.truetype(text_font[0], int(text_font[1] * scl))
            font_name = text_font[0]
        except Exception:
            try:
                # Seemingly supplied by default with PIL
                print(f'{text_font[0]} could not be found')
                font = ImageFont.truetype('DejaVuSans.ttf', int(text_font[1] * scl))
                font_name = 'DejaVuSans.ttf'
            except Exception:
                # Fallback font but can't be sized properly
                font = ImageFont.load_default()
                font_name = None

        text_xy = tuple(v * scl for v in text_xy)

        # Adjust font size if needed
        if max_length is not None and font_name is not None:
            fl = font.getlength(text)
            if fl > max_length * scl:
                factor = max_length * scl / fl
                new_size = text_font[1] * scl * factor
                font = ImageFont.truetype(font_name, new_size)

        draw.text(xy=text_xy, text=text, fill=fill, font=font, align=align, anchor=anchor,
                  stroke_width=stroke_width)
        im = Image.alpha_composite(im.convert('RGBA'), text_img).convert('RGB')

    return im


def limit_font_size(texts=(), text_font=('HelveticaNeueThin.otf', 16), max_length=64):
    """
    Get optimal font size required to match a minimum text length given a list of text
    :param list texts: List of texts
    :param tuple[str,float] text_font:
    :param float max_length:
    :return: Adjusted size
    :rtype: float
    """
    try:
        font = ImageFont.truetype(text_font[0], text_font[1])
    except Exception:
        try:
            # Seemingly supplied by default with PIL
            print(f'{text_font[0]} could not be found')
            font = ImageFont.truetype('DejaVuSans.ttf', text_font[1])
        except Exception:
            print('No suitable font found')
            return text_font[1]

    result = []
    for text in texts:
        tl = font.getlength(text)
        if tl > max_length:
            new_size = text_font[1] * max_length / tl
            result.append(new_size)
        else:
            result.append(text_font[1])

    return min(result)


def apply_symbol(im: Image, symbol_path: str | Path = '',
                 pos_xy: tuple[int, int] = (0, 0), size: tuple[int, int] = (32, 32),
                 color: tuple[float, float, float, float] = (1, 1, 1, 1), scl: float = 1) -> Image:
    """
    Composite symbol image over a background image
    :param im: PIL image input
    :param symbol_path:
    :param pos_xy:
    :param size:
    :param color: rgba color
    :param scl: Scale factor
    :return:
    :rtype: Image
    """
    scl_pos_xy = tuple(int(round(v * scl)) for v in pos_xy)
    scl_size = tuple(int(round(v * scl)) for v in size)
    symbol = Image.open(str(symbol_path)).convert('RGBA').resize(scl_size, Image.LANCZOS)
    symbol_color = tuple(int(v * 255) for v in color)
    bg_color = tuple(int(v * 255) for v in list(color)[:-1] + [0])
    overlay = Image.new('RGBA', im.size, bg_color)
    symbol_fill = Image.new('RGBA', symbol.size, symbol_color)
    overlay.paste(im=symbol_fill, box=scl_pos_xy, mask=symbol)
    return Image.alpha_composite(im.convert('RGBA'), overlay).convert('RGB')


def write_pil_image(filepath, im, quality=95):
    """
    :param str or Path filepath:
    :param Image im:
    :param int quality:
    :return: Image path
    :rtype sr: str
    """
    im.save(str(filepath), subsampling=0, quality=quality)
    return filepath


def basic_button(filepath, size=32, rgba=(1, 1, 1, 1), overwrite=True):
    """
    Render an anti-aliased circle

    :param str filepath:
    :param int size:
    :param list or tuple rgba:
    :param bool overwrite:

    :return:
    :rtype: str
    """
    if not overwrite and Path(filepath).is_file():
        return filepath
    elif overwrite and Path(filepath).is_file():
        os.remove(filepath)

    r = size / 2

    y, x = np.ogrid[:size, :size]
    cx, cy = size / 2 - .5, size / 2 - .5
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    alpha = 1 - linstep(r - 2, r - 1, d)

    img = np.zeros((size, size, 4))
    img[:, :] = np.array(rgba)
    img[:, :, -1] = alpha * rgba[-1]

    im = Image.fromarray(np.round(img * 255).astype(np.uint8), mode='RGBA')
    im.save(filepath)

    return filepath


def blank_button(filepath, size=32, overwrite=True):
    """
    Write a blank square image

    :param str filepath:
    :param int size:
    :param list or tuple rgba:
    :param bool overwrite:

    :return:
    :rtype: str
    """
    if not overwrite and Path(filepath).is_file():
        return filepath
    elif overwrite and Path(filepath).is_file():
        os.remove(filepath)

    wh = (size, size)
    im = Image.new(mode='RGBA', size=wh, color=(0, 0, 0, 0))
    im.save(filepath)

    return filepath


def adjust_palette(plt_data, adjust=(0, 1, 1)):
    """
    Recursively modify palette data
    :param dict or OrderedDict plt_data:
    :param adjust: HSV adjust
    :return:
    """
    for key, value in plt_data.items():
        if isinstance(value, dict):
            plt_data[key] = adjust_palette(value, adjust)
        else:
            argb = hex_to_rgba(value)
            argb[1:] = hsv_adjust(argb[1:], adjust=adjust)
            plt_data[key] = rgba_to_hex(argb)
    return plt_data


def hex_to_rgba(hexcolor: str = 'ff808080', shift: int = 0) -> tuple[float, float, float, float]:
    """
    NOTE : Decent Sampler uses argb and not rgba
    :param str hexcolor:
    :param shift: 1 rgba to argb, 0 unchanged, -1 argb to rgba
    :return: list of float
    """
    values = [int(hexcolor[i * 2:i * 2 + 2], 16) / 255 for i in range(len(hexcolor) // 2)]
    values = np.roll(values, shift).tolist()
    return values


def rgba_to_hex(rgba: tuple[float, float, float, float] = (1, 1, 1, 1), shift: int = 0) -> str:
    """
    NOTE : Decent Sampler uses argb and not rgba
    :param rgba: list of float values
    :param shift: 1 rgba to argb, 0 unchanged, -1 argb to rgba
    :return: Hex string
    """
    values = np.clip(np.round(np.roll(np.array(rgba), shift) * 255), 0, 255).astype(np.uint8).tolist()
    hexcolor = [hex(int(v))[2:].zfill(2) for v in values]
    return ''.join(hexcolor).upper()


def hsv_adjust(rgb=(1, 0, 0), adjust=(0, 1, 1)):
    """
    Adjust RGB color using HSV conversion
    :param list or tuple rgb:
    :param list or tuple adjust: hsv adjustment
    :return:
    :rtype: list
    """
    ha, sa, va = adjust
    lum = np.sum(np.array(rgb) * np.array([.3, .59, .11]))
    rgb = [lerp(lum, v, sa) for v in rgb]  # Saturation
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    hsv = [(h + ha) % 1, s, v * va]
    result = [clamp(value) for value in colorsys.hsv_to_rgb(*hsv)]
    return result


def plt_to_rgba(plt: str, rgba: tuple[float, float, float, float] = (1, 1, 1, 1),
                alpha: float = 1.0) -> tuple[int, int, int, int]:
    """
    Convert ds argb hex value to float rgba
    :param plt:
    :param rgba:
    :param alpha:
    """
    m = np.array(rgba, dtype=float)
    m[-1] = m[-1] * alpha
    return m * np.array(hex_to_rgba(plt, -1)).tolist()


def get_color_name(hexcolor='929edf'):
    """
    Translate a color hex value to the closest named web color
    :param str hexcolor:
    :return: Name
    :rtype: str
    """
    rgb = [int(hexcolor[i * 2:i * 2 + 2], 16) / 255 for i in range(len(hexcolor) // 2)]
    rgb = np.array(colorsys.rgb_to_hsv(*rgb))
    color_names = webcolors.CSS3_HEX_TO_NAMES
    result = ''
    mn = None
    for key in color_names.keys():
        hex_key = key[1:]
        key_rgb = [int(hex_key[i * 2:i * 2 + 2], 16) / 255 for i in range(len(hex_key) // 2)]
        key_rgb = np.array(colorsys.rgb_to_hsv(*key_rgb))
        d = np.sum((key_rgb - rgb) ** 2)
        if mn is None or d < mn:
            mn = d
            result = color_names[key]
    return result
