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

from common_math_utils import lerp, clamp


def basic_background(filepath: str | Path | None, w: int = 812, h: int = 375,
                     colors: tuple[list, ...] = ([.1] * 3, [.3] * 3), gamma: float = 2.2,
                     text: str | None = None, text_xy: tuple[float, float] = (0, 0),
                     text_font: tuple[str, float] = ('HelveticaNeueThin.otf', 16),
                     text_color: tuple[float, float, float, float] = (1, 1, 1, 1), scl: float = 1,
                     overwrite: bool = True) -> str | Path or Image:
    """
    Create basic background image (vertical gradient)

    :param filepath: Return PIL image if no path provided
    :param w: Width
    :param h: Height

    :param colors: list of RGB colors as 3 floats
    :param gamma: Gamma to compensate when interpolating colors

    :param text: Title to write in the top-left corner of the image
    :param text_xy: Text coordinates

    :param tuple text_font: Font name, Font size
    :param list or tuple text_color: RGBA color
    :param float scl: Font scaling

    :param overwrite: Overwrite background if present

    :return: Created image path or PIL Image object
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


def write_text(im: Image, text: str | None = None, text_xy: tuple[float, float] = (0, 0),
               text_font: tuple[str, float] = ('HelveticaNeueThin.otf', 16),
               text_color: tuple[float, float, float, float] = (1, 1, 1, 1), max_length: float | None = None,
               align: str = 'left', anchor: str | None = None, stroke_width: float = 0, scl: float = 1) -> Image:
    """
    Write text to a PIL Image

    :param im: PIL Image
    :param str or None text: Title to write in the top-left corner of the image
    :param text_xy: Text coordinates
    :param align: 'left', 'right', 'center'
    :param anchor: To use in combination of align
    https://pillow.readthedocs.io/en/stable/handbook/text-anchors.html
    :param stroke_width:

    :param text_font: Font name, Font size
    :param text_color: RGBA color
    :param max_length:
    :param scl: Overall scaling
    :return: Image
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


def limit_font_size(texts: list[str] = (), text_font: tuple[str, float] = ('HelveticaNeueThin.otf', 16),
                    max_length: float | None = 64) -> float:
    """
    Get optimal font size required to match a minimum text length given a list of text
    :param texts: List of texts
    :param text_font:
    :param max_length:
    :return: Adjusted size
    """
    try:
        font = ImageFont.truetype(text_font[0], text_font[1])
    except Exception:
        try:
            print(f'{text_font[0]} could not be found')
            # Seemingly supplied by default with PIL
            font = ImageFont.truetype('DejaVuSans.ttf', text_font[1])
        except Exception:
            print('No suitable font found')
            return text_font[1]

    result = []
    mx = 0
    for text in texts:
        tl = font.getlength(text)
        if max_length is None:
            if tl > mx:
                mx = tl
        else:
            if tl > max_length:
                new_size = text_font[1] * max_length / tl
                result.append(new_size)
            else:
                result.append(text_font[1])

    if max_length is None:
        return mx

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


def write_pil_image(filepath: str | Path, im: Image, quality: int = 95) -> str | Path:
    """
    :param filepath:
    :param im:
    :param quality:
    :return: Image path
    """
    im.save(str(filepath), subsampling=0, quality=quality)
    return filepath


def blank_button(filepath: str | Path, size: int = 32, overwrite: bool = True) -> str | Path:
    """
    Write a blank square image

    :param filepath:
    :param size:
    :param overwrite:

    :return: File path
    """
    if not overwrite and Path(filepath).is_file():
        return filepath
    elif overwrite and Path(filepath).is_file():
        os.remove(filepath)

    wh = (size, size)
    im = Image.new(mode='RGBA', size=wh, color=(0, 0, 0, 0))
    im.save(str(filepath))

    return filepath


def adjust_palette(plt_data: dict, adjust: tuple[float, float, float] = (0, 1, 1)) -> dict:
    """
    Recursively modify palette data
    :param plt_data: Source palette data
    :param adjust: HSV adjust
    :return: Adjusted palette data
    """
    for key, value in plt_data.items():
        if isinstance(value, dict):
            plt_data[key] = adjust_palette(value, adjust)
        else:
            argb = hex_to_rgba(value)
            argb[1:] = hsv_adjust(argb[1:], adjust=adjust)
            plt_data[key] = rgba_to_hex(argb)
    return plt_data


def hex_to_rgba(hexcolor: str = 'ff808080', shift: int = 0) -> tuple:
    """
    NOTE : Decent Sampler uses argb and not rgba
    :param str hexcolor:
    :param shift: 1 rgba to argb, 0 unchanged, -1 argb to rgba
    :return: list of float
    """
    values = [int(hexcolor[i * 2:i * 2 + 2], 16) / 255 for i in range(len(hexcolor) // 2)]
    values = np.roll(values, shift).tolist()
    return values


def rgba_to_hex(rgba: tuple[float, ...] = (1, 1, 1, 1), shift: int = 0) -> str:
    """
    NOTE : Decent Sampler uses argb and not rgba
    :param rgba: list of float values
    :param shift: 1 rgba to argb, 0 unchanged, -1 argb to rgba
    :return: Hex string
    """
    values = np.clip(np.round(np.roll(np.array(rgba), shift) * 255), 0, 255).astype(np.uint8).tolist()
    hexcolor = [hex(int(v))[2:].zfill(2) for v in values]
    return ''.join(hexcolor).upper()


def hsv_adjust(rgb: tuple[float] = (1, 0, 0),
               adjust: tuple[float, float, float] = (0, 1, 1)) -> list[float]:
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


def get_color_name(hexcolor: str = '929edf') -> str:
    """
    Translate a color hex value to the closest named web color
    :param hexcolor:
    :return: Name
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
