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

    :param str filepath:
    :param int w: Width
    :param int h: Height

    :param tuple or list colors: list of RGB colors as 3 floats
    :param float gamma: Gamma to compensate when interpolating colors

    :param str or None text: Title to write in the top-left corner of the image
    :param tuple(float,float) text_xy: Text coordinates

    :param tuple text_font: Font name, Font size
    :param list or tuple text_color: RGBA color
    :param float scl: Font scaling

    :param bool overwrite: Overwrite background if present

    :return: Created image
    :rtype: str
    """
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

    if text:
        bg_color = list(text_color)[:-1] + [0]
        fill = tuple(int(v * 255) for v in text_color)
        bg_fill = tuple(int(v * 255) for v in bg_color)
        text_img = Image.new('RGBA', im.size, bg_fill)
        draw = ImageDraw.Draw(text_img)

        try:
            font = ImageFont.truetype(text_font[0], int(text_font[1] * scl))
        except Exception:
            try:
                # Seemingly supplied by default with PIL
                print(f'{text_font[0]} could not be found')
                font = ImageFont.truetype('DejaVuSans.ttf', int(text_font[1] * scl))
            except Exception:
                # Fallback font but can't be sized properly
                font = ImageFont.load_default()

        text_xy = tuple(v * scl for v in text_xy)
        draw.text(xy=text_xy, text=text, fill=fill, font=font)
        im = Image.alpha_composite(im.convert("RGBA"), text_img).convert('RGB')

    im.save(filepath, subsampling=0, quality=95)

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


def hex_to_rgba(hexcolor='ff808080'):
    """
    NOTE : Decent Sampler uses argb and not rgba
    :param str hexcolor:
    :return: list of float
    :rtype: list of float values
    """
    rgba = [int(hexcolor[i * 2:i * 2 + 2], 16) / 255 for i in range(len(hexcolor) // 2)]
    return rgba


def rgba_to_hex(rgba=(1, 1, 1, 1)):
    """
    NOTE : Decent Sampler uses argb and not rgba
    :param list or tuple rgba: list of float values
    :return: Hex string
    :rtype: str
    """
    hexcolor = [hex(int(clamp(v * 255, 0, 255)))[2:].zfill(2) for v in rgba]
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

# basic_button(filepath='button_test.png', size=32, rgba=(1, .75, .5, 1))
