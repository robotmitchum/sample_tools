# coding:utf-8
"""
    :module: custom_bitmap_knobs.py
    :description: Render bitmap knobs atlases (vertical or horizontal strips)
    :author: Michel 'Mitch' Pecqueur
    :date: 2025.04
"""

from pathlib import Path

from color_utils import plt_to_rgba
from common_math_utils import lerp
from df_utils import *


def linear_slider(filepath: str | Path | None = 'linear_slider.png',
                  shape: tuple[int, int] = (40, 160), margin: float = 1, scl: float = 1,
                  bg_r: float | None = 4, fg_r: float | None = 4, dot_r: float | None = None,
                  bg_move: bool = True, start_end_hide: bool = True,
                  bg_color: str = 'ff000000', fg_color: str = 'ff7f7f7f', dot_color: str | None = 'ffffffff',
                  frames: int = 128) -> str | Path | np.ndarray:
    """
    Render a linear slider
    horizontal or vertical direction is determined from shape (width, height)

    :param filepath: Return array if None

    :param shape: width, height
    :param margin: Margin around the image in pixels
    :param scl: Scale factor

    :param bg_r: Background arc radius in pixels
    :param fg_r: Foreground arc radius in pixels
    :param dot_r: Dot radius in pixels

    :param bg_move: Bg follows fg to avoid fringing else remains static
    :param start_end_hide: Hide fg on 1st image and hide bg on last image

    :param bg_color: argb hex color
    :param fg_color:
    :param dot_color:

    :param frames: Number of animation frames, use odd values to get a proper middle step

    :return: Image path or array
    """

    shp = np.abs(np.array(shape) * scl).astype(np.uint).tolist()
    sx, sy = shp

    d = sx < sy
    h = min(sx, sy) / 2
    mx = max(sx, sy)

    slider_r = h - margin

    dot_color = (fg_color, dot_color)[bool(dot_color)]

    a, b = np.array((h, (h, mx - h)[d])) - .5, np.array(((mx - h, h)[d], h)) - .5

    result = []

    bg = np.zeros((sx, sy, 4))
    if bg_r and not bg_move:
        df = segment_df(shp, a, b) - min(bg_r * scl, slider_r)
        alpha = test_df(df, 0)
        bg = fill_alpha(alpha, plt_to_rgba(bg_color))

    for i in range(frames):
        w = i / (frames - 1)
        p = lerp(a, b, w)

        if bg_r and bg_move and fg_r and (True, bool(i < frames - 1))[start_end_hide]:
            df = segment_df(shp, p, b) - min(bg_r * scl, slider_r)
            alpha = test_df(df, 0)
            bg = fill_alpha(alpha, plt_to_rgba(bg_color))

        res = bg

        if fg_r and (True, bool(i))[start_end_hide]:
            df = segment_df(shp, a, p) - min(fg_r * scl, slider_r)
            alpha = test_df(df, 0)
            fg = fill_alpha(alpha, plt_to_rgba(fg_color))
            res = merge(fg, res)

        if dot_r:
            df = point_df(shp, p) - min(dot_r * scl, slider_r)
            alpha = test_df(df, 0)
            dot = fill_alpha(alpha, plt_to_rgba(dot_color))
            res = merge(dot, res)

        result.append(extend_edge(res))

    result = np.concatenate(result, axis=int(d))

    if not filepath:
        return result

    im = np_to_pil(result)
    im.save(str(filepath))
    return filepath


def rotary_knob(filepath: str | Path | None = 'rotary_knob.png',
                size: int = 64, margin: float = 1, gap: float = 90, scl: float = 1,
                bg_r: float | None = 1.5, fg_r: float | None = 1.5, dot_r: float | None = 4,
                bg_move: bool = True, start_end_hide: bool = True,
                bg_color: str = 'ff000000', fg_color: str = 'ff7f7f7f', dot_color: str | None = 'ffffffff',
                frames: int = 33, end: bool = True) -> str | Path | np.ndarray:
    """
    Render a rotary knob

    :param filepath: Return array if None

    :param size: Image size
    :param gap: Gap angle at the bottom of the knob
    :param margin: Margin around the image in pixels
    :param scl: Scale factor

    :param bg_r: Background arc radius in pixels
    :param fg_r: Foreground arc radius in pixels
    :param dot_r: Dot radius in pixels

    :param bg_move: Bg arc follows fg arc to avoid fringing else remains static
    :param start_end_hide: Hide fg on 1st image and hide bg on last image

    :param bg_color: argb hex color
    :param fg_color:
    :param dot_color:

    :param frames: Number of animation frames, use odd values to get a proper middle step
    :param end: Generate or skip end position

    :return: Image path or array
    """
    shape = np.abs(np.array((size, size)) * scl).astype(np.uint).tolist()
    sx, sy = shape

    p = np.array([sx / 2 - .5] * 2)

    knob_r = sx / 2 - max(dot_r or 0, fg_r or 0, bg_r or 0) * scl - margin

    dot_color = (fg_color, dot_color)[bool(dot_color)]

    bg = np.zeros((sx, sy, 4))
    if bg_r and not bg_move:
        df = arc_df(shape, p, rot=0, angle=gap / 2, r=knob_r) - bg_r * scl
        alpha = test_df(df)
        bg = fill_alpha(alpha, plt_to_rgba(bg_color))

    result = []
    for i in range(frames):
        w = i / (frames - int(end))

        value_bg = (1 - w) * (360 - gap)
        angle_bg = 180 - (value_bg / 2)
        rot_bg = gap / 2 - angle_bg

        if bg_r and bg_move and (True, bool(i < frames - 1))[start_end_hide]:
            df = arc_df(shape, p, rot=rot_bg, angle=angle_bg, r=knob_r) - bg_r * scl
            alpha = test_df(df)
            bg = fill_alpha(alpha, plt_to_rgba(bg_color))
        res = bg

        value_fg = w * (360 - gap)
        angle_fg = 180 - (value_fg / 2)
        rot_fg = angle_fg - gap / 2

        if fg_r and (True, bool(i))[start_end_hide]:
            df = arc_df(shape, p, rot=rot_fg, angle=angle_fg, r=knob_r) - fg_r * scl
            alpha = test_df(df)
            fg = fill_alpha(alpha, plt_to_rgba(fg_color))
            res = merge(fg, res)

        dot_angle = np.radians(value_fg + 90 + gap / 2)
        dot_p = np.array((np.cos(dot_angle), np.sin(dot_angle))) * knob_r + p

        if dot_r:
            df = point_df(shape, dot_p) - dot_r * scl
            alpha = test_df(df)
            dot = fill_alpha(alpha, plt_to_rgba(dot_color))
            res = merge(dot, res)

        result.append(extend_edge(res))

    result = np.concatenate(result, axis=0)

    if not filepath:
        return result

    im = np_to_pil(result)
    im.save(str(filepath))
    return filepath


def dial_knob(filepath: str | Path | None = 'dial_knob.png',
              size: int = 64, margin: float = 1, gap: float = 90, scl: float = 1,
              mark_p: tuple[float, float] = (16, 32),
              bg_r: float | None = None, bg_stroke_r: float = None, mark_r: float | None = 2,
              bg_color: str = 'ff000000', mark_color: str = 'ffffffff',
              frames: int = 33, end: bool = True) -> str | Path | np.ndarray:
    """
    Render a dial knob (Pan pot style)

    :param filepath: Return array if None

    :param size: Image size
    :param gap: Gap angle at the bottom of the knob
    :param margin: Margin around the image in pixels
    :param scl: Scale factor

    :param mark_p: Dial mark positions from center

    :param bg_r: Background dial radius in pixels
    :param bg_stroke_r: Background dial thickness radius in pixels
    :param mark_r: Mark radius in pixels

    :param bg_color: argb hex color
    :param mark_color:

    :param frames: Number of animation frames, use odd values to get a proper middle step
    :param end: Generate or skip end position

    :return: Image path or array
    """
    shape = np.abs(np.array((size, size)) * scl).astype(np.uint).tolist()
    sx, sy = shape

    p = np.array([sx / 2 - .5] * 2)

    mark_r = (mark_r or 0) * scl
    max_r = sx / 2 - margin
    if bg_r is not None:
        bg_r *= scl
    bg_r = min(((bg_r, max_r)[bg_r is None]), max_r)

    result = []
    for i in range(frames):
        w = i / (frames - int(end))

        df = point_df(shape, p) - bg_r
        if bg_stroke_r:
            df = stroke_sdf(df, bg_stroke_r)

        alpha = test_df(df, 0)
        res = fill_alpha(alpha, plt_to_rgba(bg_color))

        angle = w * (360 - gap)

        a_r, b_r = np.array(mark_p) * scl
        dot_angle = np.radians(angle + 90 + gap / 2)
        mark_dir = np.array((np.cos(dot_angle), np.sin(dot_angle)))

        a = p + mark_dir * min(a_r, max_r - mark_r)
        b = p + mark_dir * min(b_r, max_r - mark_r)

        if mark_r:
            df = segment_df(shape, a, b) - mark_r
            alpha = test_df(df, 0)
            fg = fill_alpha(alpha, plt_to_rgba(mark_color))
            res = merge(fg, res)

        result.append(extend_edge(res))

    result = np.concatenate(result, axis=0)

    if not filepath:
        return result

    im = np_to_pil(result)
    im.save(str(filepath))
    return filepath


def round_button(filepath: str | Path | None = 'round_button.png',
                 size: int = 64, margin: float = 1, scl: float = 1,
                 bg_r: float | None = None,
                 bg_color: str = 'ffffffff') -> str | Path | np.ndarray:
    """
    Render a basic round button

    :param filepath: Return array if None

    :param size: Image size
    :param margin: Margin around the image in pixels
    :param scl: Scale factor

    :param bg_r: radius in pixels

    :param bg_color: argb hex color

    :return: Image path or array
    """
    shape = np.abs(np.array((size, size)) * scl).astype(np.uint).tolist()
    sx, sy = shape

    p = np.array([sx / 2 - .5] * 2)

    max_r = sx / 2 - margin
    r = min((bg_r, max_r)[bg_r is None], max_r)

    df = point_df(shape, p) - r
    alpha = test_df(df, 0)

    result = extend_edge(fill_alpha(alpha, plt_to_rgba(bg_color)))

    if not filepath:
        return result

    im = np_to_pil(result)
    im.save(str(filepath))
    return filepath
