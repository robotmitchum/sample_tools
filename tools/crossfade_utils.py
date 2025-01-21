# coding:utf-8
"""
    :module: crossfade_utils.py
    :description:
    :author: Michel 'Mitch' Pecqueur
    :date: 2024.08
"""

import numpy as np

from common_audio_utils import pad_audio
from common_math_utils import equal_power_sine, smoothstep


def crossfade_clips(audio_a, audio_b, start, fade_len, fade_type='equal_power'):
    """
    Cross-fade two audio arrays

    :param np.array audio_a:
    :param np.array audio_b:
    :param int start: Start of fade
    :param int fade_len: Fade length
    :param str fade_type: 'equal_power' or 'linear'

    :return:
    """
    nch = audio_a.ndim

    start = min(start, len(audio_a) - 1)
    fade_len = min(fade_len, len(audio_a) - start, len(audio_b))

    fade = np.zeros(start)
    fade = np.append(fade, np.linspace(0, 1, fade_len))
    fade_pad = max(0, len(audio_b) - fade_len)

    if fade_pad > 0:
        fade = np.append(fade, np.ones(fade_pad))

    pad = len(fade) - len(audio_a)

    if pad > 0:
        audio_a = pad_audio(audio_a, 0, pad)
    else:
        audio_a = np.copy(audio_a[:len(fade)])

    if start > 0:
        audio_b = pad_audio(audio_b, start, 0)

    fd = [1 - fade, fade]

    match fade_type:
        case 'equal_power':
            fd = [equal_power_sine(x) for x in fd]
        case 'smoothstep':
            fd = [smoothstep(0, 1, x) for x in fd]

    if nch > 1:
        fd = [np.repeat(x.reshape(len(x), 1), nch, axis=1) for x in fd]

    result = audio_a * fd[0] + audio_b * fd[1]

    return result
