# coding:utf-8
"""
    :module: noise_reduction
    :description: Spectral gate noise reduction using reduce noise
    :author: Michel 'Mitch' Pecqueur
    :date: 2024.07
"""

import numpy as np
import soundfile as sf
from noisereduce import reduce_noise
from scipy.interpolate import interp1d
from common_audio_utils import db_to_lin
from common_math_utils import lerp


def generate_quantize_noise(output_file=None, sr=48000, length=None, bd=8, seed=0):
    """
    Generate quantization noise profile

    :param str or None output_file:
    :param int sr: Sampling Rate
    :param int or None length: Length in samples
    :param int bd: Bit-depth
    :param int seed: Noise generator seed

    :return:
    :rtype: np.array
    """
    mx = 2 ** (bd - 1)

    if length is None:
        length = sr
    np.random.seed(seed)
    result = np.random.uniform(-1, 1, length) / mx

    if output_file:
        sf.write(output_file, result, sr, subtype='PCM_24')

    return result


def denoise(audio, noise_profile, sr, mix=1.0, normalize=True):
    """
    Spectral gate noise reduction using reduce noise, basically just a more convenient re-wrap

    :param np.array audio:
    :param np.array noise_profile:
    :param int sr:
    :param float mix:
    :param bool normalize:

    :return:
    :rtype: np.array
    """
    pad = 256

    result = np.zeros_like(audio)
    nch = audio.ndim

    for c in range(nch):
        if nch > 1:
            chn_data = audio[:, c]
        else:
            chn_data = audio

        chn_result = np.pad(np.copy(chn_data), pad_width=(pad, pad), mode='reflect')
        chn_result = reduce_noise(y=chn_result, sr=sr, stationary=True, y_noise=noise_profile, prop_decrease=mix,
                                  use_torch=False)
        chn_result = chn_result[pad:len(chn_data) + pad]

        if nch > 1:
            result[:, c] += chn_result
        else:
            result += chn_result

    mx = np.max(np.abs(result))
    if mx > 1 or normalize:
        result /= mx

    return result


def declip(audio, db=-.1, threshold=None, mix=.25, normalize=False):
    """
    Simplistic audio de-clipping
    Replace samples over a given threshold by cubic interpolation

    :param np.array audio:
    :param float db: in dB
    :param float or None threshold: Threshold as a linear factor
    :param float mix: Mix result with original
    :param bool normalize:

    :return:
    :rtype: np.array
    """
    if threshold is None:
        threshold = db_to_lin(db)

    result = np.zeros_like(audio)
    nch = audio.ndim

    for c in range(nch):
        if nch > 1:
            chn_data = audio[:, c]
        else:
            chn_data = audio

        indices = np.argwhere(np.abs(chn_data) <= threshold).reshape(-1)
        values = chn_data[indices]
        if 0 not in indices:
            indices = np.append([0], indices)
            values = np.append([0], values)
        if len(chn_data) - 1 not in indices:
            indices = np.append(indices, len(chn_data) - 1)
            values = np.append(values, [0])
        tgt_idx = np.arange(len(chn_data))
        chn_result = interp1d(indices, values, kind='cubic')(tgt_idx)
        chn_result = lerp(chn_data, chn_result, mix)

        if nch > 1:
            result[:, c] += chn_result
        else:
            result += chn_result

    if normalize:
        result /= np.max(np.abs(result))

    return result
