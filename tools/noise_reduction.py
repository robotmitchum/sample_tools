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
from pathlib import Path


def generate_quantize_noise(output_file: Path | str | None = None, sr: int = 48000, length: int | None = None,
                            bd: int = 8, seed: int = 0) -> np.ndarray:
    """
    Generate quantization noise profile

    :param output_file:
    :param sr: Sampling Rate
    :param length: Length in samples
    :param bd: Bit-depth
    :param seed: Noise generator seed

    :return:
    """
    p = Path(output_file)

    mx = 2 ** (bd - 1)

    if length is None:
        length = sr
    np.random.seed(seed)
    result = np.random.uniform(-1, 1, length) / mx

    if output_file:
        cmp = ({}, {'compression_level': 1.0})[p.suffix == '.flac']
        sf.write(str(output_file), result, sr, subtype='PCM_24', **cmp)

    return result


def denoise(audio: np.ndarray, noise_profile: np.ndarray, sr: int, mix: float = 1.0,
            normalize: bool = True) -> np.ndarray:
    """
    Spectral gate noise reduction using reduce noise, basically just a more convenient re-wrap

    :param audio:
    :param noise_profile:
    :param sr:
    :param mix:
    :param normalize:

    :return:
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


def declip(audio: np.ndarray, db: float = -.1, threshold: float | None = None, mix: float = .25,
           normalize: bool = False) -> np.ndarray:
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
