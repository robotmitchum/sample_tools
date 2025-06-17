# coding:utf-8
"""
    :module: pseudo_stereo.py
    :description: Apply pseudo-stereo effect to audio
    :author: Michel 'Mitch' Pecqueur
    :date: 2024.05
"""

import random

import numpy as np
import soundfile as sf
import soxr
from scipy import signal

from common_audio_utils import rms, peak, db_to_lin, st_to_ms, ms_to_st, convolve, compensate_ir, get_silence_threshold
from common_math_utils import lerp
from fft_utils import filter_mask, h_cos


def pseudo_stereo(data: np.ndarray, sr: int = 44100, delay: int = 6, mode: str = 'velvet', seed: int = 0,
                  balance: bool = True,
                  ir_file: str | None = None, mx_len: bool = False, wet: float = 1.0,
                  cutoff: float | None = None, band: float = 100,
                  width: float = 1.0) -> np.ndarray:
    """
    Apply pseudo-stereo effect on audio

    :param data: Mono audio signal
    :param sr: Sampling rate

    :param delay: in ms
    :param mode: "haas" delay side channel or "velvet" convolve side channel with velvet impulse
    :param seed: Seed for Velvet noise

    :param balance: Attempt to balance stereo by detecting angle

    :param ir_file: IR sample to use with 'convolve' mode
    :param mx_len: Extends length to account for IR tail
    :param wet: Blend convolved result with original
    :param cutoff: Cutoff frequency for side filtering
    :param band: Frequency width of the side filter, use negative value for high pass instead of low pass
    :param width: Stereo width 0 for mono, value>1 amplifies stereo effect

    :return: Stereo audio
    """
    # Mid/Side process
    mid = data
    side = 0
    result = None

    if mode == 'convolve' and ir_file:
        ir, ir_sr = sf.read(ir_file)
        ir = compensate_ir(ir, mode='rms', sr=ir_sr)
        # Match IR to audio sampling rate
        if ir_sr != sr:
            # ir = librosa.resample(ir.T, orig_sr=ir_sr, target_sr=sr).T
            ir = soxr.resample(ir, in_rate=ir_sr, out_rate=sr)
            # target_len = np.round(len(ir) * sr / ir_sr).astype(np.int32)
            # ir = resample(ir, target_len, domain='time')

        conv = convolve(audio=data, ir=ir, wet=wet, mx_len=mx_len)
        (mid, side) = st_to_ms(conv).T
        result = ms_to_st(np.column_stack((mid, side * width)))
    elif mode == 'conv_side' and ir_file:
        ir, ir_sr = sf.read(ir_file)
        result = convolve_side(audio=data, sr=sr, ir_audio=ir, ir_sr=ir_sr, mx_len=mx_len, width=width)
    elif mode == 'haas':
        shift = int(round(sr * delay / 1000))
        side = np.pad(data, (shift, 0), mode='edge')[:data.size] * width
    elif mode == 'velvet':
        ir = velvet_ir(delay, sr, seed=seed)
        side = signal.fftconvolve(mid, ir, mode='full')[:len(mid)]
        nrm = rms(mid) / rms(side)
        side *= nrm * db_to_lin(-6) * width

    if not mode.startswith('conv'):
        # LP/HP filter side channel
        if cutoff is not None:
            pad = 256
            side = np.pad(side, pad_width=(pad, pad), mode='reflect')  # avoid popping with loops
            fft_side = np.fft.fft(side)
            n_fft = len(side)
            h_band = max(abs(band) / 2, 1e-5) * np.sign(band)
            mask = filter_mask(cutoff - h_band, cutoff + h_band, n_fft=n_fft, sr=sr)
            mask = np.exp(lerp(np.log(1e-5), np.log(1), h_cos(mask)))
            side = np.real(np.fft.ifft(fft_side * mask))
            side = np.real(side)[pad:len(mid) + pad]

        result = ms_to_st(np.column_stack((mid, side)))

        # Balance L/R
        if balance:
            # Use rms level from left and right channels as barycenter weights
            (l_chn, r_chn) = result.T
            wt = np.array([rms(l_chn), rms(r_chn)])
            wt /= sum(wt) / 2  # Normalize weights
            result /= wt

    # Normalize in case of clipping
    mx = peak(result)
    if mx > 1:
        result /= mx / db_to_lin(-.1)

    # Trim silence at end if needed
    silence_th = get_silence_threshold(bit_depth=24, as_db=True)
    result = trim_end(result, db=silence_th)

    return result


# Auxiliary functions


def velvet_ir(duration: int, sr: int, seed: int) -> np.ndarray:
    """
    Generate Velvet impulse response

    :param duration: in ms
    :param sr:
    :param seed: -1 to disable fixed seed
    :return: Generated Velvet impulse response
    """
    length = int(round(duration / 1000 * sr)) + 1

    if seed == -1:
        seed = random.randint(0, 99999)
        np.random.seed(seed)
    else:
        np.random.seed(seed)

    print(f'Random Seed: {seed}')

    k = 10
    indices = np.linspace(1, np.power(length, 1 / k), length // 20, endpoint=False) ** k
    indices += np.random.uniform(-1, 0, indices.size) * np.pad(np.diff(indices), (0, 1), mode='edge')
    indices = np.round(np.sort(indices)).astype(np.int32)
    indices[0] = 1
    indices = np.unique(indices)
    number = len(indices)

    signs = np.random.choice(np.array([-1, 1]), number, replace=True)
    signs[0] = 1

    result = np.zeros(length)
    result[indices] = signs
    dropoff = lerp(.1, .3, np.linspace(1, 0, length) ** 4) * np.random.normal(1, .25, length)
    dropoff[1] = 1

    result *= dropoff

    return result


def convolve_side(audio, sr, ir_audio, ir_sr, mx_len=False, width=1.0):
    """
    Convolve the side channel using the side channel of a given STEREO impulse response
    :param np.ndarray audio: Input audio
    :param int sr: Sampling rate
    :param np.array ir_audio: Impulse response
    :param int ir_sr: IR sampling rate (used for sampling rate matching)
    :param bool mx_len: Extend length so convolved tail is not cut
    :param float width: Stereo width
    :return:
    :rtype: np.ndarray
    """
    if audio.ndim > 1:
        mid = np.mean(audio, axis=-1)
    else:
        mid = np.copy(audio)

    ir_audio = compensate_ir(ir_audio, mode='rms', sr=ir_sr)

    (_, ir) = st_to_ms(ir_audio).T  # Use IR side channel

    if sr != ir_sr:
        # ir = librosa.resample(ir, orig_sr=ir_sr, target_sr=sr)
        ir = soxr.resample(ir, in_rate=ir_sr, out_rate=sr)

    au_l, ir_l = len(mid), len(ir)

    length = (au_l, au_l + ir_l)[mx_len]

    pad = length - au_l
    if pad > 0:
        mid = np.pad(mid, pad_width=(0, pad), mode='constant', constant_values=(0, 0))

    side = signal.fftconvolve(mid, ir, mode='full')[:length]

    ms = np.column_stack((mid, side * width))
    result = ms_to_st(ms)

    # Prevent clipping
    mx = peak(result)
    if mx > 1:
        result /= mx / db_to_lin(-.1)

    return result


def trim_end(data: np.ndarray, db: float = -60) -> np.ndarray:
    """
    Remove trailing silence at end
    :param data: Input audio
    :param db: Silence threshold in dB
    :return: processed audio
    """
    th = np.power(10, db / 20)
    silence = np.abs(data) < th
    idx = np.where(silence == 0)[0]
    result = data[:idx[-1] + 1]
    return result


# Extra functions


def meier_crossfeed(audio: np.ndarray, sr: int, strength: float = 0.5, delay: float = 0.3) -> np.ndarray:
    """
    Apply Meier cross-feed to a stereo audio signal
    :param audio:
    :param sr:
    :param strength: Effect strength
    :param delay:
    """
    delay_samples = int(sr * delay / 1000.0)

    # Low-pass filter coefficients for cross-feed
    # todo : doing LP filter this way creates click with looped audio, pad data (reflect) first or use FFT instead
    alpha = 0.5
    b = [alpha]
    a = [1, alpha - 1]

    result = np.zeros_like(audio)
    for i in range(2):
        shift = np.append(np.zeros(delay_samples), audio[:, 1 - i])[:-delay_samples]
        result[:, i] = audio[:, i] + signal.lfilter(b, a, shift) * strength

    # Normalize the output to prevent clipping
    mx = np.max(np.abs(result))
    if mx > 1.0:
        result /= mx

    return result


def rotate_stereo(audio: np.ndarray, rotate: float = 0) -> np.ndarray:
    """
    Stereo field rotation - from reaper jsfx
    :param audio:
    :param rotate: Rotation angle in degrees
    :return:
    """
    l_chn, r_chn = audio[:, 0], audio[:, 1]

    angle = np.arctan2(l_chn, r_chn)
    angle -= np.radians(rotate)
    radius = np.sqrt(l_chn ** 2 + r_chn ** 2)
    l_chn = np.sin(angle) * radius
    r_chn = np.cos(angle) * radius

    return np.column_stack((l_chn, r_chn))


def get_angle(audio: np.ndarray) -> np.ndarray:
    """
    Barycenter from volume used as angle cosine
    :param audio:
    :return:
    """
    nrm = np.sum(np.abs(audio), axis=1).reshape(len(audio), 1)
    nrm = np.repeat(nrm, 2, axis=1)
    wt = np.abs(audio) / (nrm + 1e-6)
    pnts = np.array([-1, 1]).reshape(1, 2)
    pnts = np.repeat(pnts, len(audio), axis=0)
    angle = np.arccos(np.sum(pnts * wt, axis=1)) - np.pi / 2
    return angle
