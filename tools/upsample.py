# coding:utf-8
"""
    :module: upsampler.py
    :description: Audio up-resolution by replicating upper frequency band

    Vaguely inspired by SBR
    https://en.wikipedia.org/wiki/Spectral_band_replication

    Requires rubberband executable to perform pitch shifting
    https://breakfastquay.com/rubberband/

    :author: Michel 'Mitch' Pecqueur
    :date: 2024.07
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import pyrubberband as pyrb
import soundfile as sf
from scipy.interpolate import interp1d
from scipy.signal import resample
from scipy.stats import linregress
from pathlib import Path

from common_audio_utils import lin_to_db, db_to_lin
from common_math_utils import lerp


def audio_upsample(input_file: Path | str | None, output_file: Path | str | None, audio: np.ndarray | None = None,
                   sr: int | None = None, f_max: float | None = None, target_sr: int = 48000,
                   mix: float = 1.0) -> np.ndarray | Path:
    """
    Audio up-resolution by replicating upper frequency band

    :param str or None input_file:
    :param str or None output_file:
    :param np.array audio:
    :param int sr: Input sampling rate
    :param float or None f_max: Upper frequency threshold before up-sampling
    :param int target_sr: Desired output sampling rate
    :param float mix:

    :return:
    :rtype: np.array
    """
    if input_file:
        audio, sr = sf.read(input_file)

    nch = audio.ndim

    if nch > 1:
        mono_audio = np.mean(audio, axis=-1)
    else:
        mono_audio = audio

    data = np.copy(audio)

    if f_max is not None and f_max < sr / 2:
        div = (f_max * 2) / sr
    else:
        div = 1

    # func = (lambda x: x)
    func = fade_func

    iterations = math.ceil(math.log2(target_sr / (sr * div)))
    print(f'Iterations : {iterations}')

    fft_base = np.fft.fft(mono_audio)
    mags = np.abs(fft_base)
    n_fft = len(mono_audio)
    freqs = np.fft.fftfreq(len(mono_audio), 1 / (sr * div))
    st_idx = int(div * n_fft / 4)
    ed_idx = int(div * n_fft / 2)

    # Calculate the replicated band amplitude
    reg_x = freqs[st_idx:ed_idx]
    rep_m = mags[st_idx:ed_idx]
    lin_reg = linregress(x=reg_x, y=lin_to_db(rep_m))
    reg_y = lin_reg.slope * reg_x[0:-1] + lin_reg.intercept
    factor = db_to_lin(reg_y[-1] - reg_y[0])

    factor = min(2, factor)
    factor *= mix

    print(f'Factor: {factor}')

    target_len = np.round(len(data) * target_sr / sr).astype(np.int32)
    data = resample(data, target_len, domain='time')
    # duration = target_len / target_sr

    n_fft = len(data)
    cutoff = div * sr / 2
    band_width = cutoff / 2

    # Band weights
    bw = [.25]
    bw.extend([1 for _ in range(iterations)])

    result = np.zeros_like(data)
    band = band_width * bw[0]
    lp_mask = func(filter_mask(cutoff, cutoff - band, n_fft, target_sr))
    for c in range(nch):
        if nch > 1:
            chn_data = data[:, c]
        else:
            chn_data = data
        fft_base = np.fft.fft(chn_data)
        chn_result = np.real(np.fft.ifft(fft_base * lp_mask))
        if nch > 1:
            result[:, c] = chn_result
        else:
            result = chn_result

    for c in range(nch):
        if nch > 1:
            chn_data = data[:, c]
        else:
            chn_data = data

        m = factor
        for i in range(iterations):
            oct_m = 2 ** i

            pitched_up = pyrb.pitch_shift(y=chn_data, sr=target_sr, n_steps=12 * oct_m)

            cutoff = (i + 1) * div * sr / 2

            lp_band = band_width * bw[i + 1]
            lp_mask = func(filter_mask(cutoff * 2, cutoff * 2 - lp_band, n_fft, target_sr))
            hp_band = band_width * bw[i]
            hp_mask = func(filter_mask(cutoff - hp_band, cutoff, n_fft, target_sr))

            ramp_w = .5 * mix
            prev_factor, next_factor = lerp(1, 1 / factor, ramp_w), lerp(1, factor, ramp_w)
            ramp = filter_ramp(x=[cutoff * .75, cutoff * 3], y=[prev_factor, next_factor],
                               n_fft=n_fft, sr=target_sr)

            fft_up = np.fft.fft(pitched_up)
            chn_result = np.real(np.fft.ifft(fft_up * lp_mask * hp_mask * ramp)) * m
            m *= m

            if nch > 1:
                result[:, c] += chn_result
            else:
                result += chn_result

    mx = np.max(np.abs(result))
    if mx > 1:
        result /= mx

    if output_file is not None:
        cmp = ({}, {'compression_level': 1.0})[output_file.suffix == '.flac']
        sf.write(str(output_file), result, samplerate=target_sr, **cmp)

    return result


def filter_mask(a, b, n_fft, sr):
    """
    Generate a linear ramp array between 0 and 1 to perform a filter using FFT transform
    Run result an interpolating function to modify gradient shape
    :param float a: Min frequency (0)
    :param float b: Max frequency (1)
    :param int n_fft: FFT length
    :param int sr: Sampling rate
    :return: Mask matching FFT length
    :rtype: np.array
    """
    x = np.abs((np.fft.fftfreq(n_fft, 1 / sr)))
    return np.clip((x - a) / (b - a), 0.0, 1.0)


def filter_ramp(x, y, n_fft, sr):
    """
    Generate an interpolated ramp with an arbitrary number of points with their corresponding  values
    :param list or np.array x: frequency coordinates
    :param list or np.array y: Value corresponding to frequency
    :param int n_fft: FFT length
    :param int sr: Sampling rate
    :return: Mask matching FFT length
    :rtype: np.array
    """
    freqs = np.abs((np.fft.fftfreq(n_fft, 1 / sr)))
    mn, mx = min(freqs), max(freqs)
    if mn not in x:
        x.insert(0, mn)
        y.insert(0, y[0])
    if mx not in x:
        x.append(mx)
        y.append(y[-1])
    ramp = interp1d(x, y, kind='linear')(freqs)
    # n = int(n_fft / len(x) / 2)
    # k = np.ones(n) / n
    # ramp = np.convolve(np.pad(ramp, n // 2, mode='edge'), k, mode='same')[n // 2:-n // 2 + 1]
    return ramp


def fade_func(x):
    a = 1 - (1 - x) ** 2
    return a * a * (2.0 - a)


# Statitics plotting

def plot_spectrum(input_file, audio=None, sr=None):
    if input_file:
        audio, sr = sf.read(input_file)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    n_fft = len(audio)
    dt = 1 / sr
    fft_in = np.fft.fft(audio)

    mags = np.abs(fft_in)

    freqs = np.fft.fftfreq(n_fft, dt)
    up_freqs = np.fft.fftfreq(n_fft * 2, 1 / (sr * 2))

    duration = n_fft / sr

    lim = sr / 2
    # lim = 11025 / 2
    start_idx = int(lim * .5 * duration)
    max_idx = int(lim * duration)
    up_max_idx = int(sr * duration)

    rep_m = mags[start_idx:max_idx]
    reg_x = freqs[start_idx:max_idx]
    lin_reg = linregress(x=reg_x, y=lin_to_db(rep_m))
    reg_y = lin_reg.slope * reg_x[0:-1] + lin_reg.intercept
    factor = db_to_lin(reg_y[-1] - reg_y[0])
    # factor = 1
    print(f'Factor: {factor}')

    plt.figure()
    plt.title('Band replication')
    plt.plot(up_freqs[:max_idx], mags[:max_idx])
    plt.plot(up_freqs[max_idx:up_max_idx:2][:len(rep_m)], rep_m * factor)

    x = up_freqs[start_idx:up_max_idx]
    reg_y = lin_reg.slope * x + lin_reg.intercept
    plt.plot(x, db_to_lin(reg_y), 'r', label='Fitted Line')

    plt.xlabel('Freq')
    plt.ylabel('Amplitude')
    # plt.xscale('log')
    plt.yscale('log')

    # plt.ylim(0, 1)
    plt.xlim(0, lim * 2)

    plt.legend(['signal', 'Replicated Signal', 'Slope', ])
    plt.show()
