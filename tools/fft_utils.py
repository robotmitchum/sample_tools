# coding:utf-8
"""
    :module: fft_utils.py
    :description: Utility functions to generate FFT masks
    :author: Michel 'Mitch' Pecqueur
    :date: 2024.08
"""

import numpy as np
from scipy.interpolate import interp1d


def filter_mask(a, b, n_fft, sr):
    """
    Generate a linear ramp array between 0 and 1 to perform a filter using FFT transform

    :param float a: Min frequency (0)
    :param float b: Max frequency (1)
    :param int n_fft: FFT length
    :param int sr: Sampling rate

    :return: Mask matching FFT length
    :rtype: np.array
    """
    x = np.abs((np.fft.fftfreq(n_fft, 1 / sr)))
    return np.clip((x - a) / (b - a), 0.0, 1.0)


def filter_ramp(x, y, n_fft, sr, blur=None):
    """
    Generate an interpolated ramp with an arbitrary number of points with their corresponding  values

    :param list or np.array x: frequency coordinates
    :param list or np.array y: Value corresponding to frequency
    :param int n_fft: FFT length
    :param int sr: Sampling rate
    :param float or None blur: Box blur radius

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

    if blur:
        n_k = int(blur * n_fft / len(x) / 2)
        kernel = np.ones(n_k) / n_k
        ramp = np.convolve(np.pad(ramp, n_k // 2, mode='edge'), kernel, mode='same')[n_k // 2:-n_k // 2 + 1]

    return ramp


# Interpolation functions

def h_cos(x):
    """
    Half cosine
    :param x:
    :return:
    """
    return .5 - .5 * np.cos(x * np.pi)


def q_cos(x):
    """
    Quadratic cos approximation 'Equal power' fade-in
    :param x:
    :return:
    """
    return x * (2 - x)


def q_log(x):
    """
    Quartic 'log' approximation, Pseudo-log fade-in
    :param x:
    :return:
    """
    return 1 - (1 - x) ** 4


def q_exp(x):
    """
    Quartic 'exp' approximation, Pseudo-exp fade-out
    :param x:
    :return:
    """
    return x ** 4
