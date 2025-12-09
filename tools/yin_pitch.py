# coding:utf-8
"""
    :module: yin_pitch.py
    :description: Drop-in replacement for librosa.yin
    more compact and doesn't require numba
    Made from code extracted from librosa
    This does not give the exact same output, but it's close enough
    :author: Michel 'Mitch' Pecqueur
    :date: 2025.06
"""

from typing import Optional, Union, Sequence

import numpy as np
from numpy.lib.stride_tricks import as_strided


class ParameterError(ValueError):
    """Exception raised for invalid parameter values"""
    pass


def yin(y: np.ndarray, *, fmin: float, fmax: float, sr: int = 22050,
        frame_length: int = 2048, win_length: Optional[int] = None, hop_length: Optional[int] = None,
        trough_threshold: float = 0.1, center: bool = True, pad_mode: str = "constant") -> np.ndarray:
    """
    Estimate pitch (fundamental frequency) using the YIN algorithm

    :param y: Mono audio array
    :param sr: Sampling rate
    :param fmin: Minimum frequency in Hz, 27.5 Hz (A0) Piano's lowest note is a reasonable choice
    :param fmax: Maximum frequency in Hz, 4186 Hz (C8) Piano's highest note is a reasonable choice
    :param win_length: optional, Length of window used for difference function calculation (default=frame_length // 2)
    :param frame_length: Length of the analysis frame in samples
    :param hop_length: Number of audio samples between adjacent YIN predictions
    :param trough_threshold: Absolute threshold for peak estimation (trough detection)
    :param center: If True, the signal y is padded so that frame D[:, t] is centered at y[t * hop_length]
    :param pad_mode : Padding mode to use if centering (default="constant")

    :return: Array of fundamental frequencies in Hz
    """
    if win_length is None:
        win_length = frame_length // 2

    if hop_length is None:
        hop_length = frame_length // 4

    if center:
        padding = [(0, 0)] * y.ndim
        padding[-1] = (frame_length // 2, frame_length // 2)
        y = np.pad(y, padding, mode=pad_mode)

    y_frames = frame(y, frame_length=frame_length, hop_length=hop_length)

    min_period = int(np.floor(sr / fmax))
    max_period = min(int(np.ceil(sr / fmin)), frame_length - win_length - 1)

    yin_frames = _cumulative_mean_normalized_difference(
        y_frames, frame_length, win_length, min_period, max_period
    )
    parabolic_shifts = _parabolic_interpolation(yin_frames)

    is_trough = localmin(yin_frames, axis=-2)
    is_trough[..., 0, :] = yin_frames[..., 0, :] < yin_frames[..., 1, :]

    is_threshold_trough = np.logical_and(is_trough, yin_frames < trough_threshold)

    target_shape = list(yin_frames.shape)
    target_shape[-2] = 1

    global_min = np.argmin(yin_frames, axis=-2)
    yin_period = np.argmax(is_threshold_trough, axis=-2)

    global_min = global_min.reshape(target_shape)
    yin_period = yin_period.reshape(target_shape)

    no_trough_below_threshold = np.all(~is_threshold_trough, axis=-2, keepdims=True)
    yin_period[no_trough_below_threshold] = global_min[no_trough_below_threshold]

    yin_period = (min_period + yin_period + np.take_along_axis(parabolic_shifts, yin_period, axis=-2))[..., 0, :]

    f0 = sr / yin_period
    return f0


def frame(x: np.ndarray, *, frame_length: int, hop_length: int, axis: int = -1) -> np.ndarray:
    """
    Slice input array into overlapping frames using strides.

    :param x: Input array to frame
    :param frame_length: Length of each frame
    :param hop_length: Number of samples to advance between frames
    :param axis: Optional, Axis along which to frame (default: -1)

    :return: Framed array with an additional dimension for frame_length
    """
    if x.shape[axis] < frame_length:
        raise ParameterError(f"Input is too short (n={x.shape[axis]}) for frame_length={frame_length}")

    if hop_length < 1:
        raise ParameterError(f"Invalid hop_length: {hop_length}")

    shape = list(x.shape)
    shape[axis] = 1 + (x.shape[axis] - frame_length) // hop_length
    shape.insert(axis + 1, frame_length)

    strides = list(x.strides)
    frame_stride = strides[axis]
    strides.insert(axis + 1, frame_stride)
    strides[axis] *= hop_length

    return as_strided(x, shape=tuple(shape), strides=tuple(strides))


def _cumulative_mean_normalized_difference(y_frames: np.ndarray, frame_length: int, win_length: int, min_period: int,
                                           max_period: int, ) -> np.ndarray:
    """
    Compute the cumulative mean normalized difference function for each frame

    This function is the core of the YIN algorithm for pitch detection

    :param y_frames: Framed audio signal
    :param frame_length: Length of each frame
    :param win_length: Window length used for difference calculation
    :param min_period: Minimum period (lag) to consider
    :param max_period: Maximum period (lag) to consider

    :return: Normalized difference function for each frame
    """

    # Auto-correlation using FFT
    a = np.fft.rfft(y_frames, frame_length, axis=-2)
    b = np.fft.rfft(y_frames[..., win_length:0:-1, :], frame_length, axis=-2)
    acf_frames = np.fft.irfft(a * b, frame_length, axis=-2)[..., win_length:, :]
    acf_frames[np.abs(acf_frames) < 1e-6] = 0

    # Energy terms
    energy = np.cumsum(y_frames ** 2, axis=-2)
    energy = energy[..., win_length:, :] - energy[..., :-win_length, :]
    energy[np.abs(energy) < 1e-6] = 0

    # Difference function
    diff = energy[..., :1, :] + energy - 2 * acf_frames

    # Numerator
    yin_numerator = diff[..., min_period: max_period + 1, :]

    # tau_range to normalize cumulative mean
    tau_range = expand_to(np.arange(1, max_period + 1), ndim=diff.ndim, axes=-2)

    # cumulative mean from 1 to max_period inclusive
    cumulative_mean = np.cumsum(diff[..., 1: max_period + 1, :], axis=-2) / tau_range

    # Denominator — careful with indexing to match numerator shape
    yin_denominator = cumulative_mean[..., (min_period - 1): max_period, :]

    # Normalize
    yin_frames = yin_numerator / (yin_denominator + tiny(yin_denominator))

    return yin_frames


def _parabolic_interpolation(x: np.ndarray, axis: int = -2) -> np.ndarray:
    """
    Perform parabolic interpolation to refine the estimated lag positions

    :param x: Input array representing values to interpolate
    :param axis: Axis along which to interpolate (default: -2)

    :return: Array of interpolation shifts
    """

    xi = x.swapaxes(-1, axis)
    shifts = np.zeros_like(x)
    x0 = xi[..., :-2]
    x1 = xi[..., 1:-1]
    x2 = xi[..., 2:]

    denom = 2 * (2 * x1 - x0 - x2)
    numer = x0 - x2
    shift_vals = np.zeros_like(x1)
    valid = denom != 0
    shift_vals[valid] = numer[valid] / denom[valid]

    shiftsi = shifts.swapaxes(-1, axis)
    shiftsi[..., 1:-1] = shift_vals

    return shifts


def expand_to(x: Union[np.ndarray, Sequence], ndim: int, axes: Union[int, Sequence[int]]) -> np.ndarray:
    """
    Expand the dimensions of x by inserting singleton dimensions at specified axes

    :param x: Input array to expand
    :param ndim: Target number of dimensions
    :param axes: Axes at which to insert singleton dimensions

    :return: Expanded array
    """

    x = np.asarray(x)
    if isinstance(axes, int):
        axes = [axes]

    shape = list(x.shape)
    for ax in sorted(axes):
        shape.insert(ax % (ndim + len(axes)), 1)

    return x.reshape(shape)


def localmin(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Detect local minima along a given axis

    :param x: Input array
    :param axis: Axis along which to find local minima (default: -1)

    :return: Boolean array indicating local minima positions
    """
    left = np.roll(x, 1, axis=axis)
    right = np.roll(x, -1, axis=axis)
    return np.logical_and(x < left, x < right)


def tiny(x: np.ndarray) -> float:
    """
    Return the smallest positive usable number for the dtype of x
    :param x: Input array
    :return: Tiny positive number for the array's dtype
    """
    return np.finfo(x.dtype).tiny
