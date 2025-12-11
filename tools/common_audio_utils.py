# coding:utf-8
"""
    :module: common_audio_utils.py
    :description:
    :author: Michel 'Mitch' Pecqueur
    :date: 2024.08
"""
import numpy as np
from scipy import signal
from scipy.signal.windows import tukey

from common_math_utils import lerp
from tools.common_math_utils import q_exp, q_log
from utils import hz_to_period


# Volume

def peak(x):
    return np.max(np.abs(x))


def rms(x):
    return np.sqrt(np.mean(x ** 2))


def avg(x):
    return np.mean(np.abs(x))


def normalize(audio: np.array, db: float = -1, prevent_clipping=False):
    """
    Normalize an audio array to given peak volume
    :param audio:
    :param prevent_clipping: Normalize only if amplitude > 1
    :param db:
    :return:
    """
    result = np.copy(audio)
    mx = peak(audio)
    if not prevent_clipping or mx >= 1:
        result /= mx / db_to_lin(db)
    return result


# dB conversion

def db_to_lin(db):
    return np.power(10, db / 20)


def lin_to_db(lin):
    return 20 * np.log10(lin)


# Fade in/out functions

def apply_fade(data: np.ndarray, fade_in: tuple[int, int, str] | None = (0, 100, 'log'),
               fade_out: tuple[int, int, str] | None = (500, 32000, 'exp')) -> np.ndarray:
    """
    Apply fade in/out to audio
    :param data: Input audio
    :param fade_in: Start, Duration, Curve
    :param fade_out: Start, Duration, Curve
    :return: Processed audio
    """
    nch = data.ndim
    length = len(data)

    # Fade in
    if fade_in:
        fi = np.append(np.zeros(fade_in[0]), np.linspace(0, 1, fade_in[1]))
        pad = length - fi.size
        if pad > 0:
            fi = np.append(fi, np.ones(pad))
        else:
            fi = fi[:length]
        if fade_in[2] == 'exp':
            fi = q_exp(fi)
        if fade_in[2] == 'log':
            fi = q_log(fi)
    else:
        fi = 1

    # Fade out
    if fade_out:
        fo = np.append(np.ones(fade_out[0]), np.linspace(1, 0, fade_out[1]))
        pad = length - fo.size
        if pad > 0:
            fo = np.append(fo, np.zeros(pad))
        else:
            fo = fo[:length]
        if fade_out[2] == 'exp':
            fo = q_exp(fo)
        if fade_out[2] == 'log':
            fo = q_log(fo)
    else:
        fo = 1

    fade = fi * fo

    if nch > 1:
        fade = np.tile(fade, (nch, 1)).T

    data *= fade

    return data


# Stereo manipulation

def st_to_ms(audio):
    (l_chn, r_chn) = audio.T
    mid = (l_chn + r_chn) / 2
    side = (l_chn - r_chn) / 2
    return np.column_stack((mid, side))


def ms_to_st(audio):
    (mid, side) = audio.T
    l_chn = mid + side
    r_chn = mid - side
    return np.column_stack((l_chn, r_chn))


def get_silence_threshold(bit_depth, as_db=True):
    mx = 2 ** (bit_depth - 1) - 1
    result = 1 / mx
    if as_db:
        return lin_to_db(result)
    return result


def balance_lr(audio: np.ndarray) -> np.ndarray:
    """
    Balance L/R channels volume
    :param audio:
    :return:
    """
    # Use rms level from left and right channels as barycenter weights
    result = np.copy(audio)
    (l_chn, r_chn) = result.T
    wt = np.array([rms(l_chn), rms(r_chn)])
    wt /= sum(wt) / 2  # Normalize weights
    result /= wt
    return result


def align_phase_lr(audio: np.ndarray, mode: str | None = 'min') -> np.ndarray:
    """
    Align phase between L/R channels so the audio feels centered
    :param audio:
    :param mode: 'min' minimum delay or best correlation
    :return:
    """
    result = np.copy(audio)
    (l_chn, r_chn) = result.T

    corr = signal.correlate(r_chn, l_chn, mode='same')

    # Evaluate delay in both direction and use result according to mode
    center = len(corr) // 2
    pos_delay = np.argmax(corr[center:])
    neg_delay = np.argmax(corr[:center][::-1])

    match mode:
        case 'min':
            delay = (-neg_delay, pos_delay)[bool(neg_delay > pos_delay)]
        case _:
            delay = (-neg_delay, pos_delay)[bool(corr[pos_delay] > corr[neg_delay])]

    if not delay:
        return result

    print(f'Delay: {delay} samples')
    r_chn = shift(r_chn, -delay, r_chn[-1])
    r_chn = apply_fade(r_chn, fade_in=(0, 8, 'log'), fade_out=(len(r_chn) - abs(delay), abs(delay), 'log'))

    result = np.column_stack((l_chn, r_chn))

    return result


def shift(arr: np.ndarray, value: int = 0, fill_value: float = 0) -> np.ndarray:
    """
    Shift an array by a given value
    :param arr:
    :param value:
    :param fill_value: Value used for padding
    :return:
    """
    result = np.empty_like(arr)
    if value > 0:
        result[:value] = fill_value
        result[value:] = arr[:-value]
    elif value < 0:
        result[value:] = fill_value
        result[:value] = arr[-value:]
    else:
        result[:] = arr
    return result


# Convolution

def convolve(audio, ir, mx_len=False, wet=1.0, comp_vol=True):
    """
    Convolve input audio with an impulse response
    :param np.ndarray audio: Input audio
    :param np.ndarray ir: Impulse response
    :param bool mx_len: Extend length so convolved tail is not cut
    :param float wet: Blend between processed and original audio
    :param bool comp_vol: Compensate rms volume so it does not change after convolution
    :return: Processed audio
    :rtype: np.ndarray
    """
    au_nch, ir_nch = audio.ndim, ir.ndim
    au_l, ir_l = len(audio), len(ir)

    # Match number of channels between IR and audio
    if ir_nch > au_nch:
        audio = np.tile(audio[:, np.newaxis], (1, ir_nch))
    elif ir_nch < au_nch:
        ir = np.tile(ir[:, np.newaxis], (1, au_nch))

    # Adjust length so both audio and IR match
    length = (au_l, au_l + ir_l)[mx_len]
    audio = np.pad(audio, pad_width=((0, length - au_l), (0, 0)), mode='constant', constant_values=0)
    # ir = np.pad(ir, pad_width=((0, length - ir_l), (0, 0)), mode='constant', constant_values=0)

    result = None
    for c in range(ir_nch):
        if ir_nch > 1:
            ir_chn = ir[:, c]
            au_chn = audio[:, c]
        else:
            ir_chn = ir
            au_chn = audio

        conv = signal.fftconvolve(au_chn, ir_chn)[:length]
        if result is None:
            result = conv
        else:
            result = np.column_stack((result, conv))

    result = lerp(audio, result, wet)

    # Compensate volume
    if comp_vol:
        result *= (rms(audio) / rms(result))

    return result


def deconvolve(audio, reference, lambd=1e-3, mode='minmax_sum'):
    """
    Deconvolve audio from a reference sound (typically a sweep) to an impulse response

    :param np.ndarray audio: Convolved audio
    :param np.ndarray reference: Reference audio
    :param float lambd: Peak signal-to-noise ratio
    :param str mode: Match length mode between audio and ref, "min", "max" or 'minmax_sum' (to alleviate wrap-around)

    :return: Resulting IR
    :rtype: np.ndarray
    """
    au_nch, ref_nch = audio.ndim, reference.ndim

    result = None
    for i in range(au_nch):
        if au_nch > 1:
            conv_data = audio[:, i]
        else:
            conv_data = audio

        if ref_nch > 1:
            ref_data = reference[:, i]
        else:
            ref_data = reference

        # Match convolved and reference audio length
        r_l, c_l = len(ref_data), len(conv_data)
        mn_l, mx_l = min(r_l, c_l), max(r_l, c_l)
        if mode == 'min':
            length = mn_l
            kernel = ref_data[:length]
            conv_data = conv_data[:length]
        else:
            length = mx_l
            pad_length = (length, length + mn_l)[mode == 'minmax_sum']
            conv_data = np.pad(conv_data, (0, max(0, pad_length - c_l)), mode='constant', constant_values=0)
            kernel = np.pad(ref_data, (0, max(0, pad_length - r_l)), mode='constant', constant_values=0)

        # Wiener Deconvolution
        # Taken fom "Example of Wiener deconvolution in Python"
        # Written 2015 by Dan Stowell. Public domain.
        fft_k = signal.fft(kernel)
        deconv = np.real(signal.ifft(signal.fft(conv_data) * np.conj(fft_k) / (fft_k * np.conj(fft_k) + lambd ** 2)))[
                 :length]

        if result is None:
            result = deconv
        else:
            result = np.column_stack((result, deconv))

    return result


def generate_sweep(duration=4, sr=48000, db=-6, start_freq=20, window=True, os=1):
    """
    Generate logarithmic sweep tone
    :param float duration: in seconds
    :param int sr: Sample Rate
    :param float db: Volume
    :param float start_freq: in Hz
    :param bool window: Apply window
    :param int os: Oversampling factor
    :return: Generated audio
    :rtype: np.ndarray
    """
    length = int(duration * sr)
    end_freq = sr / 2

    pad = np.array([0])
    if window:
        pad = hz_to_period(start_freq, sr) * np.array([1, 1], dtype=np.int16)

    freq = np.logspace(np.log10(start_freq), np.log10(end_freq), (length - sum(pad)) * os, endpoint=True)

    freq[[0, -1]] = [start_freq, end_freq]
    freq = np.pad(freq, pad_width=pad * os, mode='edge')

    phase = np.cumsum(2 * np.pi * freq / (sr * os))
    phase -= phase[0]  # Start from 0
    sweep = np.sin(phase) * db_to_lin(db)

    if window:
        a = (length * os / hz_to_period(start_freq, sr * os)) * 2
        w = tukey(length * os, a)
        # w = sweep_window(length * os, int(hz_to_period(start_freq, sr * os)))
        sweep *= w

    if os > 1:
        sweep = signal.decimate(sweep, os, zero_phase=True)

    return sweep


def compensate_ir(audio, mode='rms', sr=48000):
    """
    Compensate impulse response volume so convolved audio keeps approximately the same gain as original
    :param np.ndarray audio: Input impulse response
    :param str mode: Normalization mode, 'peak' or 'rms'
    :param int sr: Sampling rate
    :return: processed IR
    :rtype: np.ndarray
    """
    nch = audio.ndim
    length = len(audio)

    vol_func = {'peak': peak, 'rms': rms}[mode]

    test_tone = generate_sweep(length / sr, sr, db=-6, start_freq=20, window=True, os=1)

    values = []
    for c in range(nch):
        if nch > 1:
            data = audio[:, c]
        else:
            data = audio
        orig_vol = vol_func(test_tone)
        conv = signal.fftconvolve(test_tone, data, mode='full')[:length]
        conv_vol = vol_func(conv)
        factor = orig_vol / conv_vol
        values.append(factor)
    gain = np.mean(values)

    return audio * gain


# Other functions

def pad_audio(audio: np.ndarray, before: int = 0, after: int = 0, mode: str = 'constant') -> np.ndarray:
    """
    Simplified multichannel audio padding
    :param  audio:
    :param before:
    :param after:
    :param mode:
    :return:
    """
    if audio.ndim > 1:
        nch = audio.shape[1]
    else:
        nch = 1

    result = np.copy(audio)
    result = result.reshape(len(result), nch)

    if mode == 'constant':
        result = np.pad(result, pad_width=((before, after), (0, 0)), constant_values=0, mode='constant')
    else:
        result = np.pad(result, pad_width=((before, after), (0, 0)), mode=mode)

    if nch == 1:
        result = result.reshape(-1)

    return result


def pan_audio(audio, value=0):
    """
    Apply stereo panning to an audio array
    :param np.ndarray audio:
    :param float value: value between -1 and 1
    :return:
    :rtype: np.ndarray
    """
    nch = audio.ndim
    if nch > 1:
        st_audio = audio
        mono_audio = np.mean(audio, axis=1)
        result = lerp(st_audio, np.column_stack([mono_audio, mono_audio]), abs(value))
    else:
        result = np.column_stack([audio, audio])

    x = (value + 1) / 2
    result = result * np.array([1 - x, x])

    return result


def balance_audio(audio, value=0):
    """
    Apply stereo balance to an audio array
    :param np.ndarray audio:
    :param float value: value between -1 and 1
    :return:
    :rtype: np.ndarray
    """
    if audio.ndim > 1:
        st_audio = audio
    else:
        st_audio = np.column_stack([audio, audio])

    x = (value + 1) / 2
    result = st_audio * np.array([1 - x, x])

    return result


def pitch_audio(audio, pitch=0):
    """
    Basic pitch shifter for preview purpose
    :param np.ndarray audio:
    :param float pitch: Pitch shifting in semitones
    :return:
    :rtype: np.ndarray
    """
    nch = audio.ndim

    x = np.linspace(0, 1, num=len(audio), endpoint=True)
    new_length = len(audio) * pow(2, -pitch / 12)
    new_length = int(round(new_length))
    xi = np.linspace(0, 1, num=new_length, endpoint=True)

    result = np.zeros(new_length)
    result = np.column_stack([result] * nch)

    for c in range(nch):
        if nch > 1:
            chn = audio[:, c]
        else:
            chn = audio
        chn_result = np.interp(xi, x, chn)
        if nch > 1:
            result[:, c] = chn_result
        else:
            result = chn_result

    return result
