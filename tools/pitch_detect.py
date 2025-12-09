# coding:utf-8
"""
    :module: pitch_detect.py
    :description:
    :author: Michel 'Mitch' Pecqueur
    :date: 2024.06
"""

try:
    import crepe

    has_crepe = True
except Exception as e:
    has_crepe = False
    pass

try:
    import librosa

    has_librosa = True
except Exception as e:
    has_librosa = False
    pass

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import soxr
from scipy.ndimage import median_filter, uniform_filter1d, gaussian_filter1d
from scipy.signal import find_peaks, correlate

from common_audio_utils import rms
from utils import note_to_hz, hz_to_period, hz_to_note, note_to_name, name_to_note
from yin_pitch import yin
from common_math_utils import clamp


def pitch_detect(audio, sr, mode='yin', resample=None, note_range=(20, 109), st_ed=.25):
    """
    Detect pitch using the 'pYIN' algorithm, reasonably reliable but somewhat slow

    :param np.array audio:
    :param int sr: Input sampling rate
    :param str mode: Pitch detection algorithm
    'corr', 'yin', 'pyin' (requires librosa) or 'crepe' (requires tensorFlow)
    :param int or None resample: Sample rate used for pitch detection, down-sampling may speed up process
    :param list or tuple or None note_range: Piano range is (21,108) for example
    :param float or tuple or list or None st_ed: start and end of range for used audio as a list or tuple
    or a float percentage to read just a few periods

    :return: Frequency
    :rtype: float
    """
    if audio.ndim > 1:
        mono_audio = audio.mean(axis=-1)
    else:
        mono_audio = np.copy(audio)

    # Note range
    note_range = note_range or []
    frange = [note_to_hz(f) for f in note_range]
    kwargs = {}
    if len(frange) == 2:
        kwargs = dict(zip(['fmin', 'fmax'], frange))

    if resample is not None:
        mono_audio = soxr.resample(mono_audio, in_rate=sr, out_rate=resample)
        sr = resample

    # Use only a few periods
    if isinstance(st_ed, float):
        period = hz_to_period(frange[0], sr=sr)
        st = int(round(st_ed * len(mono_audio)))
        data = mono_audio[st:st + period * 8]
    # Use portion of audio
    elif type(st_ed) in (tuple, list):
        st, ed = np.round(len(mono_audio) * np.array(st_ed)).astype(np.int32)
        data = mono_audio[st:ed]
    # Use the whole audio
    else:
        data = mono_audio

    # Pitch detection
    if mode == 'yin':
        # 'yin' algorithm - no voicing or probabilities
        # if has_librosa:
        #     # Librosa implementation
        #     freqs = librosa.yin(data, sr=sr, **kwargs)
        # else:
        #     # Refactored from librosa.yin
        freqs = yin(data, sr=sr, **kwargs)
        freqs = median_filter(freqs, size=3)
        freqs = np.nanmedian(freqs)
        probs = np.array(1.0)
    elif mode == 'pyin' and has_librosa:
        # 'pyIn' algorithm
        freqs, flags, probs = librosa.pyin(data, sr=sr, **kwargs)
        # Exclude nan values from both frequency and probability arrays
        nans = np.isnan(freqs)
        freqs = freqs[~nans]
        probs = probs[~nans]
    elif mode == 'crepe' and has_crepe:
        # 'crepe' algorithm, uses tensorFlow and slower to initialize, might work better in some cases
        # Verbosity was set to 0 so script does not crash when executing with pythonw
        time, freqs, probs, activation = crepe.predict(data, sr, viterbi=False, model_capacity='full', verbose=0)
    else:
        freqs = pitch_autocorrelate(audio=mono_audio, sr=sr, pos=st_ed, size=8192)
        freqs = np.array(freqs)
        probs = np.array(1.0)

    # Frequency with the highest probability
    # mx_prob = np.argmax(probs)
    # freq = freqs[mx_prob]

    # Confidence weighted average of frequency
    freq = np.sum(freqs * probs) / np.sum(probs)

    return float(freq)


def pitch_autocorrelate(audio: np.ndarray, sr: int, resample: int | None = 22050, pos: float = .25, size: int = 4096,
                        graph: bool = False) -> float:
    """
    Simple pitch detection by auto-correlation
    Reasonably fast and reliable though it fails on bell-like sounds
    :param audio:
    :param sr:
    :param resample: LPF to remove un-wanted harmonics
    :param pos: Audio position to analyse
    :param size: Data size for correlation
    :param graph:
    :return: Note frequency
    """
    st = int(len(audio) * pos)

    segment = np.copy(audio[st:min(st + size, len(audio))])
    if audio.ndim > 1:
        segment = np.mean(segment, axis=-1)

    if resample and resample < sr:
        segment = soxr.resample(segment, sr, resample)
        segment = soxr.resample(segment, resample, sr)

    # LPF to remove un-wanted harmonics
    segment = uniform_filter1d(segment, 63, mode='nearest')

    # Auto-correlate signal
    corr = correlate(segment, segment, mode='same')[len(segment) // 2:]

    # Find the highest peak
    peaks = find_peaks(corr)[0]

    if len(peaks) < 1:
        return np.nan

    # Next highest peak
    amp_peaks = corr[peaks]
    per_idx = np.argmax(amp_peaks)
    per = peaks[per_idx]

    # Average periods to refine estimation
    fltr_peaks = [0, per]
    value = per
    while True:
        x = value + per
        idx = np.argmin(np.abs(peaks - x))
        value = peaks[idx]
        if value in fltr_peaks or abs(value - fltr_peaks[-1]) < per * .8:
            break
        fltr_peaks.append(value)

    avg_per = np.mean(np.diff(fltr_peaks))
    freq = sr / avg_per

    if graph:
        pitch = hz_to_note(freq)
        note = int(round(pitch))
        pitch_fraction = (pitch - note) * 100

        notename, octave = note_to_name(int(round(pitch)))
        title_str = f'Period: {per} samples, {freq:.3f} Hz, Note: {note} ({notename}{octave}) {pitch_fraction:.3f}'
        print(title_str)
        plt.plot(corr, label='Auto-Correlation')
        plt.vlines(fltr_peaks, ymin=min(corr), ymax=max(corr), label='Periods', colors='red')
        plt.title(title_str)

        plt.legend()
        plt.show()

    return freq


def fine_tune(audio: np.ndarray, sr: int, pos: float = .25,
              os: int = 16, note: int = 60, graph: bool = True) -> tuple[float, float]:
    """
    Detect pitch fraction starting from a given note by auto-correlation using mean square error
    Modified version with gaussian pre-filtering and segment oversampling

    :param audio: Input audio
    :param sr: Sampling rate
    :param note: Root note
    :param period_factor: Period multiplier, improves accuracy
    :param pos: Sample time
    :param size: Window size in samples
    :param os: oversampling factor, improves tuning accuracy
    :param graph: Graph the result of the auto-correlation process

    :return: Pitch fraction in semitone cents, confidence
    """
    if audio.ndim > 1:
        mono_audio = audio.mean(axis=-1)
    else:
        mono_audio = np.copy(audio)

    osr = sr * os
    os_len = len(audio) * os

    p = [hz_to_period(note_to_hz(note + o), sr=osr) for o in [-.5, 0, .5]]
    period = p[1]
    a, b = p[2] - p[1], p[0] - p[1]  # Search range from note periods

    mx_per = os_len / period

    # Array must contain at least 2 periods
    if mx_per < 2:
        return 0, -1  # fail

    if mx_per > 4:
        n_per = 4
    else:
        n_per = 2

    h = period * n_per // 2  # half window

    # Constrain position sampling considering sound length and window size
    min_t = h
    max_t = os_len - period - h
    t = clamp(int(os_len * pos), min_t, max_t)

    seg_st = (t - h) // os
    seg_ed = (t + period + h * 2) // os
    segment = mono_audio[seg_st:seg_ed + 1]
    segment = gaussian_filter1d(segment, 3, mode='nearest')
    os_seg = soxr.resample(segment, sr, osr)

    ref = os_seg[:h * 2]

    min_mse, value = -1, 0
    for i in range(a, b + 1):
        p = period + i
        win = os_seg[p: p + h * 2]
        mse = rms(ref - win)
        if mse < min_mse or min_mse == -1:
            min_mse = mse
            value = i

    f0 = osr / (period + value)
    detected_note = hz_to_note(f0)
    pf = (detected_note - note) * 100

    epsilon = 1e-6
    confidence = 1.0 / (min_mse + epsilon)

    max_confidence, pitch_fraction = -1, pf

    n, o = note_to_name(note)

    if graph:
        # Graph
        plt.figure()
        plt.title(f'Fine Tuning: {f0} Hz, {n}{o} {round(pitch_fraction, 3)}')
        p = period + value
        win = os_seg[p: p + h * 2]
        plt.plot(ref, label='Reference')
        value_sign = ('', '+')[value > 0]
        plt.plot(win, label=f'Result: {value_sign}{value / os} ({period} {value_sign}{value} samples x{os})')
        plt.plot(np.square(ref - win), label=f'MSE, Confidence {confidence}')

        plt.xlabel('y')
        plt.ylabel('y')
        plt.legend()
        plt.show()

    return pitch_fraction, confidence


def fine_tune_a(audio: np.ndarray, sr: int, pos: float = .25,
                os: int = 16, note: int = 60, graph: bool = True) -> tuple[float, float]:
    """
    Detect pitch fraction starting from a given note by auto-correlation using mean square error
    Naive full length oversampling, might be heavier

    :param audio: Input audio
    :param sr: Sampling rate
    :param note: Root note
    :param period_factor: Period multiplier, improves accuracy
    :param pos: Sample time
    :param size: Window size in samples
    :param os: oversampling factor, improves tuning accuracy
    :param graph: Graph the result of the auto-correlation process

    :return: Pitch fraction in semitone cents, confidence
    """
    if audio.ndim > 1:
        mono_audio = audio.mean(axis=-1)
    else:
        mono_audio = np.copy(audio)

    osr = sr * os
    os_len = len(audio) * os

    p = [hz_to_period(note_to_hz(note + o), sr=osr) for o in [-.5, 0, .5]]
    period = p[1]
    a, b = p[2] - p[1], p[0] - p[1]  # Search range from note periods

    mx_per = os_len / period

    # Array must contain at least 2 periods
    if mx_per < 2:
        return 0, -1  # fail

    if mx_per > 4:
        n_per = 4
    else:
        n_per = 2

    h = period * n_per // 2  # half window

    # Constrain position sampling considering sound length and window size
    min_t = h
    max_t = os_len - period - h
    t = clamp(int(os_len * pos), min_t, max_t)

    os_audio = soxr.resample(mono_audio, sr, osr)
    ref = os_audio[t - h:t + h]

    min_mse, value = -1, 0
    for i in range(a, b + 1):
        p = t + period + i
        win = os_audio[p - h: p + h]
        mse = rms(ref - win)
        if mse < min_mse or min_mse == -1:
            min_mse = mse
            value = i

    f0 = osr / (period + value)
    detected_note = hz_to_note(f0)
    pf = (detected_note - note) * 100

    epsilon = 1e-6
    confidence = 1.0 / (min_mse + epsilon)

    max_confidence, pitch_fraction = -1, pf

    n, o = note_to_name(note)

    if graph:
        # Graph
        plt.figure()
        plt.title(f'Fine Tuning: {f0} Hz, {n}{o} {round(pitch_fraction, 3)}')
        p = t + period + value
        win = os_audio[p - h: p + h]
        plt.plot(ref, label='Reference')
        value_sign = ('', '+')[value > 0]
        plt.plot(win, label=f'Result: {value_sign}{value / os} ({period} {value_sign}{value} samples x{os})')
        plt.plot(np.square(ref - win), label=f'MSE, Confidence {confidence}')

        plt.xlabel('y')
        plt.ylabel('y')
        plt.legend()
        plt.show()

    return pitch_fraction, confidence


def fine_tune_b(audio, sr, pos=.5, note=60, size=8192, os=16, graph=True):
    """
    Fine-tuning using numpy/scipy correlate

    :param pos: Analysis position
    :param size: Analysis size
    :param os: Oversampling factor to increase accuracy
    :param note: Root note to finetune from
    :param graph:

    :return: finetuning in semitone cents
    """
    notename, octave = note_to_name(note)

    st = int(len(audio) * pos)

    segment = np.copy(audio[st:min(st + size, len(audio))])
    if audio.ndim > 1:
        segment = np.mean(segment, axis=-1)

    # Oversampling
    osr = sr * os

    segment = soxr.resample(segment, sr, osr)

    corr = correlate(segment, segment, mode='same')[len(segment) // 2:]
    corr = gaussian_filter1d(corr, 3, mode='nearest')

    freqs = [note_to_hz(note + o) for o in [.5, -.5]]
    pr = [hz_to_period(f, osr) for f in freqs]

    z = corr[pr[0]:pr[1]]
    idx = np.argmax(z)
    p = pr[0] + idx
    f = osr / p
    finetuning = (hz_to_note(f) - note) * 100

    if graph:
        plt.plot(z, label='Auto-Correlation')
        plt.vlines([0, len(z)], min(z), max(z), colors='blue', label=f'Range ({freqs[1]:.02f} - {freqs[0]:.02f} Hz)')
        plt.vlines(idx, min(z), max(z), colors='red', label=f'Result {f:.03f} Hz, {finetuning:.2f} cents')
        plt.title(f'Fine Tuning from {notename}{octave} ({note})')
        plt.legend()
        plt.show()

    return finetuning


def fine_tune_lr(audio, sr, note):
    """
    Fine-tuning using librosa, a bit hit or miss
    :param audio:
    :param sr:
    :param note:
    :return:
    """
    if audio.ndim > 1:
        mono_audio = audio.mean(axis=1)
    else:
        mono_audio = audio
    result = librosa.estimate_tuning(y=mono_audio, sr=sr, resolution=1e-3, fmin=note_to_hz(note - 1),
                                     fmax=note_to_hz(note + 1), n_fft=16384)

    if np.isnan(result) or result is None:
        return None

    return result * 100


def pitch_test(input_file, mode='yin', pos=.25):
    """
    Test function
    :param str input_file:
    :param str mode:
    :param float pos:
    :return:
    """
    y, sr = sf.read(input_file)
    freq = pitch_detect(audio=y, sr=sr, mode=mode, resample=16000, st_ed=pos)
    pitch = hz_to_note(freq)
    note = int(round(pitch))
    note_name, octave = note_to_name(note)
    pitch_fraction = int(round((note - pitch) * 100))
    print(f'{freq} Hz, {note} {note_name}{octave} {pitch_fraction}')

# fp = r"D:\AUDIO\doodles\PS1_core\sources\tekken\T3_menu_valid_G3.flac"
# fp = r"D:\Instruments\Piano_M1\Samples\PianoM1_C7.flac"
# fp = r"D:\Instruments\Organ_M1\Samples\OrganM1_C2.flac"
# y, sr = sf.read(str(fp))
# f = pitch_detect(y, sr, mode='corr')
# n = hz_to_note(f)
# n, o = note_to_name(int(round(n)))
# print(f, f'{n}{o}')

# pf = fine_tune(y, sr, note=name_to_note('g3'), graph=True, os=16, pos=.5)
# print(pf)

# print(round(note_to_hz(name_to_note('a0')), 1), round(note_to_hz(name_to_note('c8')), 1))
