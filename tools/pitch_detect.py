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
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.ndimage import median_filter

from utils import note_to_hz, hz_to_period, hz_to_note, note_to_name
from yin_pitch import yin


def pitch_detect(audio, sr, mode='pyin', resample=None, note_range=(20, 109), st_ed=.25):
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
        freqs = pitch_autocorrelate(audio=mono_audio, sr=sr, pos=st_ed, length=8192)
        freqs = np.array(freqs)
        probs = np.array(1.0)

    # Frequency with the highest probability
    # mx_prob = np.argmax(probs)
    # freq = freqs[mx_prob]

    # Confidence weighted average of frequency
    freq = np.sum(freqs * probs) / np.sum(probs)

    return float(freq)


def pitch_autocorrelate(audio, sr, pos=.25, length=8192, graph=False):
    """
    Simple pitch detection by auto-correlation
    Reasonably fast and reliable though it fails on bell-like sounds
    :param np.array audio:
    :param int sr:
    :param float pos: Audio position to analyse
    :param int length: Data size for correlation
    :param bool graph:
    :return:
    :float:
    """
    p = int(len(audio) * pos)

    segment = np.copy(audio[p:p + length])
    if audio.ndim > 1:
        segment = np.mean(segment, axis=-1)

    # LPF to remove un-wanted harmonics
    kl = 63
    k = np.ones(kl) / kl
    segment = np.convolve(segment, k)

    # Auto-correlate signal
    corr = np.correlate(segment, segment, mode='same')[len(segment) // 2:]

    # Find the highest peak
    peaks = find_peaks(corr)[0]

    if len(peaks) < 1:
        return np.nan

    amp_peaks = corr[peaks]

    # Next highest peak
    per_idx = np.argmax(amp_peaks)
    per = peaks[per_idx]

    # Average periods to refine estimation
    fltr_peaks = [0, per]
    value = fltr_peaks[-1]
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


def fine_tune(audio, sr, note, period_factor=3, t=8000, d=50, os=16, graph=True):
    """
    Detect pitch fraction starting from a given note by auto-correlation

    :param np.array audio: Input audio
    :param int sr: Sampling rate
    :param int note: Root note
    :param int period_factor: Period multiplier improves accuracy
    :param t: Sample time
    :param float d: Window size in ms
    :param int os: oversampling factor, improves tuning accuracy
    :param bool graph: Graph the result of the auto-correlation process

    :return: Pitch fraction in semitone cents, confidence
    :rtype: tuple
    """
    # Adjust window size if needed
    min_d = min(d, min(t - len(audio), len(audio) - t) / sr * 1000)

    if audio.ndim > 1:
        mono_audio = audio.mean(axis=1)
    else:
        mono_audio = audio

    max_confidence, pitch_fraction = -1, 0
    for factor in range(1, period_factor + 1):
        transpose = (1 - factor) * 12
        p = [hz_to_period(note_to_hz(note + transpose + o), sr=sr * os) for o in [-.51, 0, .51]]
        period = p[1]
        st, ed = p[1] - p[0], p[1] - p[2]

        ds = sr * os * min_d / 1000  # duration in samples
        nper = max(int(np.ceil(ds / period)), int(4 / period_factor))
        ws = period * nper

        shift = period
        audio_segment = mono_audio[t + st // os:t + (ws + shift + ed) // os + 1]

        x = np.linspace(0, 1, len(audio_segment))
        x_new = np.linspace(0, 1, len(audio_segment) * os)
        os_audio = interp1d(x, audio_segment, kind='cubic', axis=0)(x_new)

        ref_audio = os_audio[-st: -st + period * nper]

        min_mse, value = -1, 0
        for i in range(st, ed + 1):
            window = os_audio[-st + shift + i: -st + shift + ws + i]
            mse = np.mean(np.square(ref_audio - window))
            if mse < min_mse or min_mse == -1:
                min_mse = mse
                value = i

        # print(f'Period correction (sample): {value / os} ({value} x{os}) mse: {min_mse}')
        freq = (sr * os) / (period + value)
        detected_note = hz_to_note(freq) - transpose
        freq = note_to_hz(detected_note)
        p_f = (detected_note - note) * 100

        epsilon = 1e-6
        confidence = 1.0 / (min_mse + epsilon)

        # print(p_f, confidence)

        if confidence > max_confidence:
            max_confidence = confidence
            pitch_fraction = p_f

    n, o = note_to_name(note)
    # print(f'{freq} Hz, {n}{o} {pitch_fraction} semitone cents, confidence: {confidence}')

    if graph:
        # Graph
        plt.figure()
        plt.title(f'Fine Tuning: {freq} Hz, {n}{o} {round(pitch_fraction, 3)}')
        window = os_audio[-st + shift + value: -st + shift + ws + value]
        plt.plot(ref_audio, label='Reference')
        value_sign = ('', '+')[value > 0]
        plt.plot(window, label=f'Result: {value_sign}{value / os} ({period} {value_sign}{value} samples x{os})')
        plt.plot(np.square(ref_audio - window), label=f'MSE, Confidence {confidence}')

        plt.xlabel('y')
        plt.ylabel('y')
        plt.legend()
        plt.show()

    return pitch_fraction, max_confidence


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


def pitch_test(input_file, mode='pyin', pos=.25):
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
