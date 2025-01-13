# coding:utf-8
"""
    :module: estimate_offset.py
    :description: Estimate sample offset to set midi delay for sampled instrument

    :author: Michel 'Mitch' Pecqueur
    :date: 2025.01
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf

from split_audio import envelope_transform


def sampleset_offset(input_path='', smp_fmt=('wav', 'flac', 'aif'), verbose=True, **kwargs):
    """
    Estimate average sample offset for a given sample set

    :param str input_path:
    :param tuple or list smp_fmt:
    :param bool verbose:
    :param dict kwargs:

    :return:
    :rtype: int or float
    """
    samples = []

    p = Path(input_path)
    if p.is_dir():
        samples = p.glob('*.flac')
    elif p.is_file():
        if p.suffix[1:] in smp_fmt:
            samples = [p]

    offsets = []
    for smp in samples:
        audio, sr = sf.read(smp)
        o = estimate_offset(audio, sr, **kwargs)
        if verbose:
            print(f'{smp.stem} : {o:.3f}')
        offsets.append(o)

    if offsets:
        mean = np.mean(offsets)
        if verbose:
            print(f'Mean : {mean:.3f}')
        return mean


def estimate_offset(audio, sr, amp=.5, factor=1.0, rtyp='ms'):
    """
    Estimate sample time offset from amplitude envelope

    :param np.ndarray audio: Input audio
    :param int sr: Sampling rate
    :param float amp: (0-1) factor applied to found attack end amplitude to use as starting point
    :param float factor: (0-1) Result multiplier
    :param str rtyp: Return type: 'ms' (milliseconds), 's' (seconds) or samples
    :return: Estimated offset
    :rtype: int or float
    """
    if audio.ndim > 1:
        mono_audio = audio.mean(axis=1)
    else:
        mono_audio = audio

    # Calculate envelope from audio
    envelope = envelope_transform(mono_audio, w=1024, mode='max', interp='linear')

    # "Blur" envelope to smooth potential noise
    k_size = sr // 16
    kernel = np.ones(k_size)
    envelope = np.convolve(envelope, kernel, mode='same') / k_size

    # Find where the attack ends
    attack_end = np.argwhere(np.diff(envelope) <= 0).reshape(-1)

    # Find where the audio reach the
    start_amp = envelope[attack_end[0]] * amp
    offset = np.argwhere(envelope >= start_amp).reshape(-1)[0] * factor

    match rtyp:
        case 'ms':
            return offset / sr * 1000
        case 's':
            return offset / sr
        case _:
            return int(round(offset))


def main():
    if len(sys.argv) > 1:
        files = sys.argv[1:]
        try:
            for f in files:
                sampleset_offset(input_path=f, smp_fmt=('wav', 'flac', 'aif'),
                                 verbose=True, amp=.5, factor=1, rtyp='ms')
        except Exception as e:
            traceback.print_exc()
            print(f'Encountered an error - {e}')
    else:
        print('Nothing supplied')
    input('Press ENTER to continue...')


if __name__ == "__main__":
    main()
