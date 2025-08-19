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
from scipy.signal import fftconvolve

from split_audio import envelope_transform


def sampleset_offset(input_path: str | list, smp_fmt: tuple[str, ...] = ('wav', 'flac', 'aif'),
                     verbose: bool = False, **kwargs) -> float:
    """
    Estimate average sample offset for a given sample set

    :param input_path:
    :param smp_fmt:
    :param verbose:
    :param kwargs: Optional keyword arguments supported by sub-function estimate offset
    :return: Average offset
    """
    if verbose:
        print('Sample offset estimation:')

    if isinstance(input_path, str):
        input_path = [input_path]

    offsets = []
    for item in input_path:
        samples = []

        p = Path(item)
        if p.is_dir():
            for ext in smp_fmt:
                samples.extend(p.glob('*.' + ext))
        elif p.is_file():
            if p.suffix[1:] in smp_fmt:
                samples = [p]

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


def estimate_offset(audio: np.ndarray, sr: int, amp: float = .5, factor: float = 1.0,
                    rtyp: str | None = 'ms') -> float:
    """
    Estimate sample time offset from amplitude envelope

    :param audio: Input audio
    :param sr: Sampling rate
    :param amp: (0-1) factor applied to found attack end amplitude to use as starting point
    :param factor: (0-1) Result multiplier
    :param rtyp: Return type: 'ms' (milliseconds), 's' (seconds) or samples
    :return: Estimated offset
    """
    if audio.ndim > 1:
        mono_audio = audio.mean(axis=1)
    else:
        mono_audio = audio

    # Calculate envelope from audio
    envelope = envelope_transform(mono_audio, w=1024, mode='max', interp='linear')

    # "Blur" envelope to smooth noise
    k_size = sr // 16
    kernel = np.ones(k_size)
    envelope = fftconvolve(envelope, kernel, mode='same') / k_size

    # Find where the attack ends
    attack_end = np.argwhere(np.diff(envelope) <= 0).reshape(-1)

    # Find where the audio reaches a given factor of the attack end volume
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
            sampleset_offset(input_path=files, smp_fmt=('wav', 'flac', 'aif'),
                             verbose=True, amp=.5, factor=1, rtyp='ms')
        except Exception as e:
            traceback.print_exc()
            print(f'Encountered an error - {e}')
    else:
        print('Nothing supplied')
    input('Press ENTER to continue...')


if __name__ == "__main__":
    main()
