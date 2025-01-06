# coding:utf-8
"""
    :module: __init__.py
    :description: Re-synthesize audio from FFT analysis
    :author: Michel 'Mitch' Pecqueur
    :date: 2024.05
"""

import numpy as np
import soundfile as sf
from scipy.interpolate import interp1d
from scipy.signal import resample
from scipy.signal.windows import tukey

from common_audio_utils import rms, peak, db_to_lin
from common_math_utils import lerp


def fft_resynth(input_file=None, input_data=None, sr=44100, target_sr=None,
                start=0, fft_size=2048, freqs=None,
                make_stereo=False, seed=0, width=.25,
                atonal_mix=1, duration=1, normalize=None,
                output_file=None):
    """
    Generate seamlessly looping audio from a short sample using FFT re-synthesis

    :param str or None input_file:
    :param np.array input_data:
    :param int sr: Sampling Rate when using array as input

    :param int or None target_sr: Resample to desired input before processing

    :param int start: Start of input audio used for FFT
    :param int fft_size: Length of input audio used for FFT

    :param list(float) float or None freqs: Frequencies used to determine harmonic series and generate "tonal" content
    These frequencies will keep their phase intact, other frequencies wil get random phases

    :param bool make_stereo: mono to stereo
    :param int seed: Seed for random phase (noise)
    Strong pseudo-stereo effect as left and right channel phases are uncorrelated except for "tonal" content
    :param float width: Stereo width

    :param float atonal_mix: "atonal" (random phases) content amplitude

    :param float duration: Desired output length (in s)

    :param float normalize: Normalize result to given dB

    :param str or None output_file:

    :return: Generated audio
    :rtype: np.array
    """
    if input_file:
        input_data, sr = sf.read(input_file)

    if target_sr is not None and target_sr != sr:
        target_len = np.round(len(input_data) * target_sr / sr).astype(np.int32)
        input_data = resample(input_data, target_len, domain='time')
        sr = target_sr

    chn_random = False
    if input_data.ndim < 2 and make_stereo:
        input_data = np.column_stack((input_data, input_data))
    if make_stereo:
        chn_random = True
    nch = input_data.ndim

    input_data = input_data[start:min(start + fft_size, len(input_data)) + 1]

    fft_size = len(input_data)
    fft_duration = fft_size / sr
    result_size = int(sr * duration)

    fft_win = tukey(fft_size, .5)

    out_data = np.zeros(shape=(result_size, nch))
    for c in range(nch):
        if nch > 1:
            in_chn = input_data[:, c]
        else:
            in_chn = input_data

        in_chn *= fft_win
        fft_in = np.fft.fft(in_chn)

        # FFT from noise to generate random phases
        if chn_random:
            np.random.seed(seed + c)
        else:
            np.random.seed(seed)

        noise_pattern = np.random.uniform(-1, 1, result_size)
        fft_noise = np.fft.fft(noise_pattern)

        # Create an "empty" FFT
        fft_result = np.zeros(result_size, dtype=complex)

        # Interpolate FFT and use fft noise for phases
        src_idx = np.arange(len(fft_in))
        tgt_idx = np.linspace(0, len(fft_in) - 1, len(fft_result))
        magnitudes = interp1d(src_idx, np.abs(fft_in), kind='cubic')(tgt_idx)
        phases = np.angle(fft_noise)

        w = (1, atonal_mix)[bool(freqs)]
        fft_result = magnitudes * np.exp(1j * phases) * w

        # Replace phases in resulting bins with fft_input according to harmonic series
        if freqs:
            h_list = get_harmonic_series(freqs, max_freq=sr / 2, number=0)

            src_f_idx = (h_list * fft_duration).astype(np.int32)
            tgt_f_idx = (h_list * duration).astype(np.int32)

            m = magnitudes[tgt_f_idx]
            phi = np.angle(fft_in[src_f_idx])

            fft_result[tgt_f_idx] = m * np.exp(1j * phi)
            fft_result[-tgt_f_idx] = np.conj(fft_result[tgt_f_idx])

        out_chn = np.real(np.fft.ifft(fft_result))

        # Match original volume
        factor = peak(in_chn) / peak(out_chn)
        out_chn *= factor

        out_data[:, c] = out_chn

    if make_stereo:
        mono = np.mean(out_data, axis=-1)
        out_data = lerp(np.column_stack((mono, mono)), out_data, width)

    if out_data.ndim == 2 and nch == 1:
        out_data = out_data.reshape(-1)

    mx = peak(out_data)

    if normalize is not None:
        out_data /= mx / db_to_lin(normalize)
    elif mx >= 1:
        # Prevent clipping
        out_data /= mx / db_to_lin(-.5)

    if output_file and out_data is not None:
        sf.write(output_file, out_data, samplerate=sr)

    return out_data


def get_harmonic_series(freqs=440, max_freq=20000, number=0):
    """
    Get harmonic series of a given fundamental frequency

    :param list or float freqs: List of fundamental frequencies
    :param float max_freq: Return harmonics up to this frequency
    :param int number: Number of harmonics if value > 0

    :return: Array of frequencies
    :rtype: np.array
    """
    result = np.empty((0,))
    if type(freqs) in [int, float]:
        freqs = [freqs]
    for f in freqs:
        itr = (int(max_freq / f), number)[number > 0]
        result = np.append(result, (np.arange(itr) + 1) * f)
    return np.unique(result)
