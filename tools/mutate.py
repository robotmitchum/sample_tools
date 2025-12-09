# coding:utf-8
"""
    :module: mutate.py
    :description:
    :author: Michel 'Mitch' Pecqueur
    :date: 2025.04
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pyrubberband as pyrb
import scipy.ndimage as ndimage
import soundfile as sf
from scipy.interpolate import interp1d
from scipy.signal import ShortTimeFFT, find_peaks
from soxr import resample

from common_audio_utils import peak, rms, db_to_lin, pad_audio
from common_math_utils import string_to_seed
from sample_utils import Sample, append_metadata, set_md_tags
from split_audio import trim_audio
from tools.common_audio_utils import apply_fade
from file_utils import resolve_overwriting


def mutate(audio: np.ndarray = None, sr: int = None, input_file: Path | str | None = None,
           output_file: Path | str | None = None, os_factor: int = 1, target_sr: int or None = None,
           bit_depth: int | None = None, no_overwriting: bool = True,
           multi_chn: bool = True, trim_db: float = -60, min_duration: float = .1,
           stft_size: int = 8192, iterations: int = 3,
           end_mode: str = 'loop', loop: tuple[int, int] = None,
           count: int = 3, seed_mode: str = 'name', seed: int = 12345,
           params: tuple[tuple[float, float, float], tuple[float, float, float]] = ((1, 1, 0), (1, 1, 0)),
           random_params: tuple[tuple[float, float, float], tuple[float, float, float]] = ((.5, .5, 2), (.25, .5, .05)),
           match_cues: bool = True, interp='linear', fade_db: float = -40,
           progress_bar=None, worker=None, progress_callback=None, message_callback=None, range_callback=None
           ) -> list[Path] | list[np.ndarray] | None:
    """
    Generate randomized variants from input audio

    :param audio: Input Audio data
    :param sr: Sampling rate

    :param input_file: Overrides the following kwargs   audio, sr, loop

    :param multi_chn: Enable multichannel processing, mono if False
    :param min_duration: Minimum duration between transient cues in seconds

    :param output_file:

    :param os_factor: Oversampling factor
    :param target_sr: Output sampling rate
    :param bit_depth: Output bit-depth
    :param no_overwriting:

    :param count: Number of desired variants
    :param seed_mode:
    'name' use file name as seed (input_file has to be provided)
    'value' use supplied seed value
    :param seed: Random generator seed value

    :param trim_db: Trim threshold for input and output

    :param stft_size: Size of stft analysis for noise/tonal splitting
    :param iterations: Iterations for noise/tonal splitting

    :param end_mode: See split_noise_tonal function docstrings
    :param loop: Loop start, Loop end when end_mode is set to 'loop'

    :param params: Noise|Tonal parameters (amp, rate, pitch)
    :param random_params: Noise|Tonal randomization parameter (amp, rate, pitch)

    :param match_cues: Scale tonal cues so resulting tonal and noise splits have the same length
    :param interp: Interpolation of amplitude randomization 'linear', 'quadratic' or 'cubic'

    :param fade_db: Fade in/out volume threshold

    :return: List of file paths if output fil is provided otherwise list of audio arrays
    """

    # Progress bar init
    pb_val = None
    pb_steps = count

    if progress_bar is not None:
        # Get current bar progress
        pb_val = progress_bar.value()

    default_bd = 16
    if input_file is not None:
        audio, sr = sf.read(str(input_file))
        smp = Sample(str(input_file))
        if smp.loopStart is not None:
            loop = smp.loopStart, smp.loopEnd
        else:
            loop = None
        default_bd = smp.params.sampwidth * 8
        if seed_mode == 'name':
            p = Path(input_file)
            seed = string_to_seed(p.stem)
    else:
        smp = None

    bit_depth = bit_depth or default_bd

    if target_sr is None:
        target_sr = sr

    nch = audio.ndim
    data, _ = trim_audio(audio, trim_db)
    rms_in = rms(data)

    if nch > 1:
        mono_audio = np.mean(data, axis=-1)
    else:
        mono_audio = data

    if not multi_chn:
        data = mono_audio
        nch = 1

    # Transient cues
    cues = get_transient_cues(mono_audio, 128, 64, True, True)

    min_len = int(round(min_duration * sr))
    cues = decimate_cues(cues, min_len=min_len, keep_last=True)

    min_ncues = {'linear': 2, 'quadratic': 3, 'cubic': 4}[interp]
    if len(cues) < min_ncues:
        cues = interp1d(np.arange(len(cues)), cues, kind='linear')(np.linspace(0, len(cues) - 1, min_ncues)).astype(int)

    cues = cues * os_factor
    cues_diffs = np.diff(cues)

    # Split noise/tonal content
    split = split_noise_tonal(data, sr, stft_size=stft_size, iterations=iterations, end_mode=end_mode, loop=loop)

    # Oversampling
    if os_factor > 1:
        split = [resample(s, sr, sr * os_factor, quality='VHQ') for s in split]

    result = []
    pad_len = len(str(count))

    output_path = None

    last_cue = None
    for i in range(count):

        if worker is not None:
            if worker.is_stopped():
                return None

        variant_result = None
        variant_seed = seed + i
        np.random.seed(variant_seed)

        for j, (data, rand_prm, prm) in enumerate(zip(split, random_params, params)):
            amp_m, time_m, pitch_ofs = prm
            rand_amp, rand_time, rand_pitch = rand_prm

            # Time map
            rate = time_m * (np.random.uniform(-1, 1, len(cues_diffs)) * rand_time + 1)
            rl = np.round(cues_diffs / rate).astype(int)
            new_cues = np.sort(np.append(0, np.cumsum(rl)) + cues[0])

            if match_cues:
                if j == 0:
                    last_cue = new_cues[-1]
                else:
                    new_cues = np.round(new_cues * (last_cue / new_cues[-1])).astype(int)

            time_map = list(zip(cues, new_cues))

            # Pitch map
            pitch_values = np.random.uniform(-1, 1, len(cues)) * rand_pitch + pitch_ofs
            pitch_map = list(zip(cues, pitch_values))
            pitchmap_file = write_map_file(mapping=pitch_map)

            res = pyrb.timemap_stretch(data, sr * os_factor, time_map=time_map,
                                       rbargs={'--pitchmap': pitchmap_file.name, '--fine': ''})

            # Amplitude randomization
            amp = amp_m * (np.random.uniform(-1, 1, len(new_cues)) * rand_amp + 1)
            x_new = np.arange(0, new_cues[-1])
            amp = interp1d(new_cues, amp, kind=interp)(x_new)[:len(res)]
            if nch > 1:
                amp = np.tile(amp, (nch, 1)).T

            res *= amp

            if variant_result is None:
                variant_result = res
            else:
                pad = abs(len(res) - len(variant_result))
                if len(res) > len(variant_result):
                    variant_result = pad_audio(variant_result, before=0, after=pad, mode='constant')
                elif len(variant_result) > len(res):
                    res = pad_audio(res, before=0, after=pad, mode='constant')
                variant_result += res

            os.unlink(pitchmap_file.name)

        if os_factor > 1 or target_sr != sr:
            variant_result = resample(variant_result, sr * os_factor, target_sr, quality='VHQ')

        # Match original rms volume
        rms_res = rms(variant_result)
        nrm = rms_in / rms_res
        variant_result *= nrm

        # Prevent potential clipping
        mx = peak(variant_result)
        if mx >= 1:
            variant_result /= mx / db_to_lin(-.1)

        # Trim result
        variant_result, _ = trim_audio(variant_result, trim_db)

        # Fade in/out
        _, fade_cues = trim_audio(variant_result, fade_db)
        start_cue = max(fade_cues[0], 8)
        end_cue = min(fade_cues[-1], len(variant_result) - 8)
        variant_result = apply_fade(variant_result, fade_in=(0, start_cue, 'log'),
                                    fade_out=(end_cue, len(variant_result) - end_cue, 'log'))

        # Write file or return audio arrays
        if output_file is not None:
            # Directly write file if path is supplied
            p = Path(output_file)
            ext = p.suffix[1:]
            output_path = p.parent / f'{p.stem}{str(i + 1).zfill(pad_len)}{p.suffix}'
            print(f'{output_path.as_posix()} - Seed {variant_seed}')
            if not p.parent.exists():
                os.makedirs(p.parent, exist_ok=True)
            if ext == 'flac':
                bit_depth = min(bit_depth, 24)
            subtypes = {16: 'PCM_16', 24: 'PCM_24', 32: 'FLOAT'}
            subtype = subtypes[bit_depth]

            if no_overwriting and Path(output_path).resolve() == Path(input_file).resolve():
                resolve_overwriting(input_file, mode='dir', dir_name='backup_', test_run=False)

            # Write file
            sf_path = (output_path, f'{output_path}f')[ext == 'aif']
            cmp = ({}, {'compression_level': 1.0})[ext == 'flac']
            sf.write(str(sf_path), variant_result, target_sr, subtype, **cmp)
            if sf_path != output_path:
                os.rename(sf_path, output_path)

            if smp is not None:
                # Remove loop information as they can't simply be duplicated
                smp.loopStart, smp.LoopEnd = None, None
                # Append metadata
                if ext == 'wav':
                    append_metadata(str(output_path), note=smp.note, pitch_fraction=smp.pitchFraction,
                                    loop_start=smp.loopStart, loop_end=smp.loopEnd)
                # Add metadata as tags for flac
                elif ext == 'flac':
                    keys = ['note', 'pitchFraction', 'loopStart', 'loopEnd', 'cues']
                    values = [getattr(smp, value) for value in keys]
                    md = dict(zip(keys, values))
                    set_md_tags(str(output_path), md=md)

            # Append path to result
            result.append(output_path)
        else:
            # Append audio array to result
            result.append(variant_result)

        # Increment progress bar sub-task
        if progress_bar is not None and progress_callback is not None:
            if output_path is not None:
                message_callback.emit(f'{output_path.stem} %p%')
            pr = i + 1
            progress_callback.emit(pb_val + pr)

    # Recall progress bar initial state and increment parent task
    if progress_bar is not None and progress_callback is not None:
        progress_callback.emit(pb_val + pb_steps)

    return result


def get_transient_cues(audio: np.ndarray, size: int, hop: int,
                       start: int | bool | None = None, end: int | bool | None = None) -> np.ndarray:
    """
    Cue detection based on transients

    :param audio: Input MONO audio array
    :param size: Window size
    :param hop: Hop size
    :param start: Add given index as start cue, add 0 if True
    :param end: Add given index as end cue, add last audio sample if True
    :return: Array of cues
    """
    num_frames = 1 + (len(audio) - size) // hop

    frames = np.stack([audio[i * hop:i * hop + size] for i in range(num_frames)])
    energy = np.sum(frames ** 2, axis=1)
    mx = peak(energy)
    energy = ndimage.uniform_filter1d(energy, size=3, mode='constant', cval=0)
    energy /= mx
    cues, _ = find_peaks(energy, height=.01)
    cues *= hop

    if start is not None:
        st = (start, 0)[start is True]
        if st not in cues:
            cues = np.insert(cues, 0, st)
    if end is not None:
        ed = (end, len(audio))[end is True]
        if ed not in cues:
            cues = np.append(cues, ed)

    return cues


def decimate_cues(cues: np.ndarray, min_len: int = 256, keep_last: bool = True) -> np.ndarray:
    """
    Remove cues too close from their previous cues
    :param cues:
    :param min_len: Min length between cues (in samples)
    :param keep_last: Always keep last cue
    :return: processed cues
    """
    diff = np.diff(cues)
    result = np.append(cues[0], cues[1:][diff >= min_len])
    if keep_last and cues[-1] not in result:
        result = np.append(result, cues[-1])
    return result


def write_map_file(mapping: list[tuple[int | float],]) -> tempfile.NamedTemporaryFile:
    """
    Write a map file used by rubberband-cli
    :param mapping: List of cue pairs defined as tuples (cue,value)
    :return: Map file path
    """
    map_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    for t in mapping:
        map_file.write(f'{t[0]} {t[1]}\n')
    map_file.close()
    return map_file


def split_noise_tonal(audio: np.ndarray, sr: int, stft_size: int = 4096, iterations: int = 3,
                      end_mode: str | None = 'loop', loop: tuple[int, int] = None):
    """
    Split noise and tonal content

    :param audio: Input audio array
    :param sr: Sampling rate
    :param stft_size: transient vs frequency precision trade-off

    :param iterations: Number of times the process is repeated

    :param end_mode: Method used to process end boundary
    'loop' use provided loop start and end using loop argument
    'extend' repeat last stft frames
    'fade' fade out end samples with a duration of stft_size/4

    :param loop: Loop start, Loop end

    :return: noise, tonal
    """
    length = len(audio)
    au_in = np.copy(audio)

    if length < stft_size:
        au_in = pad_audio(au_in, before=0, after=stft_size - length, mode='constant')

    nch = au_in.ndim

    noise = None

    for c in range(nch):
        if nch > 1:
            chn_data = au_in[:, c]
        else:
            chn_data = au_in

        win = np.hanning(stft_size)

        sft = ShortTimeFFT(win=win, hop=stft_size // 4, fs=sr)

        match end_mode:
            case 'loop':
                if loop is not None:
                    chn_data = np.concatenate([chn_data[:loop[-1] + 1], chn_data[loop[0]:loop[-1] + 1]])
            case 'extend':
                # Repeat end frames
                sft_z = sft.stft(chn_data)
                end_frames = sft_z[:, -6:-4:]
                sft_z = np.concatenate([sft_z[:, :-4:], np.repeat(end_frames, 2, axis=1)], axis=1)
                chn_data = np.real(sft.istft(sft_z))
            case 'fade':
                # Fade end
                fd_out_len = stft_size // 4
                fd_out = np.append(np.ones(len(chn_data) - fd_out_len), np.hanning(fd_out_len * 2)[fd_out_len:])
                chn_data *= fd_out
            case _:
                pass

        if noise is None:
            if nch > 1:
                noise = np.zeros(shape=(len(chn_data), nch))
            else:
                noise = np.zeros(len(chn_data))
            tonal = np.zeros_like(noise)

        for i in range(iterations):
            # STFT
            sft_z = sft.stft(chn_data)

            mag = np.abs(sft_z)
            phase = np.angle(sft_z)

            # Median filter to isolate noise content
            noise_mag = ndimage.median_filter(mag, size=(65, 1))
            sft_noise = noise_mag * np.exp(1j * phase)
            ns = np.real(sft.istft(sft_noise))[:len(chn_data)]
            # print(len(ns), len(noise))

            chn_data -= ns
            if nch > 1:
                noise[:, c] += ns
            else:
                noise += ns

        if nch > 1:
            tonal[:, c] = chn_data
        else:
            tonal = chn_data

    # Trim to original length
    noise = noise[:length]
    tonal = tonal[:length]

    # Prevent clipping
    mx = max(peak(noise), peak(tonal))
    if mx >= 1:
        nrm = mx / db_to_lin(-.1)
        noise /= nrm
        tonal /= nrm

    return noise, tonal


# Additional functions
def lerp_angle(a: float | np.ndarray, b: float | np.ndarray, x: float | np.ndarray) -> float | np.ndarray:
    """
    Linear shortest interpolation between two angles a and b (radians)
    :param a: angle in radians
    :param b: angle in radians
    :param x: Blend factor between a and b
    :return: Interpolated angle
    """
    max_angle = np.pi * 2
    da = (b - a) % max_angle
    sa = 2 * da % max_angle - da
    return a + sa * x


def phase_randomizer(audio: np.ndarray, sr: int = 44100, stft_size: int = 16000, spread: float = .5, seed=42):
    """
    Perform stft then randomize phase each frequency bin of input audio
    :param audio:
    :param sr:
    :param stft_size: stft size in samples
    a small size preserve transients better though with lower frequency precision
    a larger size smear transients better but has better frequency precision
    :param spread: Randomization amplitude
    :param seed: Randomization seed
    :return: Processed signal
    """
    length = len(audio)
    nperseg = min(length // 8, stft_size)

    # STFT
    win = np.hanning(nperseg)
    sft = ShortTimeFFT(win=win, hop=nperseg // 4, fs=sr)
    sft_z = sft.stft(audio)
    mag, phase = np.abs(sft_z), np.angle(sft_z)

    # Phase shift
    rng = np.random.default_rng(seed)
    phase_shift = (rng.random(sft_z.shape) - 0.5) * 2 * np.pi
    sft_z = mag * np.exp(1j * lerp_angle(phase, phase + phase_shift, spread))

    # Inverse STFT and clamp it to original size
    result = np.real(sft.istft(sft_z))[:length]

    # Prevent clipping
    mx = peak(result)
    if mx >= 1:
        result /= mx / db_to_lin(-.1)

    return result


def cues_to_regions(cues: np.ndarray) -> np.ndarray:
    if len(cues) < 2:
        return np.array([])
    return np.column_stack((cues[:-1], cues[1:]))
