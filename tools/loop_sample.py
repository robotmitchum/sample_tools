# coding:utf-8
"""
    :module: loop_sample.py
    :description: Functions to detect and set loops to samples
    Support for wav and flac
    :author: Michel 'Mitch' Pecqueur
    :date: 2024.06
"""

import importlib
import os
from collections import namedtuple
from pathlib import Path

import numpy as np
import soundfile as sf

import crossfade_utils as cu
import fft_resynth as fftr
from common_audio_utils import db_to_lin
from common_math_utils import lerp, clamp, smoothstep, q_exp, q_log
from file_utils import resolve_overwriting
from sample_utils import Sample
from split_audio import envelope_transform
from utils import append_metadata, set_md_tags, name_to_note, note_to_hz, hz_to_period

importlib.reload(fftr)
importlib.reload(cu)


def loop_sample(input_file='', output_file='', bit_depth=None,
                env_window=50,
                shape_env={'env_threshold': -6, 'env_min': -30, 'env_mode': 'loop'},
                detect_loop={'n_cues': 90000, 'min_len': 100, 'window_size': 10, 'window_offset': .5,
                             'start_range': 1, 'end_range': 0, 'hash_search': False},
                crossfade={'fade_in': .5, 'fade_out': .5, 'mode': 'linear'},
                resynth={'fft_range': 'custom', 'start': 0.333, 'fft_size': 1.0, 'duration': 2.0,
                         'atonal_mix': 1, 'freq_mode': 'note_pf', 'freqs': None,
                         'resynth_mix': 'loop_tail', 'fade_in': .5, 'fade_out': .5, 'width': .5},
                trim_after=False, no_overwriting=True, progress_pb=None):
    """
    Loop an audio sample

    - Envelope shaping pre-proces to help looping
    - Detect and set a loop region for a given audio file (keep current loop info if disabled)
    - Apply a cross-fade to loop

    :param str input_file:
    :param str or None output_file: if None return resulting data
    :param int bit_depth:

    :param int env_window: Envelope window length in ms, also used loop detection target in 'db' mode

    :param dict or None or bool shape_env: Envelope shaping settings
    {'env_threshold': -6, 'env_min': -30, 'env_mode': 'loop'}

    :param dict or None or bool detect_loop: Loop detection settings
    {'tgt_mode': str ('percent', 'db' or 'samples'),
    'target': float or int,
    'min_len': int (minimum loop length ms),
    'window_size': int (self-correlation window length in ms)}

    :param dict or None or bool crossfade: Cross-fade settings
    {'fade_in': float (% of loop length),
    'fade_out': float (% of loop length),
    'mode': str (fade mode 'linear', 'smoothstep' or 'exp')}

    :param bool trim_after: Trim file after loop end

    :param bool no_overwriting:

    :param object progress_pb: optional QProgressBar

    :return: output result, sample info, output file path
    :rtype: namedtuple
    """

    audio, sr = sf.read(input_file)
    orig_audio = np.copy(audio)
    info = Sample(input_file)
    loop = info.loopStart, info.loopEnd

    if not detect_loop and not loop[0] and not loop[1] and not resynth:
        print(f'{input_file} does not have loop information, nothing done')
        return None

    if output_file:
        p = Path(output_file)
        name, ext = p.stem, p.suffix[1:]
        path = p.parent

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    nch = audio.ndim
    if nch > 1:
        mono_audio = np.mean(audio, axis=-1)
    else:
        mono_audio = audio

    # Envelope shaping, straighten waveform to make it easier to loop of for sound design
    use_au_b = False
    if shape_env:
        use_au_b = bool(shape_env['env_mode'] == 'loop')
        env = envelope_transform(mono_audio, w=int(sr * env_window / 1000), interp='cubic')
        # Limit envelope
        env_threshold = db_to_lin(shape_env['env_threshold'])
        env_min = db_to_lin(shape_env['env_min'])
        env = np.clip(env, a_min=env_min, a_max=env_threshold) / env_threshold
        if nch > 1:
            env = np.tile(env, (nch, 1)).T
        audio /= env

    if detect_loop:
        n_cues = detect_loop['n_cues']
        min_len = int(sr * detect_loop['min_len'] / 1000)
        window_size = int(sr * detect_loop['window_size'] / 1000)
        window_offset = detect_loop['window_offset']
        start_range = detect_loop['start_range']
        end_range = detect_loop['end_range']
        hash_search = detect_loop['hash_search']

        loop = None
        if hash_search:
            loop = search_loop_hashing(audio, window_size=64)
        if loop is None:
            loop = search_loop(audio, n_cues=n_cues, min_len=min_len,
                               window_size=window_size, window_offset=window_offset,
                               start_range=start_range, end_range=end_range,
                               progress_pb=progress_pb)

        info.loopStart, info.loopEnd = loop
        info.loops = [loop]

    # Apply Cross-Fade
    if crossfade:
        fade_in, fade_out = crossfade['fade_in'], crossfade['fade_out']
        au_b = (None, orig_audio)[use_au_b]
        if fade_in > 1:
            audio = apply_crossfade(audio=audio, au_b=None, loop=loop, fade_in=fade_in, fade_out=0,
                                    mode=crossfade['mode'])
        audio = apply_crossfade(audio=audio, au_b=au_b, loop=loop, fade_in=min(1, fade_in), fade_out=fade_out,
                                mode=crossfade['mode'])

    # FFT re-synthesis
    if resynth:
        if resynth['fft_range'] == 'custom':
            fft_start = int(resynth['fft_start'] * len(audio))
            fft_size = min(int(resynth['fft_size'] * len(audio)), len(audio) - fft_start)
        else:
            fft_start = info.loopStart
            fft_size = info.loopEnd - info.loopStart

        freqs = []
        if resynth['freq_mode'] == 'note' and info.note is not None:
            freqs.append(note_to_hz(info.note))
        elif resynth['freq_mode'] == 'note_pf' and info.note is not None:
            pf = (info.pitchFraction, 0)[info.pitchFraction is None]
            freqs.append(note_to_hz(info.note + pf / 100))
        if resynth['freqs']:
            freqs.extend(resynth['freqs'])

        print('Freqs: ', freqs)
        make_stereo = (False, True)[resynth['resynth_mix'] == 'all']

        width = resynth['width']
        atonal_mix = resynth['atonal_mix']
        normalize = (None, -1)[resynth['resynth_mix'] == 'all']
        duration = resynth['duration']
        print('Normalize:', normalize)

        # Synthesize audio from user settings
        resynth_data = fftr.fft_resynth(input_file=None, input_data=np.copy(audio), sr=sr, target_sr=None,
                                        start=fft_start, fft_size=fft_size, freqs=freqs,
                                        make_stereo=make_stereo, seed=0, width=width,
                                        atonal_mix=atonal_mix, duration=duration, normalize=normalize,
                                        output_file=None)
        loop_len = len(resynth_data)

        fade_in = int(resynth['fade_in'] * len(resynth_data))
        fade_out = int(resynth['fade_out'] * len(resynth_data))

        # Mix re-synthesis with original audio
        if resynth['resynth_mix'] == 'loop_tail':
            # Replace loop tail
            print('loop_tail')
            resynth_len = fade_in + loop_len
            reps = int(np.ceil(resynth_len / loop_len))
            if resynth_data.ndim > 1:
                reps = (reps, 1)
            print('reps:', reps)
            resynth_data = np.tile(resynth_data, reps)[:resynth_len + 1]
            audio = cu.crossfade_clips(audio, resynth_data, fft_start, fade_in, fade_type='equal_power')
            info.loopStart = fft_start + fade_in
            info.loopEnd = info.loopStart + loop_len - 1
        elif resynth['resynth_mix'] == 'loop':
            # Replace loop only
            print('loop')
            resynth_len = fade_in + loop_len + fade_out
            reps = int(np.ceil(resynth_len / loop_len))
            if resynth_data.ndim > 1:
                reps = (reps, 1)
            print('reps:', reps)
            resynth_data = np.tile(resynth_data, reps)[:resynth_len + 1]
            print('rs:', resynth_data.shape)
            cf1 = cu.crossfade_clips(audio, resynth_data, fft_start, fade_in, fade_type='equal_power')
            audio = cu.crossfade_clips(cf1, audio[fft_start + fade_in:], fft_start + fade_in + loop_len, fade_out,
                                       fade_type='equal_power')
            print('au:', audio.shape)
            info.loopStart = fft_start + fade_in
            info.loopEnd = info.loopStart + loop_len - 1
        else:
            # Replace everything
            print('all')
            audio = resynth_data
            info.loopStart = 0
            info.loopEnd = len(resynth_data) - 1

    # Trim after loop
    if trim_after:
        audio = audio[:info.loopEnd + 1]

    mx = np.max(np.abs(audio))
    if mx > 1:
        audio /= mx / db_to_lin(-.5)

    result = namedtuple('_result', ('audio', 'info', 'output_file'))

    if output_file is None:
        res = result(audio, info, None)
        return res

    # - Bit Depth -
    subtypes = {'PCM_16': 16, 'PCM_24': 24, 'FLOAT': 32}
    bd_dict = {v: k for k, v in subtypes.items()}
    if bit_depth is None:
        subtype = sf.info(input_file).subtype
        bit_depth = subtypes[subtype]
    # flac supports 24 bits at most
    if ext == 'flac':
        bit_depth = min(bit_depth, 24)

    if no_overwriting and str(output_file) == input_file:
        resolve_overwriting(input_file, mode='dir', dir_name='backup_', test_run=False)

    # Write file
    sf_path = (output_file, f'{output_file}f')[ext == 'aif']
    sf.write(sf_path, audio, samplerate=sr, subtype=bd_dict[bit_depth])
    if sf_path != output_file:
        os.rename(sf_path, output_file)

    # Append metadata
    if ext == 'wav':
        append_metadata(str(output_file), note=info.note, pitch_fraction=info.pitchFraction, loop_start=info.loopStart,
                        loop_end=info.loopEnd)
    # Add metadata as tags for flac
    elif ext == 'flac':
        keys = ['note', 'pitchFraction', 'loopStart', 'loopEnd', 'cues']
        values = [getattr(info, value) for value in keys]
        md = dict(zip(keys, values))
        set_md_tags(str(output_file), md=md)

    res = result(audio, info, output_file)

    return res


# Loop detection


def search_loop(audio, n_cues=768, min_len=2048, window_size=512, window_offset=.5,
                start_range=1, end_range=0, progress_pb=None):
    """
    Search Loop by auto-correlation, find both start and end points
    :param np.array audio: Input array
    :param float window_offset: search direction, 0 backward, .5 centered, 1 forward
    :param int n_cues: Nax number of cues to consider
    :param min_len: Minimum loop length (in samples)
    :param window_size: (in samples)
    :return: loop_start, loop_end
    :param float start_range:
    :param float end_range:
    :param QProgressBar progress_pb: Optional progress bar
    :rtype: list
    """
    if audio.ndim > 1:
        mono_audio = np.mean(audio, axis=-1)
    else:
        mono_audio = audio

    # Progress bar init
    pb_val, pb_mx, pb_fmt, pb_steps = None, None, '', 100
    if progress_pb:
        # Get current bar progress
        pb_val = progress_pb.value()
        pb_mx = progress_pb.maximum()
        # pb_fmt = progress_pb.format()

        # "Subdivide" progress bar according to sub-task count
        progress_pb.setMaximum(pb_mx * pb_steps)
        progress_pb.setValue(pb_val * pb_steps)
        # progress_pb.setFormat('Searching loop... %p%')
        progress_pb.update()

    zc_cues = zero_crossing_idx(mono_audio, mode=1)

    # Window offset
    post = int(np.round(window_size * window_offset))
    pre = window_size - post

    min_start_idx = find_idx(zc_cues, x=pre, direction=1)

    # Start range indices
    if start_range is not None:
        max_start_pos = clamp(int(len(mono_audio) * start_range), 0, len(mono_audio)) - 1
    else:
        max_start_pos = len(mono_audio) - 1
        start_range = 1

    # max_start_idx = find_idx(zc_cues, x=max_start_pos, direction=-1)

    # End range indices
    mn_end_pos = min_len - 1
    if end_range is not None:
        mn_end_pos = clamp(mn_end_pos, int(len(mono_audio) * end_range), len(mono_audio)) - 1
    else:
        end_range = 0
    min_end_idx = find_idx(zc_cues, x=mn_end_pos, direction=-1)

    # Candidates decimation weighting
    range_wt = np.array([start_range, 1 - end_range])
    n_range = np.sqrt(n_cues / np.prod(range_wt))
    range_cues = np.round(range_wt * n_range).astype(np.int32)

    # End range decimation
    end_idx_range = range(len(zc_cues) - 1, min_end_idx - 1, -1)
    end_idx_range = decimate_array(end_idx_range, range_cues[1])
    if len(end_idx_range) < range_cues[1]:
        range_cues[1] = len(end_idx_range)
        range_cues[0] = min(int(np.round(n_cues / len(end_idx_range))), len(zc_cues))

    print(f'{len(zc_cues)} cues, using {np.prod(range_cues)} combinations')

    count = len(end_idx_range)

    if count < 1:
        raise NameError('No end candidate')

    loop_start, loop_end = 0, len(audio) - 1

    if not progress_pb:
        print('[', end='')

    min_error = -1
    update_last = 0

    for i, end_idx in enumerate(end_idx_range):
        # Loop end search
        end_pos = zc_cues[end_idx]

        # Throttle progress updates
        update_value = round(i / count * 10)
        update_progress = update_value > update_last
        update_last = update_value

        if not progress_pb and update_progress:
            print('=', end='')

        ref_window = mono_audio[end_pos - pre:end_pos + post]
        if len(ref_window) < window_size:
            continue

        amp = np.std(ref_window)
        if amp < 1e-5:
            continue

        max_start_idx = find_idx(zc_cues, x=min(end_pos - min_len, max_start_pos), direction=-1)
        start_idx_range = range(min_start_idx, max_start_idx)
        start_idx_range = decimate_array(start_idx_range, range_cues[0])

        # Loop start search
        for start_idx in start_idx_range:
            start_pos = zc_cues[start_idx]
            window = mono_audio[start_pos - pre:start_pos + post]
            if len(window) < window_size:
                continue

            error = np.mean(np.abs(ref_window - window)) / amp

            if error < min_error or min_error == -1:
                min_error = error
                loop_start, loop_end = start_pos, end_pos

        # Increment progress bar sub-task
        if progress_pb is not None and update_progress:
            pr = int((i + 1) / len(end_idx_range) * 100)
            progress_pb.setValue(pb_val * pb_steps + pr)

    if not progress_pb:
        print(']')

    # Recall progress bar initial state and increment parent task
    if progress_pb is not None:
        progress_pb.setMaximum(pb_mx)
        progress_pb.setValue(pb_val + 1)
        # progress_pb.setFormat(pb_fmt)

    return [int(loop_start), int(loop_end - 1)]


def search_loop_hashing(audio, window_size=64, tol=1e-5):
    """
    Search for a perfect loop if applicable, typically in already processed audio (baked in cross-fade loop)
    :param audio:
    :param int window_size: Used for comparison
    :param float tol: tolerance for comparison
    :return:
    :rtype: list or None
    """
    hashes = {}
    for start in range(len(audio) - window_size):
        window = audio[start:start + window_size]
        if np.std(window) < tol:
            continue
        window_hash = hash(window.tobytes())
        if window_hash in hashes:
            prev_start = hashes[window_hash]
            prev_window = audio[prev_start:prev_start + window_size]
            if np.allclose(window, prev_window, atol=tol):
                return [prev_start, start - 1]
        else:
            hashes[window_hash] = start

    return None


# Auxiliary function
def apply_crossfade(audio, loop, fade_in=.5, fade_out=.25, mode='linear', au_b=None):
    """
    Applies cross-fade to audio to further smooth looped playback
    :param np.array audio: Input Audio
    :param list or tuple loop: Loop start/ End values
    :param float fade_in: Crossfade, in percent of loop length
    :param float fade_out: Post crossfade out, clamped, so it can't be longer than audio length
    :param str or None mode: Interpolation type, 'linear' or 'smoothstep'
    :param np.array au_b: Blend to optional signal outside of loop
    :return: Crossfaded audio
    :rtype: np.array
    """

    loop_start, loop_end = loop
    loop_len = loop_end - loop_start + 1

    fadein_len = min(int(round(loop_len * fade_in)), loop_end)
    fadein_start = loop_end - fadein_len
    fadeout_len = min(int(round(loop_len * fade_out)), len(audio) - loop_end)

    # Fade array
    fade = np.zeros(fadein_start)
    fade = np.append(fade, np.linspace(0, 1, fadein_len, endpoint=True))
    fade = np.append(fade, np.linspace(1, 0, fadeout_len, endpoint=True))
    fade = np.pad(fade, pad_width=(0, len(audio) - len(fade)), constant_values=(0, 0))

    # Fade mode
    if mode == 'smoothstep':
        fade = smoothstep(0, 1, fade)
    elif mode == 'exp':
        fade = q_exp(fade)

    # Multichannel
    nch = audio.ndim
    if nch > 1:
        fade = np.tile(fade, (nch, 1)).T

    # Mix original with delayed audio
    delayed = np.roll(audio, loop_len, axis=0)
    result = lerp(audio, delayed, fade)

    # Optional mix outside of loop
    if au_b is not None:
        result_b = lerp(au_b, np.roll(au_b, loop_len, axis=0), fade)

        fadein_len = min(int(round(loop_len * fade_in)), loop_start)
        fadein_start = loop_start - fadein_len
        fadeout_len = min(int(round(loop_len * fade_out)), len(audio) - loop_end)

        fade = np.zeros(fadein_start)
        fade = np.append(fade, q_log(np.linspace(0, 1, fadein_len, endpoint=True)))
        fade = np.append(fade, np.ones(loop_len - 1))
        fade = np.append(fade, q_exp(np.linspace(1, 0, fadeout_len, endpoint=True)))
        fade = np.pad(fade, pad_width=(0, len(audio) - len(fade)), constant_values=(0, 0))

        if nch > 1:
            fade = np.tile(fade, (nch, 1)).T

        result = lerp(result_b, result, fade)

    return result


# Utility functions
def zero_crossing_idx(data, mode=1):
    """
    Return zero crossing indices
    :param np.array data:
    :param int mode: Negative(-1), Positive(1) or Absolute (0)
    :return:
    :rtype: np.array
    """
    diff = np.diff(np.sign(data))
    if mode > 0:
        return np.argwhere(diff > 0).reshape(-1) + 1
    elif mode < 0:
        return np.argwhere(diff < 0).reshape(-1) + 1
    else:
        return np.argwhere(np.abs(diff) > 0).reshape(-1) + 1


def decimate_array(data, target_size):
    """
    Reduce the number of items of a numpy array
    :param np.array data: Input array
    :param int target_size: Desired size of the array
    :return: Decimated array
    :rtype: np.array
    """
    current_size = len(data)
    if target_size >= current_size:
        return data
    indices = np.linspace(0, current_size - 1, target_size).astype(np.int32)
    return np.array(data)[indices]


def find_idx(data, x, direction=0):
    """
    Return closest index to target value in input array
    :param np.array data: input array
    :param int x: Target value
    :param int direction: Search direction, previous -1, closest 0, next 1
    :return:
    :rtype: int
    """
    idx = np.argmin(np.abs(data - x))
    zc = data[idx]
    if direction < 0 and x <= zc:
        idx -= 1
    elif direction > 0 and x >= zc:
        idx += 1
    idx = clamp(idx, 0, len(data) - 1)
    return np.clip(idx, 0, len(data))


# Extra functions

def ms_to_smp(ms, sr):
    return int(sr * ms / 1000)


def pitch_correction_from_loop(note, loop_start, loop_end, sr):
    """
    Calculate pitch correction from loop
    Only works when loop cues match periods of the signal
    Typical usage : correct pitch for very short loops (a few periods)
    :param int or str note:
    :param loop_start:
    :param loop_end:
    :param int sr:
    :return: Pitch correction in semitone cents
    """
    if isinstance(note, str):
        pitch = name_to_note(note)
    else:
        pitch = note
    period = hz_to_period(note_to_hz(pitch), sr)

    loop_len = loop_end - loop_start
    pitch = sr / (loop_len / round(loop_len / period))
    pitch_correction = round((round(pitch) - pitch) * 100)

    return int(pitch_correction)


def gen_intervals(seed=123456, step=567, mn=0, mx=123456):
    """
    Return an array of evenly spaced integers given a seed in the array
    :param seed: Seed value
    :param step: Gap between values
    :param int mn: Min value
    :param int mx: Max value
    :return:
    :rtype: np.array
    """
    offset = seed % step
    start = mn - (mn % step) + offset
    return np.arange(start, mx + 1, step)


def filter_array(data, tgt, md=16):
    """
    Filter values of an array with values of another keeping only value within a given radius
    :param np.array data: Source array
    :param np.array tgt: Target array
    :param int md: Radius or max distance
    :return:
    :rtype: np.array
    """
    result = np.array([], dtype=np.int32)
    for item in tgt:
        idx = np.argwhere(np.abs(data - item) < md).reshape(-1)
        result = np.append(result, data[idx])
    return result
