# coding:utf-8
"""
    :module: split_audio.py
    :description:
    :author: Michel 'Mitch' Pecqueur
    :date: 2024.05
"""

import importlib
import math
import os.path
from collections import namedtuple
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import interpolate

import pitch_detect as pd
from sample_utils import append_markers, is_stereo
from utils import append_metadata, set_md_tags
from utils import note_to_name, hz_to_note
from common_math_utils import q_log, q_exp

importlib.reload(pd)


def split_audio(input_file='', output_file='', bit_depth=None,
                suffix=(), pitch_mode='pyin', use_note=0, use_pitch_fraction=True,
                extra_suffix='',
                split_db=-60, fade_db=-48, min_duration=.1,
                dither=True, dc_offset=True,
                write_cue_file=True, progress_pb=None, dry_run=True):
    """
    :param str input_file:
    :param str output_file:
    :param list or tuple or None suffix:
    :param str or None extra_suffix:

    :param int or None bit_depth: If None use input bit-depth
    :param bool write_cue_file: Write a non-split file with cues

    :param bool dc_offset:
    :param bool dither: Apply triangular noise to improve quantization distortion when writing to 16 bits
    Only applied to fades

    :param float split_db: Split threshold
    :param float or None fade_db: Fade threshold
    :param float min_duration: Minimum silence/sound duration

    :param str pitch_mode: 'corr' (fastest), 'pyin' (average) or 'crepe' (slow)
    :param int use_note: 0 disabled, 1 MIDI note number, 2 note name
    :param bool use_pitch_fraction: Set pitch fraction to 'smpl'

    :param QProgressBar or None progress_pb: Optional

    :param bool dry_run: Simulate process without writing anything, for debugging
    :return:
    :rtype: list
    """
    audio, sr = sf.read(input_file)

    if audio.ndim > 1:
        mono_audio = audio.mean(axis=1)
    else:
        mono_audio = audio

    window_size = max(int(min_duration * sr / 2), 64)
    envelope = envelope_transform(mono_audio, w=window_size, mode='max')
    region_data = find_regions(envelope, sr, db=split_db, min_duration=min_duration)
    region_data = trim_regions(md=region_data, data=mono_audio, db=split_db)

    regions, cues = region_data.regions, region_data.cues

    p = Path(output_file)
    name, ext = p.stem, p.suffix[1:]
    path = p.parent

    if not os.path.exists(path) and not dry_run:
        os.makedirs(path, exist_ok=True)

    if write_cue_file:
        cue_file_path = Path.joinpath(path, f'{name}_cues.wav')
        if not dry_run:
            sf.write(cue_file_path, audio, samplerate=sr)
            append_markers(str(cue_file_path), cues.tolist())
        else:
            print(f'Cue file: {cue_file_path}')

    # - Bit Depth -
    subtypes = {'PCM_16': 16, 'PCM_24': 24, 'FLOAT': 32}
    bd_dict = {v: k for k, v in subtypes.items()}
    if bit_depth is None:
        subtype = sf.info(input_file).subtype
        bit_depth = subtypes[subtype]
    # flac supports 24 bits at most
    if ext == 'flac':
        bit_depth = min(bit_depth, 24)

    # Integer peak value
    mx = 1.0
    if bit_depth <= 24:
        mx = 2 ** (bit_depth - 1) - 1

    suffix = suffix or []
    split_files = []

    # Progress bar init
    pb_val, pb_mx, count = None, None, len(regions.tolist())
    if progress_pb:
        # Get current bar progress
        pb_val = progress_pb.value()
        pb_mx = progress_pb.maximum()
        # "Subdivide" progress bar according to found regions
        progress_pb.setMaximum(pb_mx * count)
        progress_pb.setValue(pb_val * count)

    for i, region in enumerate(regions.tolist()):
        filepath = ''
        data = audio[region[0]: region[1] + 1]
        mono_data = mono_audio[region[0]: region[1] + 1]

        # Keep only one channel if audio is not stereo
        if not is_stereo(data, db=-48):
            data = mono_data

        note, pitch_fraction = 60, 0
        md = {}
        freq = None
        if use_note:
            freq = pd.pitch_detect(mono_data, sr, mode=pitch_mode, resample=None, st_ed=.25)
            if np.isnan(freq) or freq is None:
                filepath = Path.joinpath(Path(path), f'{name}_pitchfail{i + 1:03d}.{ext}')
            else:
                pitch = hz_to_note(freq)
                note = int(round(pitch))

                # Pitch fraction is negative pitch Correction / tuning
                pitch_fraction = round((pitch - note) * 100, 3)

                pitch_fraction = (0, pitch_fraction)[use_pitch_fraction]
                md['note'], md['pitchFraction'] = note, pitch_fraction
                print(f'{i + 1:03d}', md)
                if use_note == 2:
                    note_name, octave = note_to_name(note)
                    filepath = Path.joinpath(Path(path), f'{name}_{note_name}{octave}.{ext}')
                else:
                    filepath = Path.joinpath(Path(path), f'{name}_{note:03d}.{ext}')

        if not use_note or freq is False:
            if i < len(suffix):
                filepath = Path.joinpath(Path(path), f'{name}_{suffix[i]}.{ext}')
            else:
                filepath = Path.joinpath(Path(path), f'{name}_{i + 1:03d}.{ext}')

        if extra_suffix:
            filepath = filepath.parent.joinpath(f'{filepath.stem}{extra_suffix}.{ext}')

        # Prevent overwriting
        incr = 0
        p = Path(filepath)
        while filepath in split_files:
            incr += 1
            filepath = Path.joinpath(Path(path), f'{p.stem}_{incr:03d}.{ext}')

        if fade_db is not None:
            _, fade_cues = trim_audio(data, db=fade_db)

            if dc_offset:
                dc_offset = np.mean(data)
                data -= dc_offset

            orig_data = np.array(data)
            data = apply_fade(data, fade_in=(0, fade_cues[0], 'log'),
                              fade_out=(fade_cues[1], len(data) - 1 - fade_cues[1], 'log'))

            if dither is True and bit_depth < 24:
                dither_mask = (np.abs(orig_data - data) > 0).astype(np.float64)
                dither = dither_mask * np.random.triangular(-1, 0, 1, data.shape)
                data = np.round(data * mx + dither).astype(np.int16)

        if not dry_run:
            # Soundfile only recognizes aiff and not aif when writing
            sf_path = (filepath, f'{filepath}f')[ext == 'aif']
            sf.write(str(sf_path), data, samplerate=sr, subtype=bd_dict[bit_depth])
            if sf_path != filepath:
                os.rename(sf_path, filepath)
        else:
            print(f'Split {i + 1:03d}: {filepath}')

        # Add 'smpl' metadata
        if use_note:
            if ext == 'wav':
                append_metadata(str(filepath), note=note, pitch_fraction=pitch_fraction, loop_start=None, loop_end=None)
            # Add metadata as tags for flac
            elif ext == 'flac':
                set_md_tags(str(filepath), md=md)

        # Increment progress bar sub-task
        if progress_pb is not None:
            progress_pb.setValue(pb_val * count + i + 1)

        split_files.append(filepath)

    # Restore progress bar maximum and increment parent task
    if progress_pb is not None:
        progress_pb.setMaximum(pb_mx)
        progress_pb.setValue(pb_val + 1)

    return split_files


# Auxiliary defs


def trim_audio(data, db=-60):
    """
    Remove trailing silence
    :param np.array data: Input audio
    :param float db: Silence threshold in dB
    :return: processed audio
    :rtype: np.array
    """
    if data.ndim > 1:
        mono_audio = data.mean(axis=-1)
    else:
        mono_audio = data

    th = np.power(10, db / 20)
    silence = np.abs(mono_audio) < th
    idx = np.where(silence == 0)[0]

    trim = namedtuple('trim', 'data cues')
    result = trim(data[idx[0]:idx[-1] + 1], (idx[0], idx[-1]))

    return result


def find_regions(envelope, sr, db=-60, min_duration=.1):
    """
    Detect regions from a volume envelope

    :param np.array envelope: Input envelope

    :param int sr: Sample rate for duration
    :param float db: Silence threshold in dB
    :param float min_duration: in s
    :return: Return regions and cues as a named tuple

    :rtype: namedtuple
    """

    th = np.power(10, db / 20)
    silence = np.abs(envelope) < th

    mn_d = int(min_duration * sr)

    # Get index when silence changes
    change = np.argwhere(np.diff(silence) != 0).reshape(-1)
    change = np.sort(np.append(change, [0, len(envelope) - 1]))
    # Remove indices too close from each other
    idx = np.argwhere(np.diff(change) >= mn_d).reshape(-1)
    idx = np.append(change[idx], len(envelope) - 1)

    # Keep regions with audio and "blob" together neighboring regions
    cues = []
    for i in range(len(idx) - 1):
        st, ed = idx[i:i + 2].tolist()
        if np.mean(silence[st + 1:ed]) < .5:
            if cues:
                if cues[-1] != st:
                    cues.append(st)
                else:
                    cues.pop(-1)
            else:
                cues.append(st)
            cues.append(ed)

    cues = np.array(cues)
    regions = cues.reshape(-1, 2)

    metadata = namedtuple('metadata', 'regions cues')

    result = metadata(regions, cues)

    return result


def trim_regions(md, data, db=-60):
    """
    Refine regions by trimming from audio data
    :param namedtuple md: regions,cues metadata
    :param np.array data: Audio data, mono or stereo
    :param float db: Silence threshold
    :return: Processed metadata
    :rtype: namedtuple
    """
    cues = []
    for region in md.regions:
        trim = trim_audio(data[region[0]:region[1] - 1], db=db)
        st, ed = trim.cues
        cues.extend([region[0] + st, region[0] + ed])
    cues = np.array(cues)
    regions = cues.reshape(-1, 2)

    metadata = namedtuple('metadata', 'regions cues')
    result = metadata(regions, cues)

    return result


def envelope_transform(data, w=1024, mode='max', interp='linear'):
    """
    Transform input audio to volume envelope
    :param np.array data: Input single channel audio
    :param int w: Rolling window size
    :param str mode: 'max' for peak or 'rms' for root-mean-square
    :param str interp: Interpolation type 'linear' or 'cubic'
    'linear' is better for volume estimation (no overshoots)
    'cubic' is better for shaping (introduces less distortion)
    :return:
    :rtype: np.array
    """

    length = data.size
    half_w = w // 2
    hop = half_w

    # Pad data so rolling window is centered on corresponding sample
    pad_data = np.pad(data, half_w, mode='edge')

    steps = math.ceil(length / half_w)
    result = np.zeros(steps)
    for i in range(steps):
        frame = pad_data[i * hop:i * hop + w]
        if mode == 'max':
            result[i] = np.max(np.abs(frame))
        elif mode == 'rms':
            result[i] = np.sqrt(np.mean(frame ** 2))

    # Interpolate result preserving original length
    x = np.linspace(0, 1, steps)
    x_new = np.linspace(0, 1, steps * hop)
    result = interpolate.interp1d(x, result, kind=interp)(x_new)[:length]

    return result


# Fade in/out functions

def apply_fade(data=None, fade_in=(0, 100, 'log'), fade_out=(500, 32000, 'exp')):
    """
    Apply fade in/out to audio
    :param np.array data: Input audio
    :param tuple fade_in: Start, Duration, Curve
    :param tuple fade_out: Start, Duration, Curve
    :return: Processed audio
    :rtype: np.array
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
