# coding:utf-8
"""
    :module: split_audio.py
    :description:
    :author: Michel 'Mitch' Pecqueur
    :date: 2024.05
"""

import math
import os.path
import tempfile
from collections import namedtuple
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import interpolate

import pitch_detect as pd
from common_math_utils import q_log, q_exp
from file_utils import resolve_overwriting
from sample_utils import append_markers, is_stereo, info_from_name
from utils import append_metadata, get_md_tags, set_md_tags
from utils import note_to_name, hz_to_note


# importlib.reload(pd)


def split_audio(input_file: Path | str = '', output_file: Path | str = '', bit_depth: int | None = None,
                suffix: list[str] | None = (), pitch_mode: str = 'pyin', use_note: int | None = 0,
                use_pitch_fraction: bool = True,
                extra_suffix: str | None = '',
                split_db: float = -60, fade_db: float = -48, min_duration: float = .1,
                dither: bool = False, dc_offset: bool = True,
                write_cue_file: bool = True, dry_run: bool = True,
                progress_bar: None = None,
                worker=None, progress_callback=None, message_callback=None, range_callback=None) -> list[Path]:
    """
    :param input_file:
    :param output_file:
    :param suffix:
    :param extra_suffix:

    :param bit_depth: If None use input bit-depth
    :param write_cue_file: Write a non-split file with cues

    :param dc_offset:
    :param dither: Apply triangular noise to improve quantization distortion when writing to 16 bits
    Only applied to fades

    :param split_db: Split threshold
    :param fade_db: Fade threshold
    :param min_duration: Minimum silence/sound duration

    :param pitch_mode: 'corr' (fastest), 'pyin' (average) or 'crepe' (slow)
    :param use_note: 0 disabled, 1 MIDI note number, 2 note name
    :param use_pitch_fraction: Set pitch fraction to 'smpl'

    :param progress_bar: Optional, only used to query info about the progress bar
    :param worker: Worker class (unused)
    :param progress_callback:
    :param message_callback: (unused)
    :param range_callback:

    :param dry_run: Simulate process without writing anything, for debugging

    :return: List of resulting files
    """
    audio, sr = sf.read(str(input_file))

    if audio.ndim > 1:
        mono_audio = audio.mean(axis=1)
    else:
        mono_audio = audio

    window_size = max(int(min_duration * sr / 4), 64)
    envelope = envelope_transform(mono_audio, w=window_size, mode='rms')
    region_data = find_regions(envelope, sr, db=split_db, min_duration=min_duration)
    region_data = trim_regions(md=region_data, data=mono_audio, db=split_db)

    regions, cues = region_data.regions, region_data.cues

    p = Path(output_file)
    name, ext = p.stem, p.suffix[1:]
    output_dir = p.parent

    if not output_dir.exists() and not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Get input subtype
    subtype = sf.info(str(input_file)).subtype

    # Cue file
    if write_cue_file:
        cue_file_path = output_dir / f'{name}_cues.wav'
        if not dry_run:
            cmp = ({}, {'compression_level': 1.0})[ext == 'flac']
            sf.write(str(cue_file_path), audio, samplerate=sr, subtype=subtype, **cmp)
            append_markers(cue_file_path, cues.tolist())
        else:
            print(f'Cue file: {cue_file_path}')

    # - Bit Depth -
    subtypes = {'PCM_16': 16, 'PCM_24': 24, 'FLOAT': 32}
    bd_dict = {v: k for k, v in subtypes.items()}

    if bit_depth is None:
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
    if progress_bar:
        # Get current bar progress
        pb_val = progress_bar.value()
        pb_mx = progress_bar.maximum()
        # "Subdivide" progress bar according to found regions
        range_callback.emit(0, pb_mx * count)
        progress_callback.emit(pb_val * count)

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
                filepath = output_dir / f'{name}_pitchfail{i + 1:03d}.{ext}'
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
                    filepath = output_dir / f'{name}_{note_name}{octave}.{ext}'
                else:
                    filepath = output_dir / f'{name}_{note:03d}.{ext}'

        if not use_note or freq is False:
            if i < len(suffix):
                filepath = output_dir / f'{name}_{suffix[i]}.{ext}'
            else:
                filepath = output_dir / f'{name}_{i + 1:03d}.{ext}'

        if extra_suffix:
            filepath = filepath.parent / f'{filepath.stem}{extra_suffix}.{ext}'

        # Prevent overwriting
        incr = 0
        p = Path(filepath)
        while filepath in split_files:
            incr += 1
            filepath = output_dir / f'{p.stem}_{incr:03d}.{ext}'

        if dc_offset:
            dc_offset = np.mean(data)
            data -= dc_offset

        if fade_db is not None:
            _, fade_cues = trim_audio(data, db=fade_db, prevent_empty=False)
            if fade_cues is not None:
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
            cmp = ({}, {'compression_level': 1.0})[ext == 'flac']
            sf.write(str(sf_path), data, samplerate=sr, subtype=bd_dict[bit_depth], **cmp)
            if sf_path != filepath:
                os.rename(sf_path, filepath)
        else:
            print(f'Split {i + 1:03d}: {filepath}')

        # Add 'smpl' metadata
        if use_note:
            if ext == 'wav':
                append_metadata(filepath, note=note, pitch_fraction=pitch_fraction, loop_start=None, loop_end=None)
            # Add metadata as tags for flac
            elif ext == 'flac':
                set_md_tags(filepath, md=md)

        # Increment progress bar sub-task
        if progress_bar is not None:
            progress_callback.emit(pb_val * count + i + 1)

        split_files.append(filepath)

    # Restore progress bar maximum and increment parent task
    if progress_bar is not None:
        range_callback.emit(0, pb_mx)
        progress_callback.emit(pb_val + 1)

    return split_files


def trim_file(input_file: Path | str = '', output_file: Path | str = '', bit_depth: int | None = None,
              trim_db: float = -60, fade_db: float = -48,
              md: dict | None = None, extra_suffix: str | None = '',
              dither: bool = False, dc_offset: bool = True,
              dry_run: bool = True, no_overwriting: bool = False) -> Path:
    """
    Trim audio file while preserving metadata

    :param input_file:
    :param output_file:
    :param bit_depth:
    :param trim_db:
    :param fade_db:
    :param md:
    :param extra_suffix:
    :param dither:
    :param dc_offset:
    :param dry_run:
    :param no_overwriting:

    :return: Result file
    """
    audio, sr = sf.read(str(input_file))

    if audio.ndim > 1:
        mono_audio = audio.mean(axis=1)
    else:
        mono_audio = audio

    # Keep only one channel if audio is not stereo
    if not is_stereo(audio, db=-48):
        data = mono_audio
    else:
        data = audio

    p = Path(output_file)
    name, ext = p.stem, p.suffix[1:]
    output_dir = p.parent

    info = info_from_name(input_file)
    md = md or {}
    if ext == 'flac':
        md = get_md_tags(input_file) | md

    if not output_dir.exists() and not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # - Bit Depth -
    # Get input subtype
    subtype = sf.info(str(input_file)).subtype

    bit_depth = bit_depth or info.params.sampwidth * 8
    if ext == 'flac':
        bit_depth = min(bit_depth, 24)
    subtypes = {16: 'PCM_16', 24: 'PCM_24', 32: 'FLOAT'}
    subtype = subtypes.get(bit_depth, subtype)

    # Integer peak value
    mx = 1.0
    if bit_depth <= 24:
        mx = 2 ** (bit_depth - 1) - 1

    filepath = output_dir / f'{name}.{ext}'

    if extra_suffix:
        filepath = filepath.parent / f'{filepath.stem}{extra_suffix}.{ext}'

    if dc_offset:
        dc_offset = np.mean(data)
        data -= dc_offset

    data, trim_cues = trim_audio(data, db=trim_db, prevent_empty=True)
    length = len(data)

    if dry_run:
        print(f'Trim: {len(mono_audio)} -> {trim_cues}')

    if fade_db is not None:
        _, fade_cues = trim_audio(data, db=fade_db, prevent_empty=False)

        if dry_run:
            print(f'Fade: {len(data)} -> {fade_cues}')

        if fade_cues is not None:
            orig_data = np.array(data)
            data = apply_fade(data, fade_in=(0, fade_cues[0], 'log'),
                              fade_out=(fade_cues[1], len(data) - 1 - fade_cues[1], 'log'))

            if dither is True and bit_depth < 24:
                dither_mask = (np.abs(orig_data - data) > 0).astype(np.float64)
                dither = dither_mask * np.random.triangular(-1, 0, 1, data.shape)
                data = np.round(data * mx + dither).astype(np.int16)

    # Adjust loops and cues with trim
    if hasattr(info, 'loops'):
        if info.loops is not None:
            info.loops = [[min(val - trim_cues[0], length - 1) for val in loop] for loop in info.loops]
            info.loopStart, info.loopEnd = info.loops[0]
            if dry_run:
                print(f'Loops: {info.loops}')
    if hasattr(info, 'cues'):
        info.cues = [min(cue - trim_cues[0], length - 1) for cue in info.cues]
        if dry_run:
            print(f'Cues: {info.cues}')

    if not dry_run:
        output_dir = p.parent

        # Temporary file name
        with tempfile.NamedTemporaryFile(dir=output_dir, suffix=f'.{ext}', delete=False) as temp_file:
            tmp_name = temp_file.name
        cmp = ({}, {'compression_level': 1.0})[ext == 'flac']
        sf_path = (tmp_name, f'{filepath}f')[ext == 'aif']
        sf.write(str(sf_path), data, samplerate=sr, subtype=subtype, **cmp)
        # if sf_path != filepath:
        #     os.rename(sf_path, filepath)

        # Write Metadata
        if ext == 'wav':
            append_metadata(tmp_name, note=info.note, pitch_fraction=info.pitchFraction,
                            loop_start=info.loopStart, loop_end=info.loopEnd)
            if hasattr(info, 'cues'):
                append_markers(tmp_name, markers=info.cues)
        elif ext == 'flac':
            attrs = ['loopStart', 'loopEnd', 'loops', 'cues']
            values = [getattr(info, attr) for attr in attrs]
            md |= dict(zip(attrs, values))
            set_md_tags(str(tmp_name), md=md)

        if no_overwriting:
            resolve_overwriting(output_file, mode='dir', dir_name='backup_', test_run=False)
        elif output_file.is_file():
            os.remove(output_file)

        os.rename(tmp_name, output_file)

    return filepath


# Auxiliary defs


def trim_audio(data: np.ndarray, db: float = -60, prevent_empty: bool = True) -> namedtuple:
    """
    Remove trailing silence
    :param data: Input audio
    :param db: Silence threshold in dB
    :param prevent_empty: Return original audio if trimming fails otherwise returns (None,None)
    :return: Return processed audio and new cues as a named tuple (data,cues)
    """
    if data.ndim > 1:
        mono_audio = data.mean(axis=-1)
    else:
        mono_audio = data
    data_len = len(mono_audio)
    # mono_audio = np.pad(mono_audio, pad_width=(1, 1), mode='constant', constant_values=.0)

    th = np.power(10, db / 20)
    silence = np.abs(mono_audio) < th
    # idx = np.clip(np.where(silence == 0)[0] - 1, 0, data_len - 1)
    idx = np.where(silence == 0)[0]

    trim = namedtuple('trim', 'data cues')

    if len(idx) > 0:
        result = trim(data[idx[0]:idx[-1] + 1], (idx[0], idx[-1]))
    else:
        if prevent_empty:
            result = trim(data, (0, data_len - 1))
        else:
            result = trim(None, None)

    return result


def find_regions(envelope: np.ndarray, sr: int, db: float = -60, min_duration: float = .1) -> namedtuple:
    """
    Detect regions from a volume envelope

    :param envelope: Input envelope

    :param sr: Sample rate for duration
    :param db: Silence threshold in dB
    :param min_duration: in s
    :return: Return regions and cues as a named tuple (regions,cues)
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


def trim_regions(md: namedtuple, data: np.ndarray, db: float = -60) -> namedtuple:
    """
    Refine regions by trimming from audio data
    :param md: regions,cues metadata
    :param data: Audio data, mono or stereo
    :param db: Silence threshold
    :return: Processed metadata as a named tuple (regions,cues)
    """
    cues = []
    for region in md.regions:
        trim = trim_audio(data[region[0]:region[1] - 1], db=db, prevent_empty=True)
        st, ed = trim.cues
        cues.extend([region[0] + st, region[0] + ed])
    cues = np.array(cues)
    regions = cues.reshape(-1, 2)

    metadata = namedtuple('metadata', 'regions cues')
    result = metadata(regions, cues)

    return result


def envelope_transform(data: np.ndarray, w: int = 1024, mode: str = 'max', interp: str = 'linear') -> np.ndarray:
    """
    Transform input audio to volume envelope
    :param data: Input single channel audio
    :param w: Rolling window size
    :param mode: 'max' for peak or 'rms' for root-mean-square
    :param interp: Interpolation type 'linear' or 'cubic'
    'linear' is better for volume estimation (no overshoots)
    'cubic' is better for shaping (introduces less distortion)
    :return: Resulting envelope
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

def apply_fade(data: np.ndarray, fade_in: tuple[int, int, str] = (0, 100, 'log'),
               fade_out: tuple[int, int, str] = (500, 32000, 'exp')) -> np.ndarray:
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
