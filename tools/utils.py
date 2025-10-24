# coding:utf-8
"""
    :module: utils.py
    :description: Various conversion note frequency and metadata functions
    :author: Michel 'Mitch' Pecqueur
    :date: 2024.05
"""

import math
import os
import re
from pathlib import Path

import mutagen
import numpy as np
import soundfile as sf


# Set/Append Tags / Metadata functions
def append_metadata(input_file: Path | str, note: int, pitch_fraction: float | None,
                    loop_start: int | None, loop_end: int | None):
    """
    Simplistic note and region loop appending, properly recognized by Kontakt and other software

    Based on information found here :

    official WAV specification by Microsoft
    https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/Docs/RIFFNEW.pdf

    https://www.recordingblogs.com/wiki/sample-chunk-of-a-wave-file
    and in wave_mod.py (enhanced version of the wave module)

    The data is simply appended, so it's meant to be used once on a wav without any note/loop metadata

    :param pitch_fraction:
    :param note:
    :param input_file:
    :param loop_start:
    :param loop_end:
    """

    # Get sample rate and RIFF size
    info = sf.info(str(input_file))
    sr = info.samplerate
    riff_size = os.path.getsize(input_file) - 8  # File size minus header header chunks
    riff_size += 68  # Metadata Size

    # Update RIFF size to make the file valid after appending metadata
    with open(str(input_file), 'r+b') as wf:
        wf.seek(4)
        wf.write(as_chunk(riff_size, 4))
    wf.close()

    bin_data = metadata_to_bin(sr, note, pitch_fraction, loop_start, loop_end)

    with open(str(input_file), 'ab') as f:
        f.write(bin_data)
    f.close()


def metadata_to_bin(sr: int, note: int | None, pitch_fraction: float | None,
                    loop_start: int | None, loop_end: int | None) -> bin:
    """
    Minimal Note / Loop metadata to riff bin chunk data

    :param sr: Sampling Rate, necessary to set 'sample period' info correctly
    :param note: MIDI note number in semitones, integer between 0 and 127
    For example, 60 is the default note (C4) 69 is A4
    :param pitch_fraction: In semitone cents
    :param loop_start: Use None to disable looping, only one forever loop is supported
    :param loop_end:

    :return: Binary data
    """
    if note is None:
        note = 60
    if pitch_fraction is None:
        pitch_fraction = 0
    if loop_start is None or loop_end is None:
        st, ed, loops = 0, 0, 0
    else:
        st, ed, loops = loop_start, loop_end, 1

    # 68 bytes
    bin_data = b'smpl' + as_chunk(60, 4)  # chunk name + size

    bin_data += as_chunk(0, 4)  # manufacturer
    bin_data += as_chunk(0, 4)  # product
    bin_data += as_chunk(1000000000 // sr, 4)  # sample period

    # Support for negative pitch fraction
    if pitch_fraction < 0:
        note -= 1
        pitch_fraction += 100

    bin_data += as_chunk(note, 4)  # MIDI note
    bin_data += pitch_fraction_to_bin(pitch_fraction)  # pitch fraction

    bin_data += as_chunk(0, 4)  # smpte format
    bin_data += as_chunk(0, 4)  # smpte offset

    bin_data += as_chunk(loops, 4)  # number of loops
    bin_data += as_chunk(0, 4)  # "sampler data" optional specific data (unused)

    bin_data += as_chunk(loops, 4)  # "cuepointid" (set to 1 to make the loop valid)
    bin_data += as_chunk(0, 4)  # Loop type: 0 is forward (1 ping-pong, 2 backward)
    bin_data += as_chunk(st, 4)  # loop start
    bin_data += as_chunk(ed, 4)  # loop end

    bin_data += as_chunk(0, 4)  # fraction: loop fractional areas between samples (unused)
    bin_data += as_chunk(0, 4)  # play count: 0 is infinite

    # 28 bytes - Optional label (copied from a valid file)
    bin_data += b'LIST' + as_chunk(20, 4)  # chunk name + size
    bin_data += b'adtl'  # Associated Data List
    bin_data += b'labl' + as_chunk(12, 4)  # label + size of following data
    bin_data += as_chunk(1, 4)  # id
    bin_data += b'Loop01' + as_chunk(0, 2)  # Loop label

    return bin_data


def get_md_tags(input_file: Path | str) -> dict:
    """
    Get all tags as dict using mutagen

    :param input_file:
    """
    audio = mutagen.File(str(input_file))
    return {k: v[0] for k, v in audio.tags.items()}


def set_md_tags(input_file: Path | str, md: dict | None = None):
    """
    Set tags from dict using mutagen
    Currently only supports FLAC format

    :param input_file:
    :param md: Metadata dictionary
    """
    if not md:
        return None
    audio = mutagen.File(str(input_file))
    for tag, value in md.items():
        if value is not None:
            audio[tag] = str(value)
    audio.save()


# String Manipulation
def rep_word_from_name(name: str = '', word: str = '', repstr: str = '', idx: int = -1, min_count: int = 1,
                       ignore_case: bool = False) -> str:
    """
    Replace word from name

    :param name:
    :param word:
    :param repstr:
    :param idx:
    :param min_count:
    :param ignore_case:

    :return:
    """
    words = re.findall(word, name)
    kwargs = {}
    if ignore_case:
        kwargs = {'flags': re.IGNORECASE}
    if len(words) > min_count:
        splits = re.split(word, name, **kwargs)
        words[idx] = repstr
        result = ''.join([a for b in zip(splits, words + ['']) for a in b])
    else:
        result = name
    return result


# Note/Pitch Conversion
def hz_to_note(freq: float | int, ref_freq: float = 440, ref_note: int = 69) -> float:
    """
    Return MIDI note number from frequency in Hz
    :param freq: Given frequency (Hz)
    :param ref_freq: Reference frequency (Hz)
    :param ref_note: Reference note number
    :return: MIDI note number
    """
    return ref_note + 12 * math.log2(freq / ref_freq)


def note_to_hz(note: int | float, ref_freq: float = 440, ref_note: int = 69) -> float:
    """
    Return frequency from MIDI note
    :param note: MIDI note number
    :param ref_freq: Reference frequency (Hz)
    :param ref_note: Reference note number
    :return: frequency (Hz)
    """
    return ref_freq * 2 ** ((note - ref_note) / 12)


def note_to_name(note: int) -> tuple[str, int]:
    """
    Return note name and octave number from MIDI note
    :param note: MIDI note number
    :return: Note name, Octave number
    """
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_name = note_names[note % 12]
    octave = int((note // 12) - 1)
    return note_name, octave


def name_to_note(name: str) -> int:
    """
    Return MIDI note number from note name following this pattern "C#3"
    :param name: Note name
    :return: MIDI note number
    """
    note = dict(zip('CDEFGAB', [0, 2, 4, 5, 7, 9, 11]))[name.upper()[0]]
    if '#' in name:
        note += 1
    octave = int((re.findall('(-?\\d+)$', name) or [4])[0])
    note = note + (octave + 1) * 12
    return note


def is_note_name(name: str) -> bool:
    """
    Identify if a string is likely to be a note name
    :param name:
    :return:
    """
    if 2 <= len(name) <= 3:
        if name[0].upper() in 'CDEFGAB':
            if name[-1].isdigit():
                return True
    return False


def hz_to_period(f: float, sr: int = 48000) -> int:
    return int(round(sr / f))


def rm_digit_words(name: str, sep: str = '-_ ') -> str:
    """
    Conform name by removing digits only words in a name

    :param name: Name string to process
    :param sep: Separators

    :return: Processed string
    """
    pattern = f'[{sep}]'
    splits = re.split(pattern, name)
    splits = [s for s in splits if not s.isdigit()]
    return '_'.join(splits)


def find_notes_from_name(name: str) -> list:
    """
    Extract note from string
    :param name:
    :return:
    """
    pattern = r"[A-G][#bN]?\d"
    note_names = re.findall(pattern, name, re.IGNORECASE)
    return note_names


def pitch_fraction_to_bin(value: int | float) -> bin:
    """
    Convert pitch fraction (semitone cents) to a binary chunk (4 bytes integer in Intel "little endian" format)
    :param value: Pitch fraction in semitone cents, can only be positive
    :return: Pitch fraction as unsigned 32 bits value little-endian
    """
    return int(value * 0xFFFFFFFF / 100).to_bytes(4, byteorder='little')


# Velocity/Dynamic Conversion


def dyn_table(names: tuple[str] = ('ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff'),
              mn: int = 15, mx: int = 127) -> dict:
    """
    Conversion table from dynamic name to velocity
    :param names: Dynamics names in increasing order
    :param mn: Min velocity
    :param mx: Max velocity (typically 127)
    :return: Table as a dict
    """
    # Generate values using a simple linear interpolation between min and max
    values = np.linspace(mn, mx, len(names), endpoint=True, dtype=np.dtype(np.int16))
    return dict(zip(names, values.tolist()))


def is_dyn_name(name: str) -> bool:
    """
    Identify if a string is likely to be a dynamic name
    :param name:
    :return:
    """
    if name.lower() in dyn_table():
        return True
    return False


def dyn_to_vel(name: str) -> int:
    """
    Convert a dynamic name to its corresponding velocity value
    :param name: Dynamic name
    :return: Velocity value (0-127)
    """
    dyn_vel = dyn_table()
    key = name.lower()
    if key in dyn_vel:
        return dyn_vel[key]
    else:
        return 127


def vel_to_dyn(value: int | None) -> str:
    """
    Convert a velocity value to the closest dynamic name
    :param value: Velocity value (0-127)
    :return: Dynamic name
    """
    if value is None:
        return 'fff'
    vel_dyn = {v: k for k, v in dyn_table().items()}  # Reverse table
    values = np.array(list(vel_dyn.keys()))
    idx = np.argmin(np.abs(values - value))
    key = values[idx]
    return vel_dyn[key]


# Utility

def as_chunk(value: int, length: int) -> bin:
    """
    Encode a given value as a "little endian" chunk
    :param value: Integer value
    :param length: Chunk length in bytes
    :return: Binary chunk
    """
    return int(value).to_bytes(length, byteorder='little')
