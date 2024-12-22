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

import mutagen
import numpy as np
import soundfile as sf


# Set/Append Tags / Metadata functions
def append_metadata(input_file, note, pitch_fraction, loop_start, loop_end):
    """
    Simplistic note and region loop appending, properly recognized by Kontakt and other software

    Based on information found here :
    https://www.recordingblogs.com/wiki/sample-chunk-of-a-wave-file
    and in wave_mod.py (enhanced version of the wave module)

    The data is simply appended, so it's meant to be used once on a wav without any note/loop metadata

    :param float pitch_fraction:
    :param int note:
    :param str input_file:
    :param int or None loop_start:
    :param int or None loop_end:
    :return:
    """

    # Get sample rate and RIFF size
    info = sf.info(input_file)
    sr = info.samplerate
    riff_size = os.path.getsize(input_file) - 8  # File size minus header header chunks
    riff_size += 68  # Metadata Size

    # Update RIFF size to make the file valid after appending metadata
    with open(input_file, 'r+b') as wf:
        wf.seek(4)
        wf.write(as_chunk(riff_size, 4))
    wf.close()

    bin_data = metadata_to_bin(sr, note, pitch_fraction, loop_start, loop_end)

    with open(input_file, 'ab') as f:
        f.write(bin_data)
    f.close()


def metadata_to_bin(sr, note, pitch_fraction, loop_start, loop_end):
    """
    Note / Loop metadata to riff bin chunk data
    :param int sr: Sampling Rate, necessary to set 'sample period' info correctly
    :param int or None note: MIDI note number in semitones, integer between 0 and 127
    For example, 60 is the default note (C4) 69 is A4
    :param float or None pitch_fraction: In semitone cents
    :param int or None loop_start: Use None to disable looping, only one forever loop is supported
    :param int or None loop_end:
    :return:
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
    bin_data += as_chunk(0, 4)  # "sampler data" ?

    bin_data += as_chunk(loops, 4)  # "cuepointid" (set to 01 to make the loop valid)
    bin_data += as_chunk(0, 4)  # "cuetype" ? (optional)
    bin_data += as_chunk(st, 4)  # loop start
    bin_data += as_chunk(ed, 4)  # loop end
    bin_data += as_chunk(0, 8)  # "fraction" ?, "playcount" (0 is seemingly infinite loop)

    # 28 bytes - Optional
    bin_data += b'LIST' + as_chunk(20, 4)
    bin_data += b'adtllabl' + as_chunk(8, 4)
    bin_data += as_chunk(1, 8)  # ? (copied from a valid file)

    return bin_data


def set_metadata_tags(input_file, note, pitch_fraction, loop_start, loop_end):
    """
    Set sample metadata as tags using mutagen
    Currently only supports FLAC format
    :param str input_file:
    :param int note:
    :param float pitch_fraction:
    :param int loop_start:
    :param int loop_end:
    :return:
    """
    audio = mutagen.File(input_file)
    tags = ['note', 'pitchFraction', 'loopStart', 'loopEnd']
    values = [note, pitch_fraction, loop_start, loop_end]
    for tag, value in zip(tags, values):
        if value is not None:
            audio[tag] = str(value)
    audio.save()


def set_md_tags(input_file, md=None):
    """
    Set ID3 tags from dict
    Currently only supports FLAC format
    :param str input_file:
    :param dict md:
    :return:
    """
    if not md:
        return None
    audio = mutagen.File(input_file)
    for tag, value in md.items():
        if value is not None:
            audio[tag] = str(value)
    audio.save()


# String Manipulation
def rep_word_from_name(name='', word='', repstr='', idx=-1, min_count=1, ignore_case=False):
    """
    Replace word from name

    :param bool ignore_case:
    :param str name:
    :param str word:
    :param str repstr:
    :param int idx:
    :param int min_count:

    :return:
    :rtype: str
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
def hz_to_note(freq, ref_freq=440, ref_note=69):
    """
    Return MIDI note number from frequency in Hz
    :param float or int freq:
    :param float ref_freq:
    :param int ref_note:
    :return:
    :rtype: float
    """
    return ref_note + 12 * math.log2(freq / ref_freq)


def note_to_hz(note, ref_freq=440, ref_note=69):
    """
    Return frequency from MIDI note
    :param int or float note: MIDI note number
    :param float ref_freq:
    :param int ref_note:
    :return: frequency
    :rtype: float
    """
    return ref_freq * 2 ** ((note - ref_note) / 12)


def hz_to_period(f, sr=48000):
    return int(round(sr / f))


def note_to_name(note):
    """
    Return note name and octave number from MIDI note
    :param int note: MIDI note number
    :return: Note name, Octave number
    :rtype: list
    """
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_name = note_names[note % 12]
    octave = int((note // 12) - 1)
    return note_name, octave


def rm_digit_words(name, sep='-_ '):
    """
    Conform name by removing digits only words in a name

    :param str name: Name string to process
    :param str sep: Separators

    :return: Processed string
    :rtype: str
    """
    pattern = f'[{sep}]'
    splits = re.split(pattern, name)
    splits = [s for s in splits if not s.isdigit()]
    return '_'.join(splits)


def name_to_note(name):
    """
    Return MIDI note number from note name following this pattern "C#3"
    :param str name:
    :return:
    :rtype: int
    """
    note = dict(zip('CDEFGAB', [0, 2, 4, 5, 7, 9, 11]))[name.upper()[0]]
    if '#' in name:
        note += 1
    octave = int(name[-1])
    note = note + (octave + 1) * 12
    return note


def is_note_name(name):
    """
    Identify if a string is likely to be a note name
    :param str name:
    :return:
    :rtyoe: bool
    """
    if 2 <= len(name) <= 3:
        if name[0].upper() in 'CDEFGAB':
            if name[-1].isdigit():
                return True
    return False


def find_notes_from_name(name):
    """
    Extract note from string
    :param str name:
    :return:
    :rtype: str
    """
    pattern = r"[A-G][#bN]?\d"
    note_names = re.findall(pattern, name, re.IGNORECASE)
    return note_names


def pitch_fraction_to_bin(value):
    """
    Convert pitch fraction (semitone cents) to a binary chunk (4 bytes integer in Intel "little endian" format)
    :param int or float value: Pitch fraction in semitone cents, can only be positive
    :return: Pitch fraction as unsigned 32 bits value little-endian
    :rtyoe: binary
    """
    return int(value * 0xFFFFFFFF / 100).to_bytes(4, byteorder='little')


# Velocity/Dynamic Conversion


def dyn_table(names=('ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff'), mn=15, mx=127):
    """
    Conversion table from dynamic name to velocity
    :param str names: Dynamics names in increasing order
    :param int mn: Min velocity
    :param int mx: Max velocity
    :return:
    :rtype: dict
    """
    # Generate values using a simple linear interpolation between min and max
    values = np.linspace(mn, mx, len(names), endpoint=True, dtype=np.dtype(np.int16))
    return dict(zip(names, values.tolist()))


def is_dyn_name(name):
    """
    Identify if a string is likely to be a dynamic name
    :param str name:
    :return:
    :rtyoe: bool
    """
    if name.lower() in dyn_table():
        return True
    return False


def dyn_to_vel(name):
    """
    Convert a dynamic name to its corresponding velocity value
    :param str name:
    :return:
    """
    dyn_vel = dyn_table()
    key = name.lower()
    if key in dyn_vel:
        return dyn_vel[key]
    else:
        return 127


def vel_to_dyn(value):
    """
    Convert a velocity value to the closest dynamic name
    :param int or None value: Velocity value (0-127)
    :return: Dynamic name
    :rtype: str
    """
    if value is None:
        return 'fff'
    vel_dyn = {v: k for k, v in dyn_table().items()}  # Reverse table
    values = np.array(list(vel_dyn.keys()))
    idx = np.argmin(np.abs(values - value))
    key = values[idx]
    return vel_dyn[key]


# Utility

def as_chunk(value, length):
    """
    Encode a given value as a "little endian" chunk
    :param int value:
    :param int length: Chunk length in bytes
    :return:
    :rtype: binary
    """
    return int(value).to_bytes(length, byteorder='little')

# print(round(44100 * note_to_hz(68.5) / 440))
