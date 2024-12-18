# coding:utf-8
"""
    :module: sf2_to_wav.py
    :description: Extract samples from a Soundfont 2 file
    :author: Michel 'Mitch' Pecqueur
    :Original version: Andrew Ostler, Copyright (c) 2020 Expert Sleepers Ltd. MIT License.
    :date: 2024.04
"""

import json
import os
import struct
import sys
import traceback
import wave
from collections import OrderedDict

import numpy as np
from chunkmuncher.chunk import Chunk

from utils import append_metadata, note_to_name, rep_word_from_name, rm_digit_words, name_to_note, is_note_name, \
    find_notes_from_name


def sf2_to_wav(sf2file, output_dir=None, rename=False,
               transpose=0, force_pitch_from_name=True, default_sample_name='sample'):
    """
    Extract samples from a Soundfont 2 file
    Samples are written as wav files with proper metadata recognized by Kontakt

    :param str sf2file: Input sf2 file path
    :param str or None output_dir: Output path, if None creates a folder at the same location as the source file
    :param bool rename: Attempt to auto-rename samples

    :param bool force_pitch_from_name: Use sample name to set pitch instead of pitch information
    :param int transpose: Semitone offset applied to notes read from sample names
    :param str default_sample_name: Give a default name

    :return:
    """
    samples = []

    sample_types = {1: 'mono', 2: 'right', 4: 'left', 8: 'linked'}

    with open(sf2file, 'rb') as f:
        chfile = Chunk(f)
        riff = chfile.getname()
        WAVE = chfile.read(4)

        while True:
            try:
                chunk = Chunk(chfile, bigendian=0)
            except EOFError:
                break

            name = chunk.getname()

            if name == b'LIST':
                listname = chfile.read(4)
                print('\t', listname)
            elif name == b'smpl':
                sample_data_start = chfile.tell() + 8
                print('Sample data starts at', sample_data_start)
                chunk.skip()
            elif name == b'shdr':
                for i in range((chunk.chunksize // 46) - 1):
                    s = SfSample(chfile)
                    samples.append(s)
                chfile.read(46)
            else:
                chunk.skip()

    for s in samples:
        type_name = sample_types[s.type & 0x7fff]
        print('{} {} {} {} {} {} {} {} {} {}'.format(filter_text(s.name), s.start, s.end, s.startLoop, s.endLoop,
                                                     s.sampleRate, s.pitch, s.correction, s.link, type_name))

    sf2_in1 = open(sf2file, 'rb')
    sf2_in2 = open(sf2file, 'rb')

    base_name = os.path.basename(sf2file).split('.')[0]
    base_name = "".join(x for x in base_name if x.isalnum() or x in ' -_()#')
    samples_dir = os.path.join(base_name, 'Samples')
    print(base_name)

    if not output_dir:
        output_dir = os.path.dirname(sf2file)

    dest_dir = os.path.join(output_dir, samples_dir)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    os.chdir(dest_dir)

    log_dict = OrderedDict()

    for s in samples:
        if s.type not in [1, 4]:
            continue

        sample_name = filter_text(s.name)
        sample_name = sample_name.replace('(L)', '')  # Remove channel from name
        wav_name = sample_name

        pitch = s.pitch
        correction = s.correction

        if rename:
            # Remove note from sample_name
            found = find_notes_from_name(wav_name)
            note_name = None
            pitch_from_name = None
            if found:
                note_name = found[-1]
                if is_note_name(note_name):
                    pitch_from_name = name_to_note(note_name)
                    wav_name = rep_word_from_name(name=wav_name, word=note_name, idx=-1, min_count=0,
                                                  ignore_case=True).rstrip('-_')

            # Remove digits
            wav_name = rm_digit_words(wav_name)

            # Give some default name if the string is empty
            if wav_name == '':
                wav_name = default_sample_name

            if force_pitch_from_name:
                pitch = pitch_from_name or 0 + transpose
            else:
                # Only transpose note number with default pitch of 60
                if s.pitch == 60 and note_name:
                    b, _ = note_to_name(name_to_note(note_name))
                    if b != 'C':
                        pitch += pitch_from_name + transpose

            filename = f"{wav_name}_{pitch}.wav"

        else:
            filename = f"{wav_name}.wav"

        # Avoid overwriting samples
        i = 1
        while os.path.isfile(filename):
            filename = f'{wav_name}_{i:02d}.wav'
            i += 1

        log_dict[filename] = {'sample_name': sample_name, 'pitch': s.pitch}
        sys.stdout.write(f'    {filename}')

        wav_out = wave.open(filename, mode='w')

        wav_out.setsampwidth(2)
        wav_out.setframerate(s.sampleRate)
        sf2_in1.seek(sample_data_start + 2 * s.start)
        frames = s.end - s.start + 1

        if s.type == 1:  # Mono
            wav_out.setnchannels(1)
            data = sf2_in1.read(frames * 2)
            wav_out.writeframesraw(data)
        else:  # Stereo
            wav_out.setnchannels(2)
            # Read both channels
            sf2_in2.seek(sample_data_start + 2 * samples[s.link].start)
            data1 = sf2_in1.read(frames * 2)
            data2 = sf2_in2.read(frames * 2)
            # Interleave left and right channels using numpy
            np_array = np.empty((frames * 2,), dtype=np.int16)
            np_array[0::2] = np.frombuffer(data1, dtype=np.int16)
            np_array[1::2] = np.frombuffer(data2, dtype=np.int16)
            # Write data
            wav_out.writeframesraw(np_array.tobytes())
        wav_out.close()

        loop_length = s.endLoop - s.startLoop

        if loop_length > 1:
            append_metadata(filename, note=pitch, pitch_fraction=correction, loop_start=s.startLoop - s.start,
                            loop_end=s.endLoop - s.start - 1)
        else:
            sys.stdout.write('  loop off')
            append_metadata(filename, note=pitch, pitch_fraction=correction, loop_start=None, loop_end=None)

        # Write log file
        log_file = os.path.join(output_dir, '{}/{}_log.json'.format(base_name, base_name.replace(' ', '_')))
        write_json(data=log_dict, filepath=log_file, sort_keys=True, indent=4)

        sys.stdout.write('\n')


# Auxiliary definitions
class SfSample:
    """
    Class providing info about a Soundfont sample
    """

    def __init__(self, f):
        self.name = f.read(20)
        self.start = _read_dword(f)
        self.end = _read_dword(f)
        self.startLoop = _read_dword(f)
        self.endLoop = _read_dword(f)
        self.sampleRate = _read_dword(f)
        self.pitch = _read_byte(f)
        self.correction = _read_byte(f)
        self.link = _read_word(f)
        self.type = _read_word(f)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'sfSample(name="{self.name}",start={self.start})'


def filter_text(data):
    """
    Keep only meaningful characters (printable text) from binary data
    Stops at first outlying byte encountered
    :param data:
    :return:
    :rtype: str
    """
    result = ''
    for byte in data:
        if 32 <= byte <= 126:
            result += chr(byte)
        else:
            break
    return result


def write_json(data, filepath, separators=(',', ':'), sort_keys=True, indent=None):
    """
    Write given data to disk as a json file
    Create necessary folder structure if non-existent
    :param int or None indent: Indent value
    :param bool sort_keys: Ordered keys
    :param list separators:
    :param any data: Python data structure
    :param str filepath: File path
    :return: If no file path supplied return json string
    :rtype: bool or str
    """
    if os.path.isdir(filepath):
        print('Cannot overwrite an existing folder.')
        return False
    filedir = os.path.dirname(filepath)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    json_data = json.dumps(data, separators=tuple(separators), indent=indent, sort_keys=sort_keys)
    if not filepath:
        return data
    else:
        with open(filepath, 'w') as f:
            f.write(json_data)
        return True


# Binary functions from sf2_to_dex

def _read_dword(f):
    return struct.unpack('<i', f.read(4))[0]


def _read_word(f):
    return struct.unpack('<h', f.read(2))[0]


def _read_byte(f):
    return struct.unpack('<b', f.read(1))[0]


def _write_dword(f, v):
    f.write(struct.pack('<i', v))


def _write_word(f, v):
    f.write(struct.pack('<h', v))


# Main

def main():
    if len(sys.argv) == 2:
        try:
            sf2_to_wav(sys.argv[1], output_dir=None)
        except Exception as e:
            traceback.print_exc()
            input('Press ENTER to continue...')
    else:
        sys.stderr.write("No file supplied\n")
        input('Press ENTER to continue...')


if __name__ == "__main__":
    main()
