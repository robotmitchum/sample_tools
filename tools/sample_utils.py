# coding:utf-8
"""
    :module: sample_utils.py
    :description: Base functions to get, set or edit sample information.
    These data are used by sampler programs to define instruments.
    They are typically stored in the 'smpl' chunk of wav riff files.
    :author: Michel 'Mitch' Pecqueur
    :date: 2024.05
"""

import os
import re
import struct
import tempfile
from collections import namedtuple
from copy import deepcopy
from pathlib import Path

import mutagen
import numpy as np
import soundfile as sf
from soxr import resample

from common_math_utils import clamp
from file_utils import resolve_overwriting
from parseAttrString import parse_string, compose_string
from pitch_detect import fine_tune, fine_tune_lr, pitch_detect
from utils import (append_metadata, set_md_tags,
                   hz_to_note, note_to_name, name_to_note, is_note_name, find_notes_from_name,
                   is_dyn_name, dyn_to_vel, vel_to_dyn,
                   rm_digit_words)


class Sample:
    """
    Class to hold sample attributes
    """

    def __init__(self, path='', extra_tags=None):
        """
        :param str path:
        :param list extra_tags:
        """
        if not path:
            return

        p = Path(path)
        info = sf.info(path)

        self.path = path
        self.name, self.ext = str(p.stem), str(p.suffix)
        self.filetype = self.ext[1:]

        bd_dict = {'PCM_16': 16, 'PCM_24': 24, 'PCM_32': 32, 'FLOAT': 32, 'DOUBLE': 64}
        subtype = info.subtype

        if subtype in bd_dict:
            bitdepth = bd_dict[subtype]
        else:
            bitdepth = 16

        nchannels = info.channels
        sampwidth = bitdepth // 8
        framerate = info.samplerate
        nframes = info.frames

        params = namedtuple('_wave_params', ('nchannels', 'sampwidth', 'framerate', 'nframes'))
        self.params = params(nchannels, sampwidth, framerate, nframes)

        # 'smpl' chunk attributes
        self.note, self.pitchFraction, self.loopStart, self.loopEnd, self.loops = None, None, None, None, None
        self.cues = None

        # Read ID3 tags
        md = read_metadata_tags(path, extra_tags)
        # Convert md dict as attributes
        for tag in md:
            setattr(self, tag, md[tag])

        # Wav fourCC tags always override ID3 tags
        if self.filetype == 'wav':
            md.update(read_metadata(path))
            # Convert md dict as attributes
            for tag in md:
                setattr(self, tag, md[tag])
            self.cues = get_cues(path)

            # Adjust note and pitch fraction when pitch fraction is superior to 50 to make values more meaningful
            # Only do it for wav which has no way of storing negative pitch fraction
            if self.pitchFraction is not None:
                if self.pitchFraction > 50:
                    self.pitchFraction -= 100
                    self.note += 1

        self.noteName = None
        self.set_note(self.note)

        self.group = None
        self.seqPosition = None
        self.trigger = None
        self.vel = None
        self.dyn = None

        # Sanitize loops and cues
        if not self.check_loops():
            self.sanitize_loops()
            self.sanitize_cues()

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'Sample({vars(self)})'

    def __copy__(self):
        result = type(self)()
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo=None):
        result = type(self)()
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    # Preferred methods to set note and vel
    # as they maintain relationship between integer value and string representation

    def set_note(self, value):
        if value is None or not isinstance(value, int):
            return
        self.note = int(clamp(value, 0, 127))
        note, octave = note_to_name(self.note)
        self.noteName = f'{note}{octave}'

    def set_notename(self, value):
        if value is None:
            return
        if is_note_name(value):
            self.note = name_to_note(value)
            self.noteName = value.upper()
        else:
            self.note, self.noteName, self.pitchFraction = None, None, None

    def transpose(self, value):
        if any([value is None, value == 0, self.note is None]):
            return
        self.set_note(self.note + value)

    def set_vel(self, value):
        if value is None:
            return
        if value <= 0:
            self.vel, self.dyn = None, None
        else:
            self.vel = min(value, 127)
            self.dyn = vel_to_dyn(self.vel)

    def set_dyn(self, value):
        if value is None:
            return
        if is_dyn_name(value):
            self.vel = dyn_to_vel(value)
            self.dyn = value.lower()
        else:
            self.vel, self.dyn = None, None

    def check_loops(self):
        """
        Get info about invalid loops
        """
        if self.loops:
            mx = self.params.nframes - 1
            for i, loop in enumerate(self.loops):
                for c, cue in enumerate(loop):
                    if cue < 0 or cue > self.params.nframes - 1:
                        print(f'{self.name}{self.ext} loop {i} invalid {('start', 'end')[c]} : {cue} outside 0-{mx}')
                        return False
        return True

    def sanitize_loops(self):
        """
        Fix invalid loop information written by some programs
        typically loop end off by 1 as loop end is inclusive
        https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/Docs/RIFFNEW.pdf
        Page 18 ('smpl' chunk section)
        dwEnd Specifies the endpoint of the loop in samples (this sample will also be played)
        """
        if self.loops:
            self.loops = [[clamp(cue, 0, self.params.nframes - 1) for cue in loop] for loop in self.loops]
            self.loopStart, self.loopEnd = self.loops[0]

    def sanitize_cues(self):
        """
        Fix invalid cue information written by some programs
        """
        if self.cues:
            self.cues = [clamp(cue, 0, self.params.nframes - 1) for cue in self.cues]


def info_from_name(path, pattern='{group}_{note}', override=True, force_pitch_from_name=False,
                   extra_tags=None, num_attrib=('vel', 'note', 'seqPosition')):
    """
    Extract info from name

    :param str path: input file path
    :param str or None pattern: Pattern used to extract info, use {} to mark attributes
    :param bool force_pitch_from_name: Force note from the whole sample name
    Might work even when the name is badly formatted
    :param bool override: Overwrite existing attributes
    :param list or tuple or None extra_tags:
    :param list or tuple num_attrib: List of attributes to be considered numeric so letters should be ignored

    :return: Return Info as a Sample object
    :rtype: Sample
    """
    smp = Sample(path, extra_tags=extra_tags)

    if not pattern and not force_pitch_from_name:
        return smp

    info_dict = parse_string(smp.name, pattern=pattern)

    for attr, value in info_dict.items():
        write_key = False

        if hasattr(smp, attr):
            if getattr(smp, attr) is None or (override and value is not None):
                write_key = True
        else:
            write_key = True

        if write_key:
            if value is not None:
                num_value = None
                if isinstance(value, str):
                    # Remove alpha characters from numeric attributes
                    if attr in num_attrib:
                        try:
                            num_value = eval(rm_zero_pad(rm_alpha_chr(value)))
                        except Exception as e:
                            num_value = None
                            pass

                if attr == 'note':
                    if is_note_name(value):
                        smp.set_notename(value)
                    else:
                        smp.set_note(num_value)
                elif attr == 'vel':
                    if is_dyn_name(value):
                        smp.set_dyn(value)
                    else:
                        smp.set_vel(num_value)
                elif attr == 'notename':
                    smp.set_notename(value)
                elif attr == 'dyn':
                    smp.set_dyn(value)
                else:
                    setattr(smp, attr, (value, num_value)[attr in num_attrib])

    # Force note from the whole sample name
    if force_pitch_from_name or smp.note is None:
        found = find_notes_from_name(smp.name)
        if found:
            note_name = found[-1]  # If more than one result, favor the last one
            if is_note_name(note_name):
                smp.set_notename(note_name)

    # Sanitize loops and cues
    if not smp.check_loops():
        smp.sanitize_loops()
        smp.sanitize_cues()

    return smp


def rename_sample(input_file, output_dir='', output_ext='wav', check_list=(), bit_depth=None,
                  group_name='sample', rep_str=None,
                  src_pattern='', tgt_pattern='{group}_{note}_{trigger}',
                  prefix='', suffix='',
                  extra_tags=None,
                  transpose=0, force_pitch_from_name=True,
                  detect_pitch=None, pitch_fraction_mode='keep', pitch_fraction_override=None,
                  use_loop=True, bake_pf=None,
                  test_run=False):
    """
    Rename sample while updating its metadata accordingly

    :param bit_depth:
    :param str input_file:
    :param str output_dir: Optional output writing directory
    :param str output_ext: Output format, wav or flac, aiff NOT supported
    :param list check_list: Optional file list used to resolve overwriting within the renaming process
    :param int or None bit_depth: Keep original bit-depth if None

    :param str src_pattern: Pattern used to inform sample data
    :param str tgt_pattern: Pattern used to compose resulting file name

    :param str suffix:
    :param str prefix:

    :param list or tuple or None extra_tags: List of extra tags to query from input file (flac input/output only)

    :param str group_name: Override group name

    :param list(list) rep_str: Basic search and replace in original name

    :param int transpose: Transpose notes in semitones
    :param bool force_pitch_from_name: Force note from the whole sample name

    :param str or None detect_pitch: Force pitch detection, 'corr' or 'pyin'
    :param str pitch_fraction_mode: 'keep', 'fine_tune' or 'fine_tune_lr'
    :param float or None pitch_fraction_override: Pitch fraction in semitone cents

    :param bool use_loop: Clear loop if False

    :param bake_pf: add_suffix, value

    :param bool test_run: Simulate process without doing anything

    :return:
    """
    p = Path(input_file)
    output_dir = output_dir or p.parent
    output_dir = Path(output_dir)

    smp = info_from_name(input_file, pattern=src_pattern, extra_tags=extra_tags, override=True,
                         force_pitch_from_name=force_pitch_from_name)
    name = smp.name

    # Disable loop option, for samples incorrectly set to loop (typically from sf2 files)
    if not use_loop:
        smp.loopStart, smp.loopEnd = None, None

    smp.group = group_name or smp.group

    # Search and replace
    if rep_str:
        for item in rep_str:
            if item[0] in name:
                smp.repstr = item[1]

    # Remove digits
    name = rm_digit_words(name)

    # Give some default name if the string is empty
    if name == '':
        name = smp.group

    # Detect pitch
    if detect_pitch:
        data, sr = sf.read(input_file)
        freq = pitch_detect(audio=data, sr=sr, mode=detect_pitch, resample=None, note_range=(20, 109), st_ed=.25)
        if not np.isnan(freq):
            pitch = hz_to_note(freq)
            smp.set_note(int(round(pitch)))
            smp.pitchFraction = (pitch - smp.note) * 100
        else:
            print('Pitch detection failed')

    if smp.note is None:
        smp.set_note(60)

    # Note transposition
    smp.transpose(transpose)

    if pitch_fraction_mode == 'override':
        smp.pitchFraction = pitch_fraction_override

    # Compose file name according to pattern
    if tgt_pattern:
        basename = compose_string(info=smp, pattern=tgt_pattern)
    else:
        basename = name

    filename = f'{prefix}{basename}{suffix}.{output_ext}'

    # Compose output file path
    output_file = Path.joinpath(output_dir, filename)

    # Increment file if needed
    i = 0
    while str(output_file) in check_list:
        i += 1
        output_file = output_dir.joinpath(f'{Path(filename).stem}_{i:03d}.{output_ext}')

    # Append a suffix if projected file already exists
    if output_file.exists():
        output_file = output_dir.joinpath(f'{output_file.stem}(!).{output_ext}')

    output_file = str(output_file)

    print(f'{input_file} -> {output_file}')
    if not test_run:
        data, sr = sf.read(input_file)

        # Keep left channel only if audio is mostly mono
        if is_stereo(data, db=-48) == 0:
            data = data[:, 0]

        match pitch_fraction_mode:
            case 'fine_tune':
                smp.pitchFraction, confidence = fine_tune(audio=data, sr=sr, note=smp.note, period_factor=3,
                                                          t=int(len(data) * .25), d=50, os=16, graph=False)
                print(f'Pitch fraction: {smp.pitchFraction}, Confidence: {confidence}')
            case 'fine_tune_lr':
                ft_lr = fine_tune_lr(audio=data, sr=sr, note=smp.note)
                if ft_lr is not None:
                    smp.pitchFraction = ft_lr
                    print(f'Pitch fraction (librosa): {ft_lr}')
                else:
                    print(f'Pitch fraction (librosa): failed, keeping former value')
            case 'keep' if smp.pitchFraction is not None:
                if Path(input_file).suffix == '.wav' and output_ext == 'flac':
                    print('Conformed pitch fraction for flac')
                    if smp.pitchFraction > 50:
                        smp.pitchFraction -= 100
                        if '{note}' not in src_pattern and '{noteName}' not in src_pattern:
                            smp.set_note(smp.note + 1)

        bit_depth = bit_depth or smp.params.sampwidth * 8
        if output_ext == 'flac':
            bit_depth = min(bit_depth, 24)
        subtypes = {16: 'PCM_16', 24: 'PCM_24', 32: 'FLOAT'}
        subtype = subtypes[bit_depth]

        # Temporary file name
        with tempfile.NamedTemporaryFile(dir=output_dir, suffix=f'.{output_ext}', delete=False) as temp_file:
            tmp_name = temp_file.name
        sf.write(tmp_name, data, samplerate=sr, subtype=subtype)

        # Write Metadata
        if output_ext == 'wav':
            append_metadata(tmp_name, note=smp.note, pitch_fraction=smp.pitchFraction,
                            loop_start=smp.loopStart, loop_end=smp.loopEnd)
            if hasattr(smp, 'cues'):
                append_markers(tmp_name, markers=smp.cues)
        elif output_ext == 'flac':
            attrs = ['note', 'pitchFraction', 'loopStart', 'loopEnd', 'loops', 'cues']
            if extra_tags:
                attrs.append(extra_tags)
            values = [getattr(smp, attr) for attr in attrs]
            md = dict(zip(attrs, values))
            set_md_tags(str(tmp_name), md=md)

    # Delete original files and rename temporary files
    if not test_run:
        os.remove(input_file)
        os.rename(tmp_name, output_file)

    if not test_run and bake_pf is not None:
        p = Path(output_file)
        baked_output_file = p.parent / f'{p.stem}{bake_pf[0]}{p.suffix}'
        baked_output_file = baked_output_file.resolve()
        apply_finetuning(input_file=output_file, output_file=baked_output_file, value=bake_pf[1],
                         no_overwriting=bake_pf[-1])

    return output_file


def rename_samples(root_dir, sub_dir='Samples', smp_fmt=('wav', 'flac'), **kwargs):
    """
    Batch rename
    """
    # Get files from dir
    if sub_dir:
        sample_dir = Path(root_dir).joinpath(sub_dir)
    else:
        sample_dir = Path(root_dir)

    samples = []
    for fmt in smp_fmt:
        samples.extend(Path(sample_dir).glob(f'*.{fmt}'))
    if not samples:
        return False

    result = []
    for smp in samples:
        new_name = rename_sample(input_file=str(smp), output_dir=str(sample_dir), check_list=result, **kwargs)
        result.append(new_name)

    return result


def apply_finetuning(input_file: Path | str, output_file: Path | str, value: float | bool = True,
                     no_overwriting: bool = True,
                     bit_depth: int | None = None, extra_tags: dict | None = None) -> Path | str:
    """
    Apply fine-tuning to given audio file by resampling
    metadata are updated accordingly
    :param no_overwriting:
    :param input_file:
    :param output_file:
    :param value: Fine-tuning to apply, if True apply embedded pitch fraction
    :param bit_depth: explicitly change the bit depth
    :param extra_tags: For FLAC only
    :return:
    """
    smp = info_from_name(str(input_file), extra_tags=extra_tags)
    p = Path(output_file)
    output_ext = p.suffix[1:]

    if value is True:
        value = smp.pitchFraction

    data, sr = sf.read(input_file)
    factor = 2 ** (value / 1200)

    print(f'value: {value} -> resample factor: {factor}')
    result = resample(data, sr, sr * factor, quality='VHQ')

    # Apply the pitch fraction to metadata
    smp.pitchFraction -= value
    if abs(smp.pitchFraction) < 1e-3:
        smp.pitchFraction = None

    # Adjust loops and cues with resampling factor
    if hasattr(smp, 'loops'):
        if len(smp.loops) > 0:
            smp.loops = [[int(round(val * factor)) for val in loop] for loop in smp.loops]
            smp.loopStart, smp.loopEnd = smp.loops[0]
    if hasattr(smp, 'cues'):
        smp.cues = [int(round(cue * factor)) for cue in smp.cues]

    bit_depth = bit_depth or smp.params.sampwidth * 8
    if output_ext == 'flac':
        bit_depth = min(bit_depth, 24)
    subtypes = {16: 'PCM_16', 24: 'PCM_24', 32: 'FLOAT'}
    subtype = subtypes[bit_depth]

    # Temporary file name
    output_dir = p.parent
    with tempfile.NamedTemporaryFile(dir=output_dir, suffix=f'.{output_ext}', delete=False) as temp_file:
        tmp_name = temp_file.name
    sf.write(tmp_name, result, samplerate=sr, subtype=subtype)

    # Write Metadata
    if output_ext == 'wav':
        append_metadata(tmp_name, note=smp.note, pitch_fraction=smp.pitchFraction,
                        loop_start=smp.loopStart, loop_end=smp.loopEnd)
        if hasattr(smp, 'cues'):
            append_markers(tmp_name, markers=smp.cues)
    elif output_ext == 'flac':
        attrs = ['note', 'pitchFraction', 'loopStart', 'loopEnd', 'loops', 'cues']
        if extra_tags:
            attrs.append(extra_tags)
        values = [getattr(smp, attr) for attr in attrs]
        md = dict(zip(attrs, values))
        set_md_tags(str(tmp_name), md=md)

    if no_overwriting:
        resolve_overwriting(output_file, mode='dir', dir_name='backup_', test_run=False)
    else:
        os.remove(output_file)

    os.rename(tmp_name, output_file)

    return output_file


def is_stereo(audio, db=-48):
    """
    Check if given audio data has enough difference between left and right to be considered stereo
    :param np.array audio:
    :param float db: difference threshold in dB beyond which the signal is considered "stereo enough"
    :return: -1 (audio has only one channel) 0 (audio has two channels but is mono) 1 (audio is stereo)
    :rtype: int
    """
    if audio.ndim < 2:
        return -1
    th = np.power(10, db / 20)
    result = np.max(np.abs(audio[:, 0] - audio[:, 1])) > th
    return (0, 1)[result]


# Chunk / metadata functions


def read_chunk(input_file, chunk_name='smpl', offset=0):
    """
    Retrieve data from named chunk

    :param str input_file:
    :param str chunk_name: Chunk id or name
    :param int offset: Header offset in bytes

    :return: Chunk data without 8 bytes header
    :rtype: bytes
    """
    with open(input_file, 'rb') as f:
        f.seek(offset)
        riff = f.read(12)
        if riff[:4] != b'RIFF' or riff[8:12] != b'WAVE':
            raise ValueError("Not a valid RIFF WAV file")

        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            chunk_id, chunk_size = struct.unpack('<4sI', header)

            # Read chunk data
            chunk_data = f.read(chunk_size)

            # If chunk_size is odd, read an extra padding byte (not included in chunk_data)
            # Source: https://www.daubnet.com/en/file-format-riff
            # "unused 1 byte present, if size is odd"
            if chunk_size % 2 == 1:
                f.read(1)

            if chunk_id == chunk_name.encode():
                return chunk_data


def read_metadata(input_file):
    """
    Basic riff 'smpl' metadata reader
    :param str input_file:
    :return: note, pitchFraction, loopStart, loopEnd, loops
    :rtype: dict
    """
    data = read_chunk(input_file, 'smpl')
    return bin_to_metadata(data, header=0)


def bin_to_metadata(data, header=8):
    """
    Convert binary 'smpl' chunk to metadata
    :param int header: Number of header bytes to skip - 0 if no header, typically 8 with header
    :param bytes data:
    :return: metadata
    :rtype: dict
    """
    tags = ['note', 'pitchFraction', 'loopStart', 'loopEnd', 'loops']  # Base tags
    note, pitch_fraction, loop_start, loop_end, loops = None, None, None, None, []

    if data:
        st = header
        ed = st + 36
        manuf, prod, smp_period, note, pitch_fraction, smptefmt, smpteoffs, numloops, smpdata = (
            struct.unpack('<iiiiIiiii', data[st:ed]))
        for i in range(numloops):
            st = ed
            ed = st + 24
            cuepointid, cuetype, start, end, fraction, playcount = struct.unpack('<iiiiii', data[st:ed])
            loops.append([start, end])
        if loops:
            loop_start, loop_end = loops[0]
        if pitch_fraction is not None:
            pitch_fraction = uint32_to_pitch_fraction(pitch_fraction)

    values = [note, pitch_fraction, loop_start, loop_end, loops]
    return dict(zip(tags, values))


def read_metadata_tags(input_file, extra_tags=()):
    """
    Read metadata stored as tags with support for extra tags
    Attempt to interpret them with case-sensitive names

    :param str input_file:
    :param list or tuple or None extra_tags:
    :return: Metadata with tag name as key
    :rtype: dict
    """
    audio = mutagen.File(input_file)
    tags = ['note', 'pitchFraction', 'loopStart', 'loopEnd', 'loops', 'cues']  # Base tags

    if extra_tags:
        tags.extend(extra_tags)
        tags = list(set(tags))  # Remove duplicates

    tags = [tag for tag in tags if tag in audio]
    values = [eval(audio[tag][0]) for tag in tags]
    data = dict(zip(tags, values))

    if 'loopStart' in data and 'loopEnd' in data:
        loop_start = data['loopStart']
        loop_end = data['loopEnd']
        loops = [loop_start, loop_end]
        # Create 'loops' data if absent or override
        if 'loops' not in data:
            data['loops'] = [loops]
        else:
            data['loops'][0] = loops
    else:
        data.pop('loops', None)

    return data


def get_all_tags_raw(input_file):
    """
    Get all ID3 tags from a given file without trying to interpret them
    Good enough to simply copy them from file to file
    Insufficient on their own to inform attributes as tags are not case-sensitive

    :param str input_file:
    :return: Metadata with tag name as key and value as a string
    :rtype: dict
    """
    audio = mutagen.File(input_file)
    tags = audio.keys()
    values = [audio[tag][0] for tag in tags]
    data = dict(zip(tags, values))
    return data


def get_cues(input_file):
    """
    Get markers from input file

    Info about this found here :
    https://sharkysoft.com/archive/lava/docs/javadocs/lava/riff/wave/doc-files/riffwave-content.htm

    :param str input_file: Input wav file
    :return:
    :rtype: list
    """
    cues = []
    data = read_chunk(input_file, 'cue ')
    header = 0

    if data:
        st = header
        ed = st + 4
        numcue = struct.unpack('<i', data[st:ed])[0]
        for i in range(numcue):
            chunkid, position, datachunkid, chunkstart, blockstart, sampleoffset = (
                struct.unpack('<iiiiii', data[ed + i * 24:ed + (i + 1) * 24]))
            cues.append(sampleoffset)

    return cues


def append_markers(input_file, markers):
    """
    Add markers to a given wav file in a minimalistic way (no labelling support)

    NOTE : The data is simply appended, so it's meant to be used once on a wav without any cues

    :param str input_file:
    :param list markers: List of marker positions in sample
    :return:
    """
    if not markers:
        return None

    num_cues = len(markers)

    # Get start id
    _, _, _, _, loops = read_metadata(input_file)
    start_id = len(loops) + 1

    riff_size = os.path.getsize(input_file) - 8  # File size minus header header chunks
    riff_size += (8 + num_cues * 24)  # Update size depending on the number of added markers

    # Update RIFF size to make the file valid after appending data
    with open(input_file, 'r+b') as wf:
        wf.seek(4)
        wf.write(as_chunk(riff_size, 4))
    wf.close()

    bin_data = b'cue ' + as_chunk(num_cues * 24 + 4, 4)
    bin_data += as_chunk(num_cues, 4)  # numcue

    for i, smp_ofs in enumerate(markers):
        # If loops are present this should use the next id
        bin_data += as_chunk(i + start_id, 4)  # "cue point name", actually a number id which must be unique
        bin_data += as_chunk(smp_ofs, 4)  # position, seems to be the same as sample offset for a marker
        bin_data += b'data'  # "chunkid" or name, optional label needs to be provided through another chunk
        bin_data += as_chunk(0, 4)  # "chunk start" ?
        bin_data += as_chunk(0, 4)  # "block start" ?
        bin_data += as_chunk(smp_ofs, 4)  # sample offset

    with open(input_file, 'ab') as f:
        f.write(bin_data)
    f.close()


def uint32_to_pitch_fraction(value):
    return value / 0xFFFFFFFF * 100


def as_chunk(value, length):
    """
    Encode a given value as a "little endian" chunk
    :param int value:
    :param int length: Chunk length in bytes
    :return:
    :rtype: binary
    """
    return value.to_bytes(length, byteorder='little')


def rm_alpha_chr(word: str):
    """
    Remove alpha character from string
    :param str word:
    :return:
    :rtype: str
    """
    return re.sub(r'[^0-9+\-.\n]', '', word)


def rm_zero_pad(word: str):
    """
    Remove zero padding from string
    Proper handling of '0'
    :param str word:
    :return:
    :rtype: str
    """
    return (word.lstrip('0'), word)[word == '0']


# Unused / Deprecated

def read_metadata_generic(input_file, blocksize=1):
    """
    Basic generic riff metadata reader
    Should work on both wav or flac (containing foreign metadata) files
    Not using Chunk function so slower
    :param str input_file: Path to a wav or flac file.
    :param int blocksize:
    :return: note, pitch_fraction, loop_start, loop_end
    :rtype: tuple
    """
    note, pitch_fraction, loop_start, loop_end, loops = None, None, None, None, None

    word = b'smpl'
    with open(input_file, 'rb') as f:
        header = f.read(4)
        # if header != b'RIFF' and header != b'fLaC':
        #     print('Not a valid file.')
        #     return None, None, None, None

        # if header == b'RIFF':
        #     blocksize = 4
        # else:
        #     blocksize = 1

        while True:
            block_data = f.read(blocksize)
            if not block_data:
                break  # Reached end of file

            # Read the metadata block
            if block_data == word[:blocksize]:
                if blocksize < 4:
                    block_data = f.read(4 - blocksize)  # Next bytes
                if block_data == word[blocksize:] or blocksize == 4:
                    block_data = f.read(40)[4:]
                    manuf, prod, smp_period, note, pitch_fraction, smptefmt, smpteoffs, numloops, smpdata = struct.unpack(
                        '<iiiiIiiii', block_data)
                    loops = []
                    for i in range(numloops):
                        block_data = f.read(24)
                        cuepointid, cuetype, start, end, fraction, playcount = struct.unpack('<iiiiii', block_data)
                        loops.append([start, end])
                    if loops:
                        loop_start, loop_end = loops[0]
                    break
    f.close()

    if pitch_fraction is not None:
        pitch_fraction = uint32_to_pitch_fraction(pitch_fraction)

    return note, pitch_fraction, loop_start, loop_end, loops
