# coding:utf-8
"""
    :module: adjust_sr.py
    :description: Adjust sampling rate
    No resampling is performed so the pitch is changed but, note and pitch fraction info are updated accordingly
    :author: Michel 'Mitch' Pecqueur
    :date: 2025.12

# Examples:

ip = 'C:/AUDIO/SomeInputAudio.wav'
ip = Path(ip)
op = ip.parent / f'{ip.stem}.wav'
adjust_sr(ip, op, 22050, cents=0)

# Semitone cents adjustment
adjust_sr(ip, op, new_sr=None, cents=-25)
"""

from pathlib import Path

import soundfile as sf

from sample_utils import info_from_name, info_to_md, set_md_tags, append_metadata, append_markers
from utils import note_to_hz, hz_to_note, note_to_name


def adjust_sr(input_file: Path | str, output_file: Path | str | None, new_sr: int | None, cents: float = 0) -> Path:
    """
    Adjust sampling rate - no resampling is done so pitch changes - then update note and pitch fraction accordingly

    :param input_file:
    :param output_file: Replace original if None
    :param new_sr: Desired sampling rate, keep base sampling rate if None
    :param cents: Extra sampling rate adjustment with given semitone cent offset
    Equivalent to finetuning without resampling

    :return: Output file path
    """
    p = Path(input_file)

    y, sr = sf.read(str(p))
    info = info_from_name(p)

    bit_depth = info.params.sampwidth * 8
    bit_depth = max(min(bit_depth, 24), 16)
    subtypes = {16: 'PCM_16', 24: 'PCM_24', 32: 'FLOAT'}
    subtype = subtypes[bit_depth]

    note = info.note + info.pitchFraction / 100
    f = note_to_hz(note)

    new_sr = new_sr or sr

    cents_factor = 1.0
    if cents != 0:
        cents_factor = 2 ** (cents / 1200)
        new_sr = int(round(new_sr * cents_factor))

    factor = new_sr / sr
    new_f = f * factor
    new_pitch = hz_to_note(new_f)

    note = round(new_pitch)
    pitch_fraction = round((new_pitch - note) * 100, 3)
    n, o = note_to_name(note)

    print(f'Cents factor: {cents_factor}')
    print(f'{sr} -> {new_sr}')
    print(f'{f} -> {new_f}')
    print(f'{n}{o} ({note}) {pitch_fraction}')

    output_file = output_file or input_file
    of = Path(output_file)
    ext = of.suffix[1:]
    cmp = ({}, {'compression_level': 1.0})[ext == '.flac']

    sf.write(str(of), y, new_sr, subtype=subtype, *cmp)

    # Add 'smpl' metadata
    if ext == 'wav':
        if hasattr(info, 'cues'):
            append_markers(of, info.cues)
        append_metadata(of, note=note, pitch_fraction=pitch_fraction, loop_start=info.loopStart,
                        loop_end=info.loopEnd)
    # Add metadata as tags for flac
    elif ext == 'flac':
        set_md_tags(of, md=info_to_md(info))

    return output_file
