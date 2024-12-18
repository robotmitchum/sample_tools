# coding:utf-8
"""
    :module: lossyFlac.py
    :description: Convert audio files to lossyFlac
    This is a pre-process helping flac compression at the expense of a slight loss in fidelity
    Benefits vary greatly depending on input

    Requires lossyWav.exe
    https://hydrogenaud.io/index.php/topic,112649

    :author: Michel 'Mitch' Pecqueur
    :date: 2024.07
"""

import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import soundfile as sf

from file_utils import resolve_overwriting
from sample_utils import Sample
from utils import set_md_tags


def lossy_flac(input_file, output_file, quality='X', no_overwriting=True):
    """
    Convert input audio file to lossyFlac which is a pre-process helping flac compression
    at the expense of a slight loss in fidelity

    :param str input_file:
    :param str output_file:

    :param str quality:
    (from lossyWav help)
    I, 'insane'       highest quality output, suitable for transcoding
    E, 'extreme'      higher quality output, suitable for transcoding
    H, 'high'         high quality output, suitable for transcoding
    S, 'standard'     default quality output, considered to be transparent
    C, 'economic'     intermediate quality output, likely to be transparent
    P, 'portable'     good quality output for DAP use, may not be transparent
    X, 'extra-portable'    lowest quality output, probably not transparent

    :param bool no_overwriting:

    :return:
    :rtype: str
    """
    audio, sr = sf.read(input_file)
    info = Sample(input_file)
    orig_size = os.path.getsize(input_file)

    bit_depth = info.params.sampwidth * 8
    bit_depth = max(min(bit_depth, 24), 16)
    subtypes = {16: 'PCM_16', 24: 'PCM_24', 32: 'FLOAT'}
    subtype = subtypes[bit_depth]

    p = Path(input_file)

    # Convert to temporary wav file
    with tempfile.NamedTemporaryFile(dir=p.parent, suffix='.wav', delete=False) as temp_file:
        tmp_wav = temp_file.name
    sf.write(tmp_wav, audio, sr, subtype=subtype)

    # Process temp wav with lossyWav
    print(f'{p.name} ...')
    cmd = f'lossyWav.exe "{tmp_wav}" -f -q {quality.upper()} -o "{p.parent}"'
    subprocess.run(cmd, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tmp_lossy_wav = Path.joinpath(p.parent, f'{Path(tmp_wav).stem}.lossy.wav')

    # Convert result to flac
    lossy_audio, sr = sf.read(str(tmp_lossy_wav))
    if no_overwriting and str(output_file) == input_file:
        resolve_overwriting(input_file, mode='dir', dir_name='backup_', test_run=False)
    sf.write(str(output_file), lossy_audio, samplerate=sr, subtype=subtype)

    # Add metadata
    # TODO : For input files which are already flac just query all tags and copy them
    keys = ['note', 'pitchFraction', 'loopStart', 'loopEnd', 'cues']
    values = [getattr(info, key) for key in keys]
    md = dict(zip(keys, values))
    md['lossyFlac'] = f'{quality.upper()}'
    set_md_tags(str(output_file), md=md)

    # Verify file size
    # TODO : Also calculate an overall ratio
    lossy_size = os.path.getsize(output_file)
    ratio = round(lossy_size / orig_size * 100, 2)
    if lossy_size >= orig_size:
        print(f'{ratio}% of original - FAIL, Rewriting original')
        sf.write(output_file, audio, samplerate=sr, subtype=subtype)
        md = dict(zip(keys, values))
        set_md_tags(str(output_file), md=md)
    else:
        print(f'{ratio}% of original - Success')

    # Delete intermediary temporary files
    os.remove(tmp_wav)
    os.remove(tmp_lossy_wav)

    return output_file


def main():
    if len(sys.argv) > 1:
        files = sys.argv[1:]
        try:
            for f in files:
                lossy_flac(f, f, quality='X', no_overwriting=True)
            print('Process complete')
        except Exception as e:
            traceback.print_exc()
            print('Encountered an error - Please check files or settings')
    else:
        print('No file supplied')
    input('Press ENTER to continue...')


if __name__ == "__main__":
    main()
