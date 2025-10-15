# coding:utf-8
"""
    :module: lossy_flac.py
    :description: Straightforward almost line-for-line Python port of David Robinson's lossyFLAC MATLAB code
    :author: David Robinson (2007), Michel 'Mitch' Pecqueur (Python port)
    :date: 2025.10

    This version avoids classes and keeps a linear flow matching the MATLAB script as closely as possible

    #####

    Original note from David Robinson :

    David Robinson, 2007
    This code is open source. I'll pick a licence later (need advice), but:
    Any changes to this MATLAB code must be shared freely
    Any changes to the algorithm must be shared freely
    Any other implementations / interpretations must be shared freely
    You are free to include any implementation in commercial and/or closed source code,
    but must give an acknowledgement in help/about or similar.

    No warrantee / guarantee. This is work in progress and not debugged.

    #####

    A recent implementation such as lossyWav maintained by Nick Currie is using letters for quality presets
    https://hydrogenaudio.org/index.php/topic,112649.0.html

    I don't know how this original implementation compares
    It uses the raw spectral minimum method without modern threshold shaping
    Let's call this DROP as for "David Robinson Original Preset"
    Maybe roughly equivalent to “Standard” quality in later lossyWAV versions

"""

import os
import pickle
from pathlib import Path

import mutagen
import numpy as np
import soundfile as sf
from platformdirs import user_cache_path
from scipy.fft import fft
from scipy.signal import convolve


def lossy_flac(input_file: Path | str | None, data: np.ndarray = None, sr: int | None = None,
               output_file: Path | str | None = None, bit_depth: int | None = 16,
               md: dict | None = None, check_size: bool = True) -> tuple[Path, tuple[int, int] | None]:
    """
    Convert an input audio file or an input audio array to a lossyFLAC or lossyWAV file

    :param input_file:
    :param data: Audio array
    :param sr: Sampling rate
    :param output_file:
    :param bit_depth: Resulting bit-depth, Same as input if None
    :param md: Metadata tags as dict
    :param check_size: Ensure lossy is smaller otherwise use regular flac as fallback

    :return: Resulting file path, statistics (ref_size,lossy_size) if check_size is True
    """

    if not input_file and not output_file:
        raise 'Please provide at least an audio array a sampling rate and an output file path'

    bd_dict = {'PCM_16': 16, 'PCM_24': 24, 'PCM_32': 32, 'FLOAT': 32, 'DOUBLE': 64}

    if input_file:
        data, sr = sf.read(str(input_file), always_2d=True)

        if bit_depth is None:
            info = sf.info(str(input_file))
            bit_depth = bd_dict.get(info.subtype, 16)

        if not output_file:
            output_file = Path(input_file).with_suffix('.lossy.flac')

        md = md or {}
        if input_file.suffix == '.flac':
            md = get_md_tags(input_file) | md

    sr = sr or 48000

    # Original parameters from MATLAB script
    noise_threshold_shift = 0
    low_frequency_limit = 20
    high_frequency_limit = 16000
    fix_clipped = 0
    flac_blocksize = 4096
    noise_averages = 1000
    inaudible = 1e-10
    analysis_time = [2.0e-2, 1.5e-3]
    spreading_function = [np.array([0.25, 0.25, 0.25, 0.25]), np.array([0.25, 0.25, 0.25, 0.25])]

    # Rough estimate of bit-depth
    bd = 24 if data.dtype.kind == 'f' else 16
    if fix_clipped == 1:
        bd = 23

    samples, channels = data.shape
    in_audio_int = data * (2 ** (bd - 1)) + inaudible

    # FFT setup
    fft_length = []
    low_frequency_bin = []
    high_frequency_bin = []
    number_of_analyses = len(analysis_time)

    for analysis_number in range(number_of_analyses):
        desired = analysis_time[analysis_number] * sr
        fft_len = int(2 ** np.round(np.log2(desired)))
        fft_len = min(fft_len, flac_blocksize)
        fft_length.append(fft_len)

        pad = (len(spreading_function[analysis_number]) - 1) / 2.0
        lfb = int(round(fft_len * low_frequency_limit / sr + pad))
        lfb = max(lfb, 2)
        if lfb > fft_len // 2:
            raise ValueError('low frequency too high')

        hfb = int(round(fft_len * high_frequency_limit / sr + pad))
        if hfb < 2:
            raise ValueError('high frequency too low')
        hfb = min(hfb, fft_len // 2)

        low_frequency_bin.append(lfb)
        high_frequency_bin.append(hfb)

    cache_dir = user_cache_path('lossy_flac', appauthor=False)
    cache_dir.mkdir(parents=True, exist_ok=True)
    vars_file = cache_dir / (f'lossy_vars__sr{sr}_bd{bd}_noa{number_of_analyses}_fft{fft_length}'
                             f'_lfb{low_frequency_bin}_hfb{high_frequency_bin}_nts{noise_threshold_shift}.pkl')

    if os.path.exists(vars_file):
        # Find previously computed quantization noise thresholds
        with open(vars_file, 'rb') as f:
            cache = pickle.load(f)
        threshold_index = cache['threshold_index']
    else:
        # If not, calculate quantization noise at each bit in these FFTs
        reference_threshold = []
        threshold_index = []
        for analysis_number in range(number_of_analyses):
            fft_len = fft_length[analysis_number]
            lfb = low_frequency_bin[analysis_number]
            hfb = high_frequency_bin[analysis_number]
            ref = np.zeros((noise_averages, bd))
            for av_number in range(noise_averages):
                noise_sample = np.random.rand(fft_len)
                for bits_to_remove in range(1, bd + 1):
                    this_noise_sample = np.floor(noise_sample * (2 ** bits_to_remove)) - (2 ** (bits_to_remove - 1))
                    windowed = this_noise_sample * np.hanning(fft_len)
                    spectrum = np.abs(fft(windowed))
                    smoothed = convolve(spectrum, spreading_function[analysis_number], mode='same')
                    fft_result = 20 * np.log10(np.maximum(smoothed, 1e-20))
                    ref[av_number, bits_to_remove - 1] = np.mean(fft_result[lfb - 1:hfb])
            reference_threshold.append(np.mean(ref, axis=0) - noise_threshold_shift)

            max_t = int(round(20 * np.log10(2 ** (bd + 4))))
            idx = np.zeros(max_t + 1, dtype=int)
            for t in range(1, max_t + 1):
                smaller = np.where(reference_threshold[-1] < t)[0]
                idx[t] = 0 if smaller.size == 0 else int(np.max(smaller) + 1)
            threshold_index.append(idx)

        with open(vars_file, 'wb') as f:
            pickle.dump({'threshold_index': threshold_index}, f)

    # Spectral analysis to find min bins
    min_bin = []
    min_bin_length = []
    for analysis_number in range(number_of_analyses):
        fft_len = fft_length[analysis_number]
        hop = fft_len // 2
        nblocks = int(np.floor(samples / hop)) - 1
        if nblocks < 1:
            nblocks = 1
        mb = np.zeros((nblocks, channels))
        window = np.hanning(fft_len)
        spread = spreading_function[analysis_number]

        for block_start in range(0, samples - fft_len, hop):
            block_number = block_start // hop
            for ch in range(channels):
                block = in_audio_int[block_start:block_start + fft_len, ch]
                spec = np.abs(fft(window * block))
                smoothed = convolve(spec, spread, mode='same')
                db = 20 * np.log10(np.maximum(smoothed, 1e-20))
                mb[block_number, ch] = np.min(
                    db[low_frequency_bin[analysis_number] - 1:high_frequency_bin[analysis_number]])
        min_bin.append(mb)
        min_bin_length.append(len(mb))

    # Quantization per FLAC block
    bits_to_remove = np.zeros(int(np.ceil(samples / flac_blocksize)), dtype=int)

    for block_start in range(0, samples, flac_blocksize):
        block_number = block_start // flac_blocksize
        block_end = min(block_start + flac_blocksize, samples)

        table = np.zeros((number_of_analyses, channels), dtype=int)
        for analysis_number in range(number_of_analyses):
            fft_len = fft_length[analysis_number]
            hop = fft_len // 2
            first_block = int(block_start / hop)
            last_block = first_block + int(flac_blocksize / hop)
            first_block = max(first_block, 1)
            last_block = min(last_block, min_bin_length[analysis_number])

            for ch in range(channels):
                seg = min_bin[analysis_number][first_block - 1:last_block, ch]
                if seg.size == 0:
                    this_min_bin = 0
                else:
                    this_min_bin = int(np.round(np.min(seg)))
                if this_min_bin < 1:
                    table[analysis_number, ch] = 0
                else:
                    idx = threshold_index[analysis_number]
                    table[analysis_number, ch] = idx[this_min_bin] if this_min_bin < len(idx) else idx[-1]

        bits_to_remove[block_number] = np.min(table)

        if bits_to_remove[block_number] > 0:
            factor = 2 ** bits_to_remove[block_number]
            in_audio_int[block_start:block_end, :] = np.round(in_audio_int[block_start:block_end, :] / factor) * factor

    if fix_clipped == 1:
        bd = 24

    result = in_audio_int / (2 ** (bd - 1))

    subtype_dict = {16: 'PCM_16', 24: 'PCM_24'}
    subtype = subtype_dict.get(min(bit_depth, 24), None)
    cmp = ({}, {'compression_level': 1.0})[output_file.suffix == '.flac']
    sf.write(str(output_file), result, sr, subtype=subtype, **cmp)

    stats = None

    if output_file.suffix == '.flac':
        if check_size:
            ref_flac = cache_dir / output_file.name
            sf.write(str(ref_flac), data, sr, subtype=subtype, **cmp)
            lossy_size = output_file.stat().st_size
            ref_size = ref_flac.stat().st_size

            if lossy_size >= ref_size:
                print(f'[WARN] {output_file.name} inefficient lossy compression, fall back to regular FLAC')
                ref_flac.replace(output_file)
                md.pop('comment', None)
            else:
                md['comment'] = 'LossyFLAC: quality=DROP encoder=libsndfile'

            stats = (ref_size, lossy_size)

            if ref_flac.is_file():
                try:
                    os.remove(ref_flac)
                except Exception as e:
                    print(e)
                    pass

        # Write tags
        set_md_tags(output_file, md)

    return Path(output_file), stats


def get_md_tags(input_file: Path | str) -> dict:
    """
    Get tags as dict

    :param input_file:
    """
    audio = mutagen.File(str(input_file))
    return {k: v[0] for k, v in audio.tags.items()}


def set_md_tags(input_file: Path | str, md: dict | None = None):
    """
    Set tags from dict
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
