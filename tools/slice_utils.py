# coding:utf-8
"""
    :module: slice_utils.py
    :description: Quick and dirty functions to slice loops / isolate notes from chords and loops
    No UI for the moment
    :author: Michel 'Mitch' Pecqueur
    :date: 2025.12

# Examples:

# - Slice Loop -

# Detect transients and write them as cue markers
in_p = 'C:/AUDIO/SomeInputAudio.flac'
in_p = Path(in_p)
# Intermediate file allowing manual marker editing
out_p = in_p.with_suffix('.wav')
write_transients(input_path=in_p, output_path=out_p, size=1024, hop=256, smooth=2, dcim_mode='step', dcim_value=16,
 graph=True)

# Slice input audio
in_p = out_p
out_p = 'D:/Instruments/SomeLoop/Samples/slice.flac'
result = slice_audio(input_path=in_p, output_path=out_p,
                     steps=16, use_file_cues=True, cues=(),
                     start_index=36, write_reverse=True,
                     attack=1, decay=None, stretch=.5,
                     balance=True, align_phase=False)

# - Isolate Notes -

in_p = 'C:/AUDIO/SomeInputAudio.flac'
# Base name for written samples
out_p = 'C:/Instruments/SomeInstrument/Samples/SomeInstrument.flac'

# A 16 steps note sequence with some chords
note_seq = [('b2', 'e3'), ('c#3', 'f#3'), ('', 'b3'), ('', 'c#4'),
            ('', 'e4'), ('', 'f#4'), ('', 'b4'), ('', 'c#5'),
            (), (), (), (),
            (), (), (), ()]

isolate_notes(in_p, out_p, note_sequence=note_seq, stft_size=4096, fade_noise=.333,
make_loop=.75, resynth_mix='loop_tail',atonal_mix=1.0)

"""

from pathlib import Path

import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import ShortTimeFFT
from scipy.signal import find_peaks
from soxr import resample

import loop_sample as ls
from common_audio_utils import pad_audio, balance_lr, align_phase_lr, normalize_audio, peak
from common_math_utils import lerp
from crossfade_utils import crossfade_clips
from fft_resynth import get_harmonic_series
from fft_utils import filter_mask, h_cos
from mutate import split_noise_tonal
from sample_utils import info_from_name, info_to_md, append_metadata, append_markers, set_md_tags
from split_audio import apply_fade, trim_audio
from utils import note_to_hz, name_to_note


def slice_audio(input_path: Path | str | None, audio: np.ndarray | None = None, sr: int = 48000,
                output_path: Path | str | None = None,
                use_file_cues: bool = True, steps: int = 16, cues: np.ndarray | list = (),
                start_index: int = 1,
                balance: bool = False, align_phase: bool = False,
                write_reverse: bool = False,
                attack: float | None = 1, decay: float | None = 10, stretch=.5) -> list[np.ndarray] | list[Path]:
    """
    Slice given audio into equal slices

    :param input_path: Given audio file path
    :param audio: Audio array, used when input_path is NOne
    :param sr: Sampling Rate when using audio array

    :param output_path: if None return audio arrays instead of writing files

    :param use_file_cues: Use cues embedded in the input file if True, otherwise use step division
    :param steps: Number of desired slices
    :param cues: List of cues, overrides steps

    :param start_index: Start index when writing to files, typically 36 for drums

    :param balance: Balance LR volume
    :param align_phase: Align LR phase

    :param write_reverse: Also write reversed versions of slices

    :param attack: in ms
    :param decay: in ms
    :param stretch: Factor of audio length

    :return: list of paths or numpy arrays
    """
    if input_path:
        ip = Path(input_path)
        audio, sr = sf.read(str(ip))

        if use_file_cues:
            info = info_from_name(ip)
            cues = info.cues

    if cues:
        cues = list(sorted(cues))

    length = len(audio)
    slice_len = length / steps

    slice_samples = []
    reversed_samples = []

    use_cues = bool(len(cues) > 0)
    num_slices = (steps, len(cues))[use_cues]

    print(f'Number of slices: {num_slices}')

    for i in range(num_slices):
        if use_cues:
            st = cues[i]
            if i < num_slices - 1:
                ed = cues[i + 1] - 1
            else:
                ed = len(audio) - 1
        else:
            st = int(i * slice_len)
            ed = int((i + 1) * slice_len)

        for r, samples in enumerate([slice_samples, reversed_samples]):
            if r > 0 and not write_reverse:
                continue

            seg = np.copy(audio[st:ed])
            if r < 1 and stretch > 0:
                seg = stretch_slice(seg, stretch=stretch)

            if attack is not None:
                fd_in = min(int(attack * sr / 1000), len(seg))
                fade_in = (0, fd_in, 'log')
            else:
                fd_in = 0
                fade_in = None

            if decay is not None:
                fd_out = min(int(decay * sr / 1000), len(seg) - fd_in)
                fade_out = (fd_in, fd_out, 'log')
            else:
                fade_out = None

            if balance:
                seg = balance_lr(seg)
            if align_phase:
                seg = align_phase_lr(seg)
            slc = apply_fade(seg, fade_in=fade_in, fade_out=fade_out)
            slc, _ = trim_audio(slc, db=-96, prevent_empty=True)

            samples.append(slc)

    if output_path:
        if write_reverse:
            start_index = max(start_index, num_slices + 1)  # Prevent negative indices for reversed samples

        path_dict = {}
        path_list = []
        op = Path(output_path)
        op.parent.mkdir(parents=True, exist_ok=True)

        for i, (smp, rv_smp) in enumerate(zip(slice_samples, reversed_samples)):
            cmp = ({}, {'compression_level': 1.0})[op.suffix == '.flac']

            idx = i + start_index
            file_path = op.parent / f'{op.stem}_{idx:02d}{op.suffix}'
            path_dict[idx] = file_path
            sf.write(file_path, smp, sr, **cmp)

            if write_reverse:
                idx = i - num_slices + start_index
                file_path = op.parent / f'{op.stem}_{idx:02d}{op.suffix}'
                path_dict[idx] = file_path
                sf.write(file_path, rv_smp[::-1], sr, **cmp)

            path_list = [path_dict[idx] for idx in sorted(path_dict.keys())]
        return path_list

    return slice_samples


def isolate_note_stft(audio: np.ndarray, sr: int, stft_size: int = 4096,
                      freqs: float | list[float] = 440) -> np.ndarray:
    """
    Isolate a note from given audio by performing FFT filtering according to its harmonic series

    :param audio:
    :param sr:
    :param stft_size: Trade-of between pitch accuracy and time granularity
    :param freqs: Given fundamental frequency, can be a list

    :return:
    """

    length = len(audio)
    au_in = np.copy(audio)

    if length < stft_size:
        au_in = pad_audio(au_in, before=0, after=stft_size - length, mode='constant')

    nch = au_in.ndim
    out_data = np.zeros(shape=(len(au_in), nch))

    stft_duration = stft_size / sr
    h_list = get_harmonic_series(freqs, max_freq=sr / 2, number=0)
    f_idx = (h_list * stft_duration).astype(np.int32)

    for c in range(nch):
        if nch > 1:
            chn_data = au_in[:, c]
        else:
            chn_data = au_in

        win = np.hanning(stft_size)
        sft = ShortTimeFFT(win=win, hop=stft_size // 4, fs=sr)

        # Fade end
        fd_out_len = stft_size // 4
        fd_out = np.append(np.ones(len(chn_data) - fd_out_len), np.hanning(fd_out_len * 2)[fd_out_len:])
        chn_data *= fd_out

        # STFT
        sft_z = sft.stft(chn_data)
        mag = np.abs(sft_z)
        msk = np.zeros_like(mag)
        msk[f_idx,] = 1.0

        sft_chn = sft_z * msk
        chn_data = np.real(sft.istft(sft_chn))[:len(chn_data)]

        if nch > 1:
            out_data[:, c] = chn_data
        else:
            out_data = chn_data

    # Trim to original length
    result = out_data[:length]

    # Prevent clipping
    normalize_audio(result, prevent_clipping=True, db=-.1)

    return result


def isolate_notes(input_path: Path | str | None, output_path: Path | str | None = None, stft_size=4096,
                  note_sequence: list[tuple[str, ...]] = [('b2', 'e3')],
                  hp_noise: bool = True, fade_noise: float | None = 0.333,
                  make_loop: float | None = None, resynth_mix='loop_tail', atonal_mix=1.0, duration=.2):
    """
    Isolate a note sequence from a regular arpeggio, chords possibles

    :param input_path:
    :param output_path:
    :param stft_size:

    :param note_sequence: used to determine step division and notes
    a chord can be skipped by supplying an empty tuple
    a note can be skipped by supplying an empty string

    :param hp_noise: Apply high pass filter to noise content above root note frequency
    :param fade_noise: Noise fade length, length factor

    :param make_loop: Extend audio by FFT re-synthesis - This is the loop start value, 0 for full loop (pads)
    :param resynth_mix: 'loop_tail' or 'all'
    :param atonal_mix: 0 - 1, little to no effect if the noise part is faded
    :param duration: Loop duration in seconds

    :return:
    """
    ip = Path(input_path)
    audio, sr = sf.read(str(ip))
    nch = audio.ndim

    slice_count = len(note_sequence)
    slices = slice_audio(input_path=None, audio=audio, output_path=None, steps=slice_count)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
    else:
        p = ip

    for slc, notes in zip(slices, note_sequence):
        if not notes:
            continue
        ns, tn = split_noise_tonal(slc, sr, stft_size=stft_size, iterations=1, end_mode='fade')
        for note in notes:
            op = p.parent / f'{p.stem}_{note.upper()}{p.suffix}'

            if not note:
                continue

            f = note_to_hz(name_to_note(note))
            result = isolate_note_stft(audio=slc, sr=sr, stft_size=stft_size, freqs=f)

            slc_len = len(slc)

            # Add noise component

            # High Pass noise
            if hp_noise:
                cutoff = f
                band = 1
                fft_ns = np.fft.fft(ns)
                n_fft = len(fft_ns)
                h_band = max(abs(band) / 2, 1e-5) * np.sign(band)
                mask = filter_mask(cutoff - h_band, cutoff + h_band, n_fft=n_fft, sr=sr)
                mask = np.exp(lerp(np.log(1e-5), np.log(1), h_cos(mask)))
                ns = np.real(np.fft.ifft(fft_ns * mask))

            # Fade noise
            if fade_noise is not None:
                ns_fd_len = int(slc_len * fade_noise)
                ns = apply_fade(ns, fade_in=None, fade_out=(0, ns_fd_len, 'log'))
            result += ns / len(notes)

            # Balance stereo and normalize
            if nch > 1:
                result = balance_lr(result)
                result = align_phase_lr(result)

            result = normalize_audio(result, -1)
            result = apply_fade(result, fade_in=(0, 8, 'log'), fade_out=(slc_len - 64, 64, 'log'))

            ext = op.suffix[1:]

            cmp = ({}, {'compression_level': 1.0})[ext == 'flac']
            sf.write(str(op), result, sr, **cmp)

            info = info_from_name(op, override=True, pattern='{group}_{note}')

            # Write Metadata
            match ext:
                case 'wav':
                    if hasattr(info, 'cues'):
                        append_markers(op, markers=info.cues)
                    append_metadata(op, note=info.note, pitch_fraction=info.pitchFraction,
                                    loop_start=info.loopStart, loop_end=info.loopEnd)
                case 'flac':
                    md = info_to_md(info)
                    set_md_tags(op, md=md)

            if make_loop is not None:
                # Loop end by FFT resynthesis
                ls.loop_sample(input_file=op, output_file=op, bit_depth=16,
                               progress_bar=None,
                               worker=None, progress_callback=None, no_overwriting=False,
                               message_callback=None, shape_env=None,
                               detect_loop=None, crossfade=None,
                               resynth={'fft_range': 'custom', 'fft_start': make_loop, 'fft_end': 1.0,
                                        'duration': duration, 'atonal_mix': atonal_mix, 'freq_mode': 'custom',
                                        'freqs': [f], 'resynth_mix': resynth_mix, 'fade_in': 1 - make_loop,
                                        'fade_out': 1 - make_loop, 'width': .5})


def write_transients(input_path: Path | str | None, output_path: Path | str | None = None, steps: int | None = 16,
                     dcim_mode: str = 'step', dcim_value: int = 16, **kwargs):
    """
    Detect and write transient cues
    :param input_path:
    :param output_path:
    :param kwargs: Pass arguments to detect transients
    :return:
    """
    ip = Path(input_path)
    audio, sr = sf.read(str(ip))
    info = info_from_name(ip, override=False)

    subtypes = {16: 'PCM_16', 24: 'PCM_24', 32: 'FLOAT'}
    bit_depth = info.params.sampwidth * 8
    bit_depth = max(bit_depth, 16)

    if dcim_mode == 'step':
        dcim = len(audio) // dcim_value
    else:
        dcim = dcim_value

    if steps is not None:
        cues = [len(audio) * i // steps for i in range(steps)]
    else:
        cues = detect_transients(audio=audio, dcim=dcim, **kwargs).tolist()

    info.cues = cues

    if output_path is None:
        output_path = ip

    op = Path(output_path)
    ext = op.suffix[1:]

    cmp = ({}, {'compression_level': 1.0})[ext == 'flac']

    if ext == 'flac':
        bit_depth = min(bit_depth, 24)

    sf.write(str(op), audio, sr, subtype=subtypes[bit_depth], **cmp)

    match ext:
        case 'wav':
            append_markers(op, markers=info.cues)
            append_metadata(op, note=info.note, pitch_fraction=info.pitchFraction,
                            loop_start=info.loopStart, loop_end=info.loopEnd)
        case 'flac':
            md = info_to_md(info)
            set_md_tags(op, md)

    return op


# Auxiliary functions


def detect_transients(audio: np.ndarray, size: int = 1024, hop: int = 256,
                      smooth: int = 3, dcim: int = 256,
                      graph: bool = False):
    """
    Simple rms based transient detection

    :param audio:
    :param dcim: Min length for cue decimation
    :param smooth: Smooth rms data before peak detection
    :param size: Window size
    :param hop: Hop size
    :param graph: Display a graph about the process

    :return: Detected cues
    """
    nch = audio.ndim
    if nch > 1:
        data = np.mean(audio, axis=-1)
    else:
        data = np.copy(audio)

    num_frames = 1 + (len(data) - size) // hop

    frames = np.stack([data[i * hop:i * hop + size] for i in range(num_frames)])
    energy = np.sqrt(np.mean(frames ** 2, axis=1))

    if smooth:
        energy = gaussian_filter1d(energy, sigma=smooth, mode='nearest', cval=0)
    mx = peak(energy)
    energy /= mx

    cues = find_peaks(energy)[0]
    if 0 not in cues:
        cues = np.concatenate(([0], cues))

    # cues = decimate_cues(cues=cues, min_len=dcim // hop)
    cues = decimate_cues_rms_check(cues=cues, data=energy, min_len=dcim // hop)

    # Shift and rescale cues
    cues = np.maximum((cues - 2) * hop, 0)

    # Snap to nearest previous zero cross
    zc = np.argwhere(np.abs(np.diff(np.sign(data))) > 0).reshape(-1) + 1

    zc_cues = []
    for cue in cues:
        idx = np.argmin(np.abs(zc - cue))
        if zc[idx] > cue:
            idx = np.clip(idx - 1, 0, len(zc) - 1)
        zc_cues.append(zc[idx])

    cues = np.array(zc_cues)

    if graph:
        plt.figure()
        plt.title('Transient Detection')
        plt.plot(data, label='Waveform')
        plt.plot(resample(energy, 1, hop, quality='QQ'), label='Energy')
        plt.vlines(cues, ymin=-1, ymax=1, colors='red', label='Detected Cues')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()

    return cues


def decimate_cues(cues: np.ndarray, min_len: int = 256, keep_last: bool = False) -> np.ndarray:
    """
    Remove cues too close from their previous cue
    :param cues: Cues to decimate
    :param min_len: Min length between cues (in samples)
    :param keep_last: Always keep last cue
    :return: processed cues
    """
    if not len(cues):
        return cues
    result = [cues[0]]
    for cue in cues[1:]:
        if cue - result[-1] >= min_len:
            result.append(cue)
    if keep_last and cues[-1] not in result:
        result.append(cues[-1])
    return np.array(result)


def decimate_cues_rms_check(cues: np.ndarray, data: np.ndarray, min_len: int = 256,
                            keep_last: bool = False) -> np.ndarray:
    """
    Remove cues too close from their previous cue with peak amplitude check
    :param cues: Cues to decimate
    :param data: Corresponding rms data to determine peaks to discard
    (keep the peak if its energy is higher than previous cue)
    :param min_len: Min length between cues (in samples)
    :param keep_last: Always keep last cue
    :return: processed cues
    """
    if not len(cues):
        return cues
    result = [cues[0]]
    for cue in cues[1:]:
        if cue - result[-1] >= min_len or data[cue] >= data[result[-1]]:
            result.append(cue)
    if keep_last and cues[-1] not in result:
        result.append(cues[-1])
    return np.array(result)


def stretch_slice(audio: np.ndarray, stretch: float = .4, crossfade: float = .05) -> np.ndarray:
    """
    Extend an audio segment by applying ping-pong to its tail, inspired by Recycle

    :param audio: Input audio
    :param stretch: Factor of audio length
    :param crossfade: Cross-fade tail with original audio, factor of audio length

    :return: Processed audio
    """
    len_slc = len(audio)
    tail = np.copy(audio[int(len_slc - len_slc * stretch):])[::-1]
    tail = apply_fade(tail, fade_in=None, fade_out=(0, len(tail), ''))
    fd_len = int(len_slc * crossfade)
    if fd_len > 0:
        return crossfade_clips(audio, -tail, len_slc - fd_len, fd_len, 'smoothstep')
    else:
        return np.concatenate((audio, -tail))
