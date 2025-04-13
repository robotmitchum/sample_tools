# coding:utf-8
"""
    :module: instrument_utils.py
    :description: Class and functions to perform queries and statistics or establish settings at the instrument level
    (taking full sample set in consideration instead of each individual samples)
    :author: Michel 'Mitch' Pecqueur
    :date: 2024.08
"""
from pathlib import Path

import numpy as np
import pandas as pd

from common_math_utils import clamp, random_from_string
from common_math_utils import lerp
from file_utils import recursive_search
from parseAttrString import parse_string, compose_string
from sample_utils import info_from_name
import inspect


class Instrument:
    """
    Instrument class
    """

    def __init__(self, root_dir, smp_subdir='Samples', smp_fmt=('wav', 'flac'), files=None,
                 pattern='{group}_{note}_{vel}', transpose=0, **kwargs):

        self.root_dir = root_dir
        self.smp_subdir = smp_subdir
        if root_dir:
            self.name = Path(root_dir).stem
        else:
            self.name = 'Instrument'

        self.files = files

        self.smp_fmt = smp_fmt
        self.pattern = pattern
        self.transpose = transpose
        self.kwargs = kwargs

        self.df = pd.DataFrame()
        self.samples = []

        # Build and sort data frame
        self.build_df()
        if self.df is None:
            return

        self.sort_df()

        # Instrument data
        self.limit = [0, 127]

        self.groups = None
        self.group_trigger = None
        self.vels_per_group_trigger = None
        self.notes_per_group_trigger_vel = None
        self.seqs_per_group_trigger_vel_note = None
        self.zones = None

        self.set_instrument_data()

    def build_df(self):
        """
        Retrieve instrument samples with attributes using pattern and build a data frame from it
        set df and samples attributes
        """
        if self.files is None:
            self.df, self.samples = samples_to_df(self.root_dir, smp_subdir=self.smp_subdir, smp_fmt=self.smp_fmt,
                                                  pattern=self.pattern, transpose=self.transpose, **self.kwargs)
        else:
            self.df, self.samples = audio_files_to_df(files=self.files, pattern=self.pattern, transpose=self.transpose,
                                                      **self.kwargs)

    def sort_df(self):
        self.df, self.samples = sort_df(self.df, self.samples)

    def query_df(self, fltr, attr=None):
        """
        Multi-filtered query of a given data frame

        :param pandas.core.frame.DataFrame df:
        :param dict or None fltr: Query filter as a dict
        :param str or None attr: Optional attribute to query, return a filtered data frame otherwise

        :return: filtered result
        """

        return query_df(df=self.df, fltr=fltr, attr=attr)

    def set_limit(self, limit: list | dict[list] | bool = 'autox2', mode: str = 'shared'):
        """
        Set limit attribute, list or a dict depending on mode
        See limit_from_notes function
        :param limit:
        :param mode: 'shared' or 'group'
        """
        if mode == 'shared':
            smp_notes = self.df['note'].unique().tolist()
            result = limit_from_notes(smp_notes, limit=limit)
        else:
            smp_notes = {g: self.query_df(fltr={'group': g}, attr='note').unique().tolist() for g in self.groups}
            result = {g: limit_from_notes(n, limit=limit) for g, n in smp_notes.items()}
        self.limit = result

    # Pre-made instrument queries

    def set_group_trigger(self):
        """
        Set groups_triggers attribute
        List of tuples of each (group,trigger) combination
        """
        result = []
        for (grp, trg), group_data in self.df.groupby(['group', 'trigger'], observed=True):
            result.append((grp, trg))
        self.group_trigger = result

    def set_vels_per_group_trigger(self):
        """
        Set vels_per_group_trigger attribute
        Nested dict structure with group,trigger as keys and list of velocities as values
        """
        result = dict()
        for (grp, trg), group_data in self.df.groupby(['group', 'trigger'], observed=True):
            result.setdefault(grp, {})[trg] = group_data['vel'].unique().tolist()
        self.vels_per_group_trigger = result

    def set_notes_per_group_trigger_vel(self):
        """
        Set notes_per_group_trigger_vel attribute
        Nested dict structure with group,trigger,vel as keys and list of notes as values
        """
        result = dict()
        for (grp, trg, vel), group_data in self.df.groupby(['group', 'trigger', 'vel'], observed=True):
            result.setdefault(grp, {}).setdefault(trg, {})[vel] = group_data['note'].unique().tolist()
        self.notes_per_group_trigger_vel = result

    def set_seqs_per_group_trigger_vel_note(self):
        """
        Set seqs_per_group_trigger_vel_note attribute
        Nested dict structure with group,trigger,vel,note as keys and list of sequence positions as values
        """
        result = dict()
        for (grp, trg, vel, note), group_data in self.df.groupby(['group', 'trigger', 'vel', 'note'], observed=True):
            result.setdefault(grp, {}).setdefault(trg, {}).setdefault(vel, {})[note] = group_data[
                'seqPosition'].unique().tolist()
        self.seqs_per_group_trigger_vel_note = result

    def set_instrument_data(self):
        """
        Set all of the above pre-made queries
        :return:
        """
        self.groups = self.df['group'].unique().tolist()
        self.set_group_trigger()
        self.set_vels_per_group_trigger()
        self.set_notes_per_group_trigger_vel()
        self.set_seqs_per_group_trigger_vel_note()

    # Extra defs

    def set_zones(self):
        """
        Set zones attribute
        Nested dict structure with group,trigger,vel,note as keys and grouped data frame with name as values

        Used to speed-up fake Round-Robin mapping
        """

        result = dict()
        for (grp, trg, vel, seq), group_data in self.df.groupby(['group', 'trigger', 'vel', 'seqPosition'],
                                                                observed=True):
            result.setdefault(grp, {}).setdefault(trg, {}).setdefault(vel, {})[seq] = group_data['name']
        self.zones = result

    def pad_vel(self):
        """
        Re-use / duplicate note samples for velocity layers with smaller note span in data frame
        Change name and velocity then sort data frame again

        Typical usage: instruments with velocity layers having a very different number of note samples
        """

        # Loop over group,trigger layers
        for grp, trg in self.group_trigger:
            vels = self.vels_per_group_trigger[grp][trg]

            vel_notes = self.df.groupby(['group', 'trigger', 'vel'], observed=True)['note'].unique().tolist()
            for vn_idx, (vel, notes) in enumerate(zip(vels[:-1], vel_notes[:-1])):
                # Get pad notes from higher vel layers
                pad_notes = get_pad_values(notes, vel_notes[vn_idx + 1])

                for note in pad_notes:
                    src_vel = vels[vn_idx + 1]

                    # Get sample(s) using found note and src_vel values
                    smps = self.query_df(fltr={'note': note, 'vel': src_vel}).copy()
                    smps['vel'] = vel
                    names = smps['name']

                    # Change vel and name accordingly in dataframe copy
                    new_names = []
                    for name in names:
                        attrs = parse_string(name, pattern=self.pattern)
                        attrs['vel'] = vel
                        new_names.append(compose_string(attrs, pattern=self.pattern))
                    smps['name'] = new_names

                    # Append as new entry in data frame and samples
                    self.df = pd.concat([self.df, smps], ignore_index=True)
                    for smp_idx, name in zip(smps.index, new_names):
                        new_smp = self.samples[smp_idx].__copy__()
                        new_smp.name = name
                        new_smp.set_vel(vel)
                        self.samples.append(new_smp)

        # Sort again
        self.sort_df()
        self.set_instrument_data()

    def pitch_fraction(self, mode='on', value=5., seed='', apply=False):
        """
        Modify pitch fraction

        :param mode: 
        'off'	Ignore pitch fraction, setting it to 0
        'on'	Use pitch fraction, may sound 'sterile' because tuning can be too perfect
        'on_rand'	Use and add given random
        'on_threshold'	Use over given threshold
        'on_threshold_rand'	Use over given threshold and add given random
        'mean_threshold'	Use pitch fraction over threshold and mean under threshold
        'mean_blend'	Blend between mean and pitch fraction (0-100)
        
        :param value: Threshold / Value
        
        :param str or int seed: Alter per-sample random generator
        :param bool apply: Apply modification on instrument data
        
        :return: Return pitchFraction
        :rtype:
        """

        pfs = self.df['pitchFraction'].copy()
        result = pfs.copy()

        match mode:
            case 'mean_threshold':
                mean = mean_pf(pfs, x=0)
                idx = abs(pfs - mean) < value
                result[idx] = mean
            case 'mean_blend':
                result = mean_pf(pfs, x=value / 100)
            case 'mean_scale':
                result = mean_pf(pfs, mode='scl', x=value)
            case 'on_threshold':
                result[abs(pfs) < value] = 0
            case 'on_rand' | 'on_threshold_rand':
                pf_rand = (self.df['name'] + f'{seed}').apply(random_from_string)
                result += pf_rand * value
                if mode == 'on_threshold_rand':
                    result[abs(pfs) < value] = 0
            case 'off':
                result = [0] * len(result)

        if apply:
            self.df['pitchFraction'] = result
            for smp, pf in zip(self.samples, result):
                smp.pitchFraction = pf

        return result


# Data frame functions
def samples_to_df(root_dir, smp_subdir='Samples', smp_fmt=('wav', 'flac'),
                  pattern='{group}_{note}_{vel}', exclude=(), transpose=0, **kwargs):
    """
    Convert Samples directory to a pandas data frame structure
    The data frame can then be used to retrieve collective information efficiently
    Also return a list of Sample objects ordered using the indices of the dataframe

    :param str root_dir:
    :param str smp_subdir:
    :param list or tuple smp_fmt:
    :param list or tuple exclude:
    :param str pattern: Pattern used to parse sample attributes
    :param int transpose:
    :param dict kwargs: Extra arguments used for info_from_name function

    :return: Instrument Dataframe, List of Samples
    :rtype: pandas.core.frame.DataFrame, list(Sample)
    """

    files = recursive_search(root_dir=Path.joinpath(Path(root_dir), smp_subdir), input_ext=smp_fmt,
                             exclude=exclude)
    if not files:
        return None, None

    result = audio_files_to_df(files, pattern=pattern, transpose=transpose, **kwargs)
    return result


def audio_files_to_df(files, pattern='{group}_{note}_{vel}', transpose=0, **kwargs):
    """
    Convert list of audio files to a pandas data frame structure
    The data frame can then be used to retrieve collective information efficiently
    Also return a list of Sample objects ordered using the indices of the dataframe

    :param list files:
    :param str pattern: Pattern used to parse sample attributes
    :param int transpose:
    :param dict kwargs: Extra arguments used for info_from_name function

    :return: Instrument Dataframe, List of Samples
    :rtype: pandas.core.frame.DataFrame, list(Sample)
    """

    if not files:
        return None, None

    samples = []

    # Filter unwanted keyword arguments
    func_args = inspect.getfullargspec(info_from_name)[0]
    kwargs = {k: v for k, v in kwargs.items() if k in func_args}

    for f in files:
        smp = info_from_name(str(f), pattern=pattern, **kwargs)
        smp.transpose(transpose)
        samples.append(smp)

    # Convert list of Sample objects to a DataFrame
    attrs = ['name', 'path', 'group', 'trigger', 'note', 'pitchFraction', 'vel', 'seqPosition']
    data = {attr: [getattr(s, attr) for s in samples] for attr in attrs}

    df = pd.DataFrame(data)

    # - Arrange data -

    # Set some defaults for undefined entries
    for attr in attrs:
        match attr:
            case 'trigger':
                df[attr] = df[attr].map(lambda x: 'attack' if pd.isna(x) else x)
            case 'note':
                df[attr] = df[attr].map(lambda x: 60 if pd.isna(x) else x)
            case 'pitchFraction':
                df[attr] = df[attr].map(lambda x: .0 if pd.isna(x) else x)
            case 'vel':
                df[attr] = df[attr].map(lambda x: 127 if pd.isna(x) else x)
            case 'seqPosition':
                df[attr] = df[attr].map(lambda x: 1 if pd.isna(x) else x)

    df, samples = sort_df(df, samples)

    return df, samples


def sort_df(df, samples):
    """
    Sort and reindex values in a meaningful way

    :param df: pandas.core.frame.DataFrame
    :param samples: list(Sample)
    :return:
    """
    # Sort values in a meaningful way
    df.sort_values(by=['group', 'trigger', 'vel', 'note', 'seqPosition'],
                   ascending=[True, True, True, True, True], inplace=True)

    # Optimize data
    for attr in ['group', 'trigger']:
        if attr in df:
            df[attr] = df[attr].astype('category')
    for attr in ['vel', 'note', 'seqPosition']:
        if attr in df:
            df[attr] = df[attr].astype('UInt8')

    # Sort samples using dataframe indices order
    samples = [samples[i] for i in df.index]

    # Reset indices on data frame so everything lines up
    df = df.reset_index(drop=True)

    return df, samples


def query_df(df, fltr=None, attr='note'):
    """
    Multi-filtered query of a given data frame

    :param pandas.core.frame.DataFrame df:
    :param dict or None fltr: Query filter as a dict
    :param str or None attr: Optional attribute to query, return a filtered data frame otherwise
    :return:
    """
    result = df.copy()

    if fltr and isinstance(fltr, dict):
        for k in fltr:
            if k in result:
                result = result[result[k] == fltr[k]]
        if attr:
            return result[attr]
        else:
            return result

    return result


# Auxiliary definitions

def extend_note_range(note, notes, mode='mid', limit=True):
    """
    Extend note range (spread) in relation to a list of notes
    Also used for velocity mapping

    :param int note: Given sample note, MUST be in notes
    :param list notes: List of sample notes
    :param str or None mode: Extension mode 'down', 'up', 'mid', 'none'
    :param bool or list or tuple limit: if False extend bounds to full MIDI range (0-127)
    :return: "loNote", "hiNote", "lim_mn", "lim_mx"
    :rtype: list
    """

    # No extension
    if mode in ['none', 'None', None]:
        return note, note, limit[0], limit[-1]

    idx = notes.index(note)
    mx_idx = len(notes) - 1

    mn = (note, notes[max(idx - 1, 0)] + 1)[note > notes[0]]
    down = [mn, note]

    mx = (note, notes[min(idx + 1, mx_idx)] - 1)[note < notes[- 1]]
    up = [note, mx]

    mid = [(a + b) // 2 for a, b in zip(down, up)]

    result = {'down': down, 'up': up, 'mid': mid}[mode]

    lim_mn, lim_mx = notes[0], notes[-1]
    if limit is False:
        lim_mn, lim_mx = 0, 127
    elif type(limit) in (list, tuple):
        lim_mn, lim_mx = limit[0], limit[-1]

    if note == notes[0] and limit is not True:
        result[0] = min(lim_mn, notes[0])
    if note == notes[-1] and limit is not True:
        result[-1] = max(lim_mx, notes[-1])

    result.extend([lim_mn, lim_mx])

    return result


def get_note_gap(notes, mode='median'):
    """
    Return gap value between notes
    :param list notes:
    :param str mode: 'min', 'max', 'median','avg', or 'bounds'
    :return:
    :rtype: int or list
    """
    diff = np.abs(np.diff(sorted(list(set(notes)))))
    if not len(diff):
        return None

    match mode:
        case 'min':
            return int(min(diff))
        case 'max':
            return int(max(diff))
        case 'median':
            return int(np.median(diff))
        case 'avg':
            return int(np.ceil(np.mean(diff)))
        case 'bounds':
            return [int(diff[0]), int(diff[-1])]


def rr_ofs_from_count(count=3):
    """
    Determine fake RR offsets from given count
    :param int count: Number of items
    :return: List of offsets, preferably an odd number like 3 or 5
    Example: [-1,0,1]
    :rtype: list
    """
    center = count // 2
    result = [i - center for i in range(count)]
    return result


def limit_from_notes(notes, limit='auto'):
    """
    Determine limits (note range) from given notes

    :param list or set notes: List of midi notes as integers (0-127)
    :param list or tuple or str or None limit: Polymorph argument depending on type
    2 integers as list of integers or string with '-' or '+' for offsets
    'auto' will determine the limits from note gaps
    reasonable guess when the range of the instrument is not known

    :return: Bounding notes
    :rtype: list
    """
    smp_limit = [min(notes), max(notes)]
    mn_note_smp, mx_note_smp = smp_limit

    mn_mx_note = []
    if type(limit) in (list, tuple):
        if len(limit) == 2:
            for value, smp_value in zip(limit, smp_limit):
                if type(value) is str:
                    # Offset bounds from min/max sample notes
                    if '-' in value or '+' in value:
                        mn_mx_note.append(smp_value + int(value))
                    else:
                        mn_mx_note.append(int(value))
                else:
                    mn_mx_note.append(value)
    elif isinstance(limit, str) and limit.lower().startswith('auto'):
        # Determine min/max from notes gaps
        gap = get_note_gap(notes=list(notes), mode='avg')
        if gap is None:
            mn_mx_note = [0, 127]
        else:
            if limit.lower().startswith('autox') and limit[5:].isdigit():
                gap *= int(limit[5:])
            mn_mx_note = [mn_note_smp - gap, mx_note_smp + gap]
    else:
        mn_mx_note = smp_limit

    mn_mx_note = min(mn_note_smp, mn_mx_note[0]), max(mx_note_smp, mn_mx_note[1])

    return [clamp(v, 0, 127) for v in mn_mx_note]


def get_pad_values(ra, rb):
    """
    Get items from rb which outside the min/max range of ra
    :param list ra:
    :param list rb:
    :return:
    :rtype: list
    """
    pad = [v for v in rb if v < min(ra) or v > max(ra)]
    return pad


def mean_pf(pf_values, mode='blend', x=1.0):
    """
    Compute the mean of the pitch fractions
    If the standard deviation is too high, a separate mean is computed for positive and negative values

    :param np.array pf_values: Array of pitch fraction values
    :param str mode: 'blend' blend mean with original values or 'scl' scale standard deviation to given value
    :param float x: Blend/scale factor

    :return:
    :rtype: np.array
    """
    result = pf_values.copy()
    mean = np.mean(pf_values)
    std = np.std(pf_values)
    std_type = type(std)

    tol = 1e-3

    if std > 25:
        idx = pf_values - mean < 0
        result[idx] = np.mean(pf_values[idx])
        result[-idx] = np.mean(pf_values[-idx])
        std = pf_values.copy()
        std[idx] = np.std(pf_values[idx])
        std[-idx] = np.std(pf_values[-idx])
    else:
        result = mean

    if mode == 'scl':
        if isinstance(std, std_type):
            if std > tol:
                x = 1 - (x / std)
        else:
            x = np.zeros(len(std)) + x
            std_idx = std > tol
            x[std_idx] = 1 - (x[std_idx] / std[std_idx])

    result = lerp(result, pf_values, x)
    return result


# Unused / deprecated

def offset_note_range(lo, hi, mn, mx, offset=0):
    """
    Offset a note range with respect for min/max range

    :param int lo: Input min value for the range
    :param int hi: Input max value for the range

    :param int mn: Overall output min value
    :param int mx: Overall output max value

    :param int offset: Given offset

    :return: lo_note, hi_note
    :rtype: list
    """
    # Make bounding values "stick" to bounds regardless of the offset
    # So we don't create unmapped notes in the process
    if lo != mn or offset < 0:
        lo += offset
    if hi != mx or offset > 0:
        hi += offset

    if (lo > mx and hi > mx) or (lo < mn and hi < mn):
        return None

    return [clamp(lo, mn, mx), clamp(hi, mn, mx)]
