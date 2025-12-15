# coding:utf-8
"""
    :module: dspreset_to_sfz.py
    :description: dspreset to SFZ converter
    Designed to match dspreset files produced by SMP2ds as much as possible while remaining as simple as possible

    :author: Michel 'Mitch' Pecqueur
    Thanks to kinwie, that_sfz_guy, plgDavid and DSmolken from sfz discord server
    for the insights, examples and help about sfz opcodes

    :date: 2025.07
"""

import re
import sys
import traceback
import xml.etree.ElementTree as Et
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf

from common_audio_utils import lin_to_db
from common_ui_utils import beautify_str, shorten_str

__version__ = '1.0.2'


def dspreset_to_sfz(input_file: Path | str | None,
                    xml_tree: Et.Element | None = None,
                    output_file: Path | str | None = None,

                    group_naming: str = 'keep',

                    use_eg: bool | int = True,
                    adsr: tuple[float, float, float, float] = (0.001, 10, 0, .5),
                    dr_factor: float = 2.0,

                    attenuation: float | None = -9,
                    ds_sfz_att: float | None = -2.27,

                    vel_track: float = 1.0,
                    match_ds_vel: bool = True,

                    release_off_by_attack: bool = False,

                    multi_out: bool | None = True,

                    bg_img: Path | str | None = None, bg_ctrl: Path | str | None = None,

                    reverb: bool = False,

                    engine: str | None = None

                    ) -> Path | None:
    """
    dspreset to SFZ converter

    Designed to match dspreset files produced by SMP2ds as much as possible while remaining as simple as possible

    Tested with Sfizz 1.2.2, Sfizz 1.2.3 exhibits hanging notes at the moment
    Also tested with Sforzando 1.981

    :param input_file: Input dspreset file to convert

    :param xml_tree: dspreset as a xml element
    :param output_file: override sfz file name, required when using xml_tree

    :param use_eg: Use envelope generator (egN_ampgeg) instead of SFZ default envelope (True or 2)
    When set to 1, EG will be used except for release triggers

    :param adsr: Default adsr values
    :param dr_factor: Compensation factor for decay/release to make up for shorter perceived duration with SFZ
    NOTE: only has effect on regular SFZ1 decay/release and no effect over EG envelope

    :param group_naming: 'keep', 'beautify', 'upper', 'lower','shorten'

    :param attenuation: in dB, this is used as a workaround to prevent DS output soft-limiting to kick-in
    thus maintaining a linear response and avoiding distortion induced by DS limiter

    :param ds_sfz_att: DS to SFZ volume difference, typically -2.27 dB (tested with Sfizz only)

    :param vel_track: Default amplitude velocity tracking 0-1 when none is found in source dspreset
    :param match_ds_vel: Apply linear mapping to velocity instead of default quadratic fake exp

    :param release_off_by_attack: Spare polyphony by choking a release group by its corresponding attack group
    May be a bit aggressive for some instruments

    :param multi_out: Enable/Disable multi-out, disabled with engine='sforzando'

    :param bg_img: Optional background image path, ignored by Sforzando
    :param bg_ctrl: Optional control image path (Sfizz only)

    :param reverb: Enable fverb (Sfizz only) not possible to modulate at the moment

    :param engine:
    'generic' or None - recommended default
    'sforzando' - disable unsupported opcodes such as 'output' (only output 0 is heard)
    'loop_crossfade' and 'sample_fadeout'
    'sfizz' - EG last point hack to terminate release and avoid polyphony crash, not recommended

    Enable or disable opcodes or hacks depending on target engine

    :return: Path to resulting sfz file
    """

    if input_file:
        p = Path(input_file)
        tree = Et.parse(str(input_file))
        root = tree.getroot()
        if not output_file:
            output_file = p.with_suffix('.sfz')
    elif xml_tree and output_file:
        root = xml_tree
        p = Path(output_file)
    else:
        return None

    root_dir = p.parent

    grp_idx = 0
    max_f = 22000  # Max frequency for cutoff filter
    sr = 48000

    tune_cc = 9
    vel_trk_cc = 16
    rls_vol_cc = 17

    # Get parent of each element in the tree
    parent_map = {child: item for item in root.iter() for child in item}

    # Gather some sample info
    samples = list({item.get('path', None): None for item in root.iter('sample')}.keys())
    smp_info = {p: sf.info(root_dir / Path(p)) for p in samples}
    smp_rate = {p: smp_info[p].samplerate for p in samples}
    sr = int(np.median(list(smp_rate.values()))) or sr

    # Get control values
    ctrl_value = {'{}.{}'.format(binding.get('position', item.get('parameterName')),
                                 binding.get('parameter', None)): item.get('value', None)
                  for item in root.iter('control') for binding in item.iter('binding')}

    # Get vel track value from controls
    veltrk = [v for k, v in ctrl_value.items() if k.endswith('AMP_VEL_TRACK')]
    if veltrk:
        vel_track = float(veltrk[0])

    # Get attenuation from gain effect
    att = [fx.get('level', None) for item in root.iter('effects') for fx in item.iter('effect') if
           fx.get('type', None) == 'gain'] or None
    if att:
        attenuation = float(att[0])

    total_att = attenuation + ds_sfz_att

    # Get labeled Knob values
    knob_value = {b.get('parameter', None): item.get('value', None)
                  for item in root.iter('labeled-knob') for b in item.iter('binding')}

    # Get ADSR values from controls
    keys = ['attack', 'decay', 'sustain', 'release']
    params = ['ENV_' + key.upper() for key in keys]
    adsr = tuple(knob_value.get(k, v) for k, v in zip(params, adsr))

    # Gather group info
    group_to_idx = {item.get('name', i): i for i, item in enumerate(root.iter('group'))}
    idx_to_group = {v: k for k, v in group_to_idx.items()}

    group_tags = [item.get('tags', '').split(',') for item in root.iter('group')]
    group_tags = dict(zip(group_to_idx.keys(), group_tags))

    tag_groups = {}
    for grp, tags in group_tags.items():
        for tag in tags:
            idx = group_to_idx[grp]
            if tag not in tag_groups:
                tag_groups[tag] = [idx]
            else:
                tag_groups[tag].append(idx)

    # Triggers
    group_triggers = {i: item.get('trigger', None) for i, item in enumerate(root.iter('group'))}
    has_release = 'release' in group_triggers.values()

    # Group mute
    group_mute = {}
    for item in root.iter('button'):
        val = item.get('value', None)
        if val is not None:
            for binding in item.iter('binding'):
                lvl = binding.get('level', None)
                prm = binding.get('parameter', None)
                if lvl == 'group' and prm == 'ENABLED':
                    pos = binding.get('position', None)
                    if pos is not None:
                        group_mute[int(pos)] = int(val)

    # Group Volume CCs
    start_cc = 102
    slider_idx = 0
    ctrl_group_cc_str = ''
    group_cc_str = {}
    bind_id_mute = {}

    grp_naming_func = {'beautify': beautify_str, 'upper': lambda x: x.upper(), 'lower': lambda x: x.lower(),
                       'shorten': shorten_str, 'keep': lambda x: x}

    for item in root.iter('control'):
        for binding in item.iter('binding'):
            prm = binding.get('parameter', None)
            lvl = binding.get('level', None)
            if prm == 'AMP_VOLUME' and (lvl == 'tag' or lvl == 'group'):
                bind_id = binding.get('identifier', None)

                # Skip expression control
                if bind_id == 'expression':
                    continue

                pos = binding.get('position', None)
                match lvl:
                    case 'tag':
                        grps = tag_groups.get(bind_id)
                    case _:
                        grps = ([int(pos)], [])[pos is None]

                # Exclude release control
                add_cc = list(set(True for g in grps if group_triggers[g] != 'release')) or [False]

                if add_cc[0]:
                    grp_vol_cc = start_cc + slider_idx

                    match lvl:
                        case 'tag':
                            group_name = bind_id
                            cc_value = 127
                        case _:
                            prm_name = item.get('parameterName', None)
                            group_name = item.get('label', None) or prm_name
                            # Simplify string so it's not too long
                            group_name = re.sub(rf'\b{re.escape('volume')}\b', '', group_name,
                                                flags=re.IGNORECASE).strip()
                            cc_value = ctrl_value.get(f'{pos}.AMP_VOLUME', None)
                            if cc_value is None:
                                cc_value = 127
                            else:
                                cc_value = int(round(float(cc_value) * 127))

                    cc_label = grp_naming_func[group_naming](group_name)

                    # Resolve mute value per bind_id
                    bind_id_mute[bind_id] = (list(set(v for k, v in group_mute.items() if k in grps)) or [None])[0]
                    mute_value = bind_id_mute.get(bind_id, None)
                    if mute_value is not None:
                        cc_value *= mute_value

                    ctrl_group_cc_str += f'label_cc{grp_vol_cc}={cc_label} set_cc{grp_vol_cc}={int(cc_value)}\n'
                    for g in grps:
                        group_cc_str[g] = (f'amplitude_oncc{grp_vol_cc}=100 amplitude_curvecc{grp_vol_cc}=0'
                                           f' // Group Volume\n')
                    slider_idx += 1
            else:
                continue

    def silenced_to_off(grp_str: str, tag_str: str) -> int:
        """
        Resolve silencedByTags to off_by
        silencedByTags (DS) accepts multiple tags however off_by (sfz) only accepts a single index
        This attempts to return the most adequate index and by off-loading self-choke to polyphony
        :param grp_str: group string
        :param tag_str: tag string
        :return:
        """
        indices = [tag_groups[t] for t in tag_str.split(',')]
        indices = [a for aa in indices for a in aa]  # Flatten list
        off_by_value = None

        # Only one tag, return it directly
        if len(indices) == 1:
            off_by_value = indices[0]

        # Multiple tags
        elif len(indices) > 1:
            g_idx = group_to_idx[grp_str]

            # Self silencing, use group polyphony to deal with it
            # NOTE: note_polyphony does not seem as efficient or does not seem to work with Sfizz
            if g_idx in indices:
                xtra_opcodes['polyphony'] = 1
                # xtra_opcodes['note_polyphony'] = 1

            # Pick group index with the closest tag (except for itself), so it's at least somewhat related
            rmax = 0
            for i in indices:
                r = SequenceMatcher(a=grp_str, b=idx_to_group[i]).ratio()
                if r > rmax and i != g_idx:
                    rmax = r
                    off_by_value = i

        return off_by_value

    # element to header translation table
    elem_to_header = {
        'DecentSampler': 'global',
        'groups': 'master',
        'group': 'group',
        'sample': 'region',
        'effect': None
    }

    # attrib to opcode translation table
    attrib_to_opcode = {
        'name': 'group',

        'silencingMode': 'off_mode',
        'silencedByTags': 'off_by',

        'volume': 'volume',
        'ampVelTrack': 'amp_veltrack',

        'trigger': 'trigger',

        'attack': 'ampeg_attack',
        'decay': 'ampeg_decay',
        'sustain': 'ampeg_sustain',
        'release': 'ampeg_release',

        'attackCurve': None,
        'decayCurve': None,
        'releaseCurve': None,

        'path': 'sample',

        'pitchKeyTrack': 'pitch_keytrack',

        'rootNote': 'pitch_keycenter',
        'loNote': 'lokey',
        'hiNote': 'hikey',

        'tuning': 'tune',

        'loVel': 'lovel',
        'hiVel': 'hivel',

        'seqLength': 'seq_length',
        'seqPosition': 'seq_position',

        'start': 'offset',
        'end': 'end',

        'ampEnvEnabled': 'loop_mode',

        'loopEnabled': 'loop_mode',
        'loopStart': 'loopstart',
        'loopEnd': 'loopend',

        'loopCrossfade': 'loop_crossfade',

        'pan': 'pan',

        'output2Target': 'output'
    }

    # attrib to opcode value conversion functions
    opcode_func = {
        'name': lambda v: grp_idx,  # Group name to group index

        'silencedByTags': lambda v: silenced_to_off(idx_to_group[grp_idx], v),  # tags to group index

        'volume': lambda v: volume_str_to_float(v, total_att),  # str to dB

        'ampVelTrack': lambda v: round(float(v) * 100, 2),  # 0-1 to %

        'pitchKeyTrack': lambda v: round(float(v) * 100, 3),  # semitone to semitone cents
        'tuning': lambda v: round(float(v) * 100, 3),  # semitone to semitone cents

        'ampEnvEnabled': lambda v: {'false': 'one_shot'}.get(v.lower(), None),  # One shot, typically drums
        'loopEnabled': lambda v: {'0': 'no_loop', '1': 'loop_continuous',
                                  'false': 'no_loop', 'true': 'loop_continuous'}.get(v.lower(), None),

        'attack': lambda v: round(float(v), 4),
        'decay': lambda v: round(float(v) * dr_factor, 4),  # Compensate duration
        'sustain': lambda v: round(float(v) * 100, 2),  # 0-1 to %
        'release': lambda v: round(float(v) * dr_factor, 4),  # Compensate duration

        'loopCrossfade': lambda v: round(float(v) / smp_rate.get(src_attr.get('path', ''), sr), 4),
        # samples to seconds

        'output2Target': lambda v: int(v.removeprefix('AUX_STEREO_OUTPUT_')) - 1  # Start channel number 1 to 0
    }

    # -- Generate sfz file --
    result = f'// {p.stem}\n'
    result += f'// Converted by {Path(__file__).stem} {__version__}\n\n'

    # Default CCs
    result += f'<control> label_cc1=Tone label_cc7=Volume label_cc{tune_cc}=Tune label_cc10=Pan label_cc11=Expression\n'
    result += '// Bipolar CCs are set to 63.5 for proper centering\n'
    result += f'set_cc1=127 set_cc7=127 set_cc{tune_cc}=63.5 set_cc10=63.5 set_cc11=127\n'

    result += ctrl_group_cc_str

    # amp_veltrack_oncc does not seem to do anything with Sfizz (ARIA extension)
    # result += f'label_cc{vel_trk_cc}=Vel Track set_cc{vel_trk_cc}={int(vel_track * 127)}\n'
    # result += f'amp_veltrack_oncc{vel_trk_cc}=100\n'

    if has_release:
        result += f'label_cc{rls_vol_cc}=Release Volume set_cc{rls_vol_cc}=127\n'

    # Whereas it is listed sfzformat.com, modulating wet level with a cc does not work yet
    # 'effect' opcodes are considered engine specific
    if reverb:
        result += f'label_cc91=Reverb set_cc91=127\n'
        result += (f'<effect> type=fverb reverb_type=mid_hall reverb_input=100 reverb_dry=100 '
                   f'reverb_size=80 reverb_damp=75 reverb_wet=20 reverb_wet_oncc91=0\n')

    # Background images, optimized for Sfizz (775x335)
    if bg_img and engine != 'sforzando':
        result += f'image={bg_img}\n'
        result += f'image_controls={bg_ctrl or bg_img}\n'

    # - Main loop -

    result += '\n'
    group_eg_str = {}
    env_duration = {}

    for elem in root.iter():
        if elem.tag in elem_to_header:
            header = elem_to_header.get(elem.tag)

            res = ''
            src_attr = {}  # DS attributes which will be translated and converted
            xtra_opcodes = {}  # SFZ opcodes directly used without conversion

            # Default ASDR values
            adsr_attr = ['attack', 'decay', 'sustain', 'release']
            adr_crv_attr = ['attackCurve', 'decayCurve', 'releaseCurve']

            if header == 'group':
                src_attr = dict(zip(adsr_attr, list(adsr)))

            # Parse attribute values
            parsed_attr = {k: v for k, v in elem.attrib.items()}
            src_attr.update(parsed_attr)

            # Remove envelope attributes if envelope is explicitly disabled
            if parsed_attr.get('ampEnvEnabled', None) == 'false':
                amp_env_enabled = False
                for attr in adsr_attr + adr_crv_attr:
                    src_attr.pop(attr, None)
            else:
                amp_env_enabled = True

            if header == 'group':
                grp_idx = list(root.iter('group')).index(elem)
                grp_name = src_attr.get('name', grp_idx)
                res += f'\n// {grp_name}\n'  # Add comment with group name
                grp_trg = src_attr.get('trigger', None)

                # Envelope duration
                if grp_idx not in env_duration:
                    adsr_values = [eval(str(src_attr.get(attr, 0))) for attr in adsr_attr]
                    skip_idx = ([2], [1, 2])[grp_trg == 'release']
                    env_duration[grp_idx] = sum([float(v) for i, v in enumerate(adsr_values) if i not in skip_idx])

                # EG used as replacement for default SFZ1 envelopes
                if amp_env_enabled:
                    if use_eg is True or use_eg == 2 or (use_eg == 1 and grp_trg != 'release'):
                        adsr_values = [eval(str(src_attr.pop(attr, None))) for attr in adsr_attr]
                        adr_curve = [eval(str(src_attr.pop(attr, None))) for attr in adr_crv_attr]
                        group_eg_str[grp_idx] = adsr_to_eg_str(adsr=adsr_values, adr_curve=adr_curve, eg_idx=1,
                                                               trigger=grp_trg, engine=engine)

                # Kill releases with attacks to spare polyphony
                if release_off_by_attack and grp_trg == 'release' and 'silencedByTags' not in src_attr:
                    # Find corresponding attack group (group with attack trigger with previous index)
                    attack_groups = [k for k, v in group_triggers.items() if v == 'attack'] or []
                    off_by_idx = [ag for ag in attack_groups if ag < grp_idx]
                    if off_by_idx:
                        xtra_opcodes['off_by'] = off_by_idx[-1]

                if 'pan' not in src_attr:
                    pan = ctrl_value.get(f'{grp_idx}.PAN', 0)
                    if pan:
                        src_attr['pan'] = pan

                if 'tuning' not in src_attr:
                    tuning = ctrl_value.get(f'{grp_idx}.GROUP_TUNING', 0)
                    if tuning:
                        src_attr['tuning'] = tuning

            # Opcodes filtering

            # Sforzando does not seem to support multi-out
            if multi_out is False or engine == 'sforzando':
                src_attr.pop('output2Target', None)

            # Sforzando does not support loop cross-fade
            if engine == 'sforzando':
                src_attr.pop('loopCrossfade', None)

            # + Translate and convert attributes +
            opcodes = {attrib_to_opcode.get(k, None): opcode_func.get(k, lambda x: x)(v) for k, v in src_attr.items() if
                       attrib_to_opcode.get(k, None) is not None}

            # + Post conversion tweaks +

            # Global options
            if header == 'global':
                opcodes['amp_veltrack'] = vel_track * 100  # 0-1 to %
                if vel_track > 0 and match_ds_vel:
                    # Emulate DS linear velocity, SFZ uses an approximate exp velocity by default
                    opcodes['amp_velcurve_0'] = 0
                    opcodes['amp_velcurve_127'] = 1

                # Set CC11 for Sforzando, Sfizz has it by default
                if engine == 'sforzando':
                    opcodes['amplitude_oncc11'] = 100

                opcodes['cutoff'] = max_f / 256  # max_f = cutoff * 2 ** (9600/1200)
                opcodes['cutoff_cc1'] = 9600  # 8 octaves, max value allowed by this opcode (semitone cents)
                opcodes['resonance'] = 0  # Default

                opcodes['pitch_oncc9'] = 3600  # 3 octaves in both directions (semitone cents)
                opcodes['pitch_curvecc9'] = 1  # Bipolar predefined curve

                opcodes['note_polyphony'] = 2
                opcodes['note_selfmask'] = 'off'

            # Replicate group LPF - typically used for fake releases
            if elem.tag == 'effect' and src_attr.get('type', None) == 'lowpass':
                parent = parent_map.get(parent_map.get(elem, {}), None)
                if parent is not None and parent != root:
                    opcodes['cutoff2'] = src_attr.get('frequency', sr / 2)

            if header == 'group':
                if group_triggers[grp_idx] == 'release':
                    # Release triggered only when no key is pressed
                    # (Kontakt "Unisono - Portamento" release last emulation)
                    off_by = opcodes.get('off_by', None)
                    off_trg = group_triggers.get(off_by, None)
                    has_legato = (False, True)[off_trg == 'legato']
                    # Sforzando only, does not work with Sfizz
                    if has_legato and engine == 'sforzando':
                        opcodes['lohdcc153'] = 0
                        opcodes['hihdcc153'] = 1.1

            # Release trigger polyphony fix - essentially for Sfizz
            if header == 'region' and engine != 'sforzando':
                disable_loop = False
                smp_path = src_attr.get('path', '')
                smp_sr = smp_rate.get(smp_path, sr)
                smp_start = int(opcodes.get('offset', 0)) / smp_sr

                if group_triggers[grp_idx] == 'release':
                    # Trim end of samples for release to spare polyphony
                    # TODO: trim at closest quiet sample to reduce potential popping
                    if env_duration[grp_idx] < smp_info[smp_path].duration - smp_start:
                        opcodes['end'] = int(round((smp_start + env_duration[grp_idx]) * smp_sr))
                        disable_loop = True
                    else:
                        # Fix for potential edge case when release samples are looped
                        # and envelope duration is longer than total sample duration
                        if opcodes.get('loop', 'no_loop') == 'loop_continuous':
                            loop_duration = (opcodes['loopend'] - opcodes['loopstart']) / sr
                            # Limit loop_count to envelope duration without looping forever
                            loop_count = int(np.ceil(env_duration[grp_idx] / loop_duration)) - 1
                            if loop_count > 0:
                                opcodes['loop_count'] = loop_count
                            else:
                                disable_loop = True

                    if disable_loop:
                        opcodes['loop_mode'] = 'no_loop'
                        for oc in ['loopstart', 'loopend', 'loop_crossfade']:
                            opcodes.pop(oc, None)
                        # sample_fadeout currently unsupported by both sfizz and sforzando
                        opcodes['sample_fadeout'] = round(env_duration[grp_idx] / 9, 3)

            opcodes.update(xtra_opcodes)

            # Write header and opcodes line by line
            if header is not None:
                res += f'<{header}> '
            if opcodes:
                attr_str = ' '.join(f'{k}={v}' for k, v in opcodes.items())
                res += f'{attr_str}\n'
            if header == 'group':
                res += group_cc_str.get(grp_idx, '')
                res += group_eg_str.get(grp_idx, '')

                if has_release and src_attr.get('trigger', '') == 'release':
                    res += (f'amplitude_cc{rls_vol_cc}=100 amplitude_curvecc{rls_vol_cc}=0'
                            f' // Release Volume\n')

            result += res

    # Write resulting file
    with open(output_file, 'w') as f:
        f.write(result)

    return output_file


# Auxiliary defs

def volume_str_to_float(vol_str: str, attenuation: float) -> float:
    """
    Convert volume string in float
    :param vol_str: dB or amplitude (0-1)
    :param attenuation: in dB
    :return: dB float value
    """
    if vol_str.endswith('dB'):
        vol = float(vol_str.rstrip('dB')) + attenuation
    else:
        vol = lin_to_db(float(vol_str)) + attenuation
    return round(vol, 2)


def adsr_to_eg_str(adsr: list = (0.001, 2, 0, .25),
                   adr_curve: list = (None, None, None),
                   trigger=None,
                   eg_idx: int | None = None,
                   engine: str | None = None) -> str:
    """
    Create envelope generator string from adsr and adr_curve values
    This attempts to emulate DS envelopes with curve shaping

    Might help limiting polyphony usage (no double decay/release times)
    NOTE: At the moment, Sfizz does not end voices as early as it could
    when using egN_ampeg in combination with trigger=release

    :param adsr: DS adsr values
    :param adr_curve: DS adr curve values
    :param trigger: Special behavior with trigger=release
    :param eg_idx: Envelope generator index
    :param engine: 'sfizz'

    :return:
    """
    adsr_values = [(v, d)[v is None] for v, d in zip(adsr, [0.001, 2, 1, .25])]
    eg_time = [0] + [v for i, v in enumerate(adsr_values) if i != 2]
    sustain = round(float(adsr_values[2]), 3)
    eg_lvl = [0, 1.0, sustain, 0.0]

    curve = 2  # Not oo far from DS behavior
    eg_shape = [None] + [round(sign * (val, dft)[val is None] / 100 * curve, 2) for val, dft, sign in
                         zip(adr_curve, [-100, 100, 100], [1, -1, -1])]

    eg_name = ['start', 'attack', 'decay', 'release']

    eg_str = f'eg{eg_idx or 1:02d}_ampeg=100\n'

    full_sustain = bool(sustain == 1)

    idx = 0
    for i, (t, lvl, shp, name) in enumerate(zip(eg_time, eg_lvl, eg_shape, eg_name)):
        if trigger == 'release' and name == 'decay':
            continue
        if full_sustain and name == 'decay':
            continue
        eg_str += f'eg{eg_idx or 1:02d}_time{idx}={t:0.4f} eg{eg_idx or 1:02d}_level{idx}={lvl}'
        if shp:
            eg_str += f' eg{eg_idx or 1:02d}_shape{idx}={shp}'
        if (i == 2 - int(sustain) and trigger != 'release') or (trigger == 'release' and name == 'release'):
            eg_str += f' eg{eg_idx or 1:02d}_sustain={idx}'
        eg_str += f' // {name.capitalize()}'
        eg_str += '\n'
        idx += 1

    if trigger == 'release' and engine == 'sfizz':
        # Hack ending release notes earlier in Sfizz
        # Sustain index is set to an undefined point, not guaranteed to work in a future Sfizz version
        # Not working at all in Sforzando, release are muted
        eg_str += f'eg{eg_idx or 1:02d}_sustain={idx} // Sfizz hack\n'

    # eg_str = f'eg{eg_idx or 1:02d}_points={idx - 1} ' + eg_str

    return eg_str


# Other defs

def log_curve(mn: float = 33, mx: float = 22000, steps: int = 11, remap: bool = True) -> str:
    """
    Generate Log curve for SFZ
    """
    x = np.linspace(0, 1, steps)
    y = mn * (mx / mn) ** x  # Straight line in log scale

    if remap:
        y = (y - mn) / (mx - mn)  # Remap between 0 and 1

    xy = zip(np.round(x * 127).astype(int).tolist(), np.round(y, 3).tolist())
    curve = f'<curve> curve_index=7' + ' '.join(f'v{k:03d}={v}' for k, v in xy) + '\n'

    return curve


def main():
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        try:
            result = dspreset_to_sfz(input_file)
            print(str(result.resolve()))
            print('Process complete')
        except Exception as e:
            traceback.print_exc()
            print('Encountered an error - Please check settings')
    else:
        print('No arguments provided')
    input('Press ENTER to continue...')


if __name__ == '__main__':
    main()
