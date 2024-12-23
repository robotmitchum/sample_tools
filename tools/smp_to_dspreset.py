# coding:utf-8
"""
    :module: smp_to_dspreset.py
    :description: Create a Decent Sampler Preset from audio samples
    :author: Michel 'Mitch' Pecqueur
    :date: 2024.05
"""

import getpass
import math
import os
import shutil
import tempfile
import xml.etree.ElementTree as Et
from pathlib import Path
from xml.dom import minidom

import numpy as np

import instrument_utils as iu
from color_utils import hex_to_rgba, rgba_to_hex, basic_background, basic_button, adjust_palette
from common_math_utils import linstep, lerp, clamp
from common_ui_utils import shorten_str, beautify_str
from file_utils import recursive_search, resolve_overwriting
from jsonFile import read_json

__ds_version__ = '1.11.19'
__version__ = '1.3.0'


def create_dspreset(root_dir, smp_subdir='Samples',
                    smp_fmt=('wav', 'flac'), ir_fmt=('wav', 'aif', 'flac'),

                    smp_attrib_cfg='smp_attrib_cfg.json',

                    pattern='{group}_{note}_{trigger}', group_naming='keep', override=False, loop=True,
                    transpose=0, tuning=0, seq_mode='random',
                    pad_vel=False,

                    adsr=(0.001, 0, 1, .25), adr_curve=(None, None, None),

                    fake_release=False, fk_rls_mode='start',
                    fk_rls_volume=-24, fk_rls_tuning=0, fk_rls_cutoff=1000,
                    fk_rls_adsr=(.001, .05, 0, .05), fk_rls_adr_curve=(None, None, None),

                    fake_legato=False,
                    fk_leg_start=.1, fk_leg_a=.1, fk_leg_a_curve=None,

                    note_spread='mid', limit=True, note_limit_mode='shared',
                    rr_offset=None, rr_bounds=True,
                    pf_mode='off', pf_th=5,
                    crossfade_mode='linear', crossfade=.05,

                    attenuation=-6, vel_track=1.0, note_pan=0,

                    monophonic=False,

                    bg_text=None,
                    text_font=('HelveticaNeueThin.otf', 24),

                    color_plt_cfg='plt_cfg/Dark_plt_cfg.json', plt_adjust=(0, 1, 1),
                    group_knobs_rows=1, no_solo_grp_knob=True,
                    adsr_knobs=True, max_adsr_knobs=10,
                    use_reverb=True, reverb_wet=0.2, ir_subdir='IR',

                    add_suffix='', auto_increment=True,

                    progress=None):
    """
    Create a Decent Sampler Preset from audio samples

    :param tuple(str,float) text_font: font path, font size
    :param str root_dir: Instrument root directory
    :param str smp_subdir: Samples subdirectory name
    :param str or None ir_subdir: Impulse Response subdirectory name

    :param list or tuple smp_fmt: Accepted sample format extensions
    :param list or tuple ir_fmt: Accepted IR file format extensions

    :param str smp_attrib_cfg: Path to a json file holding sample attribute configuration
    :param str color_plt_cfg: Path to a json file holding color configuration used to generate UI

    :param str pattern: Pattern used to figure the mapping
    :param str group_naming: 'keep', 'beautify', 'upper', 'lower','shorten'
    :param bool override: If True, info from sample name override metadata
    :param bool loop: Use sample loop

    :param int transpose: Transpose the whole mapping
    :param float tuning: Tuning adjustment at group level
    :param str seq_mode: Round-Robin mode - 'round_robin','random', 'true_random' or 'always' (off)

    :param bool pad_vel: Re-use/duplicate note samples for velocity layers with fewer note samples

    :param list or tuple adsr: ADSR envelope (in s, except sustain)
    :param list or tuple adr_curve: ADR Curve (-100 log, 0 lin, 100 exp)

    :param bool fake_release: Create fake release from attack group and samples
    :param str fk_rls_mode: 'start' 'loop_end' 'cue'
    :param float fk_rls_volume: Release attenuation
    :param fk_rls_tuning: Release pitch shifting
    :param fk_rls_cutoff: Low pass filter value in Hz
    :param fk_rls_adsr: ADSR envelope of release
    :param fk_rls_adr_curve: ADR Curve of release

    :param bool fake_legato: Create fake legato from attack group and samples
    :param fk_leg_start: Start of legato sample in s
    :param fk_leg_a: Attack of legato envelope
    :param fk_leg_a_curve: Attack curve of legato envelope

    :param str note_spread: 'up' 'mid' 'down' or None
    :param list or tuple or bool limit: Limit note range to bounding samples or extend bounds
    Extended bound can be given as a list
    :param str note_limit_mode: 'shared' or 'group'

    :param list or tuple rr_offset: Fake Round-Robin offsets
    :param bool rr_bounds: Steal notes further to make fake RR work on bounding notes

    :param str pf_mode: Pitch fraction mode, 'off', 'on', 'on_rand', 'on_threshold', 'on_threshold_rand'
    :param float pf_th: Pitch fraction threshold/random value

    :param str crossfade_mode: 'linear' 'equal_power' or 'off'
    :param float crossfade: Percent of loop length

    :param float attenuation: Volume attenuation to prevent saturation/clipping
    :param float vel_track: Velocity Tracking Amplitude
    :param float note_pan: Auto pan sample from left to right depending on note number
    :param bool monophonic: Set instrument to monophonic

    :param str or None bg_text: Override text written on background image otherwise use directory name
    :param int group_knobs_rows: Number of rows for group knobs
    :param bool no_solo_grp_knob: Do not generate any group knob if there is only one
    :param bool adsr_knobs: Add ASDR knobs to UI
    :param float max_adsr_knobs: Max length in s for ASDR knobs
    :param list or tuple plt_adjust: Global Hue Saturation Value adjustment of color palette

    :param bool use_reverb: Add reverb knob to UI
    :param float reverb_wet: Default reverb wet level

    :param str add_suffix: Add suffix to created file
    :param bool auto_increment: Increment file to avoid overwriting

    :param QProgressBar or None progress: Optional progress bar widget

    :return: Created file
    :rtype: str
    """

    # - Retrieve instrument samples -

    # Sample attributes Config

    # List of sample attributes offered by Decent Sampler and *hopefully* supported by this tool
    # Attribute values can be provided in the sample name (with some limitations depending on how the name is formatted)
    # or using ID3 tags (flac only at the moment, it should support any attribute, but I didn't test everything...)

    num_attrs, ds_smp_attrib = ['vel', 'note', 'seqPosition'], []
    smp_attrib_data = read_json(smp_attrib_cfg, ordered=True) or {}
    if 'num_attrib' in smp_attrib_data:
        num_attrs = smp_attrib_data['num_attrib']
    if 'ds_smp_attrib' in smp_attrib_data:
        ds_smp_attrib = smp_attrib_data['ds_smp_attrib']

    # importlib.reload(iu)
    exclude = ('backup_', 'ignore')
    instr = iu.Instrument(root_dir=root_dir, smp_subdir=smp_subdir, smp_fmt=smp_fmt, exclude=exclude,
                          pattern=pattern, transpose=transpose,
                          override=override, extra_tags=ds_smp_attrib, num_attrib=num_attrs)

    if instr.df is None:
        return False

    if pad_vel:
        instr.pad_vel()  # Modifies number of samples
    smp_count = len(instr.samples)

    instr.set_limit(limit=limit, mode=note_limit_mode)
    instr.pitch_fraction(mode=pf_mode, value=pf_th, seed='', apply=True)  # Modifies pitch fraction

    enable_release = any(['release' in t for t in instr.df['trigger'].unique()]) or fake_release
    if group_knobs_rows:
        add_grp_knobs = len(instr.groups) > 1 or not no_solo_grp_knob
    else:
        add_grp_knobs = False

    # IR Samples
    ir_samples = []
    if ir_subdir:
        ir_samples = recursive_search(root_dir=Path.joinpath(Path(root_dir), ir_subdir), input_ext=ir_fmt,
                                      exclude=exclude, relpath=root_dir)

    # Color Palette Config
    plt_data = read_json(color_plt_cfg, ordered=True) or {}
    if plt_adjust:
        plt_data = adjust_palette(plt_data=plt_data, adjust=plt_adjust)

    text_plt = list(plt_data['textColor'].values())
    track_bg_plt = list(plt_data['trackBackgroundColor'].values())
    ctrl_plt = plt_data['control']
    other_plt = plt_data['other']
    bg_plt = plt_data['backgroundRamp']
    group_plt = list(plt_data['group'].values())

    if 'muteButton' in plt_data:
        mute_btn_plt = list(plt_data['muteButton'].values())
    else:
        mute_btn_plt = [group_plt[0]]

    if 'keyboard' in plt_data:
        keyboard_plt = list(plt_data['keyboard'].values())
    else:
        keyboard_plt = [rgba_to_hex(rgba=[1, .625, .875, 1])]

    if 'backgroundText' in plt_data:
        bg_text_plt = list(plt_data['backgroundText'].values())
    else:
        bg_text_plt = [text_plt[0], '80' + text_plt[0][2:]]

    # - Base UI measurements -
    w, h = 812, 375
    top_band_h = 50  # Top black band height with UI default state
    piano_keys_h = 110  # Bottom keyboard height with UI default state
    ui_h = h - top_band_h - piano_keys_h  # Usable "canvas" height for UI, bottom position
    cx, cy = w // 2, ui_h // 2  # Center of "canvas"

    # Create background image
    bg_dir = Path.joinpath(Path(root_dir), 'Resources')
    if not bg_dir.exists():
        os.makedirs(bg_dir, exist_ok=True)
    bg_path = Path.joinpath(bg_dir, 'bg.jpg')
    bg_colors = [hex_to_rgba(h)[1:] for h in bg_plt.values()]

    bg_path = basic_background(str(bg_path), w=w, h=h, scl=2, overwrite=True, colors=bg_colors, gamma=1.0,
                               text=bg_text, text_xy=(8, top_band_h + 8), text_font=text_font,
                               text_color=np.roll(hex_to_rgba(bg_text_plt[0]), -1).tolist())
    bg_path = Path(bg_path).relative_to(root_dir)

    # Create mute buttons
    btn_paths = []
    btn_w = 16
    if add_grp_knobs:
        for i, plt in enumerate([track_bg_plt[0], mute_btn_plt[0]]):
            rgba = np.array([1, 1, 1, 1]) * np.roll(hex_to_rgba(plt), -1).tolist()
            btn_path = Path.joinpath(bg_dir, f'grp_btn{i}.png')
            btn_path = basic_button(str(btn_path), size=btn_w * 2, rgba=rgba)
            btn_path = Path(btn_path).relative_to(root_dir)
            btn_paths.append(str(btn_path))

    # Round-Robin offsets (Fake Round-Robin)
    rr_offset = rr_offset or [0]

    # - Group slider labels -
    grp_naming_func = {'beautify': beautify_str, 'upper': lambda x: x.upper(), 'lower': lambda x: x.lower(),
                       'shorten': shorten_str, 'keep': lambda x: x}

    grp_label = {g: grp_naming_func[group_naming](g) for g in instr.groups}

    # Trigger tags
    trig_tag_dict = {'attack': 'trig_atk', 'release': 'trig_rls', 'first': 'trig_1st', 'legato': 'trig_leg'}

    # - Root XML level -
    decentsampler = Et.Element('DecentSampler', attrib={'minVersion': __ds_version__})
    decentsampler.append(Et.Comment(f'Generated by {Path(__file__).stem} {__version__}'))
    tags = Et.SubElement(decentsampler, 'tags')
    effects = Et.SubElement(decentsampler, 'effects')
    midi = Et.SubElement(decentsampler, 'midi')

    # - Instrument UI -
    ui = Et.SubElement(decentsampler, 'ui',
                       attrib={'width': str(w), 'height': str(h), 'bgImage': str(bg_path.as_posix()),
                               'layoutMode': 'relative', 'bgColor': "00000000"})
    tab = Et.SubElement(ui, 'tab', attrib={'name': 'main'})

    # Information / Credits
    info_text, info_tooltip = 'Info', ''

    info_filepath = Path(root_dir).joinpath('INFO.txt')
    if info_filepath.exists():
        with open(info_filepath, mode='r', encoding='utf-8') as fr:
            for line in fr.readlines():
                line = line.strip()
                if line:
                    info_tooltip += f'{line}\n'
        info_tooltip += '\n'
    else:
        info_tooltip = f'Samplist - {beautify_str(getpass.getuser())}\n'
        info_tooltip += f"UI - {beautify_str(Path(__file__).stem)} {__version__}\n\n"

    info_ts = 24
    info_w = (info_ts * len(info_text)) // 2
    Et.SubElement(tab, 'label',
                  attrib={'text': info_text, 'x': str(w - info_w - 8), 'y': str(8), 'width': str(info_w),
                          'height': str(info_ts), 'textColor': bg_text_plt[-1], 'textSize': str(info_ts),
                          'tooltip': info_tooltip})

    # Auto RR offset
    if isinstance(rr_offset, int):
        rr_offset = iu.rr_ofs_from_count(count=clamp(rr_offset, 1, 7))

    if len(rr_offset) > 1:
        instr.set_zones()

    print(f'- {instr.root_dir} : {instr.name} - \n')
    print(f'RR Offsets: {rr_offset}')
    print(f'Note limits: {instr.limit}')

    # - Keyboard color -

    keyboard = Et.SubElement(ui, 'keyboard')

    # Add some headroom to avoid saturation
    Et.SubElement(effects, 'effect', attrib={'type': 'gain', 'level': f'{attenuation}'})

    # - Default controls -

    mx_len = np.median([len(lbl) for lbl in grp_label.values()])  # Max Word Length
    kw = 96  # Knob Width
    spc = 80  # Group knob width/height
    ts = round(min((spc / mx_len) * 3, spc / 2))  # Text size
    margin = 120  # Margin for group sliders from canvas left border

    # - "Expression" (Volume) -
    y = cy - 160 / 2
    ctrl = Et.SubElement(tab, 'control',
                         attrib={'x': '8', 'y': str(y), 'width': '48', 'height': '160',
                                 'parameterName': '<>', 'style': 'linear_vertical',
                                 'showLabel': 'true', 'textColor': ctrl_plt['expression'], 'textSize': '40',
                                 'trackForegroundColor': ctrl_plt['expression'],
                                 'trackBackgroundColor': track_bg_plt[0],
                                 'tooltip': 'Expression',
                                 'type': 'float', 'minValue': '0', 'maxValue': '1', 'value': '1',
                                 'defaultValue': '1'})
    Et.SubElement(ctrl, 'binding',
                  attrib={'type': 'amp', 'level': 'tag', 'identifier': 'expression', 'parameter': 'AMP_VOLUME'})

    # Link to CC 11
    cc = Et.SubElement(midi, 'cc', attrib={'number': '11'})
    Et.SubElement(cc, 'binding',
                  attrib={'level': 'ui', 'type': 'control', 'parameter': 'VALUE', 'position': '1',
                          'translation': 'linear', 'translationOutputMin': "0", 'translationOutputMax': "1"})

    # - "Modulation" (LP Filter) -
    steps, k = 11, 3
    x = np.linspace(0, 1, steps)
    y = np.exp(lerp(math.log(33), math.log(22000), x))  # Straight line in log scale

    table = [f'{round(float(x[i]), 3)},{int(y[i])}' for i in range(steps)]
    table = ';'.join(table)

    y = cy - 160 / 2
    ctrl = Et.SubElement(tab, 'control',
                         attrib={'x': '56', 'y': str(y), 'width': '48', 'height': '160',
                                 'parameterName': '≈', 'style': 'linear_vertical',
                                 'showLabel': 'true', 'textColor': ctrl_plt['modulation'], 'textSize': '40',
                                 'trackForegroundColor': ctrl_plt['modulation'],
                                 'trackBackgroundColor': track_bg_plt[0],
                                 'tooltip': 'Modulation (Low Pass Filter)',
                                 'type': 'float', 'minValue': '0', 'maxValue': '1', 'value': '1',
                                 'defaultValue': '1'})
    bind = Et.SubElement(ctrl, 'binding',
                         attrib={'type': 'effect', 'level': 'instrument', 'position': '1',
                                 'parameter': 'FX_FILTER_FREQUENCY', 'translation': 'table',
                                 'translationTable': table})
    bind.append(Et.Comment('Straight line in log scale'))

    # LP effect
    Et.SubElement(effects, 'effect', attrib={'type': 'lowpass', 'resonance': f'{.7}', 'frequency': f'{22000.0}'})

    # Link LP to CC 1
    cc = Et.SubElement(midi, 'cc', attrib={'number': '1'})
    Et.SubElement(cc, 'binding',
                  attrib={'level': 'ui', 'type': 'control', 'parameter': 'VALUE', 'position': '2',
                          'translation': "linear", 'translationOutputMin': "0", 'translationOutputMax': "1"})

    # - Reverb -
    if use_reverb:
        x = w - kw - 8
        y = cy - kw / 2
        verb_dv = reverb_wet * 100
        knob = Et.SubElement(tab, 'labeled-knob',
                             attrib={'x': str(x), 'y': str(y), 'width': str(kw), 'height': str(kw),
                                     'parameterName': '∞', 'type': 'percent',
                                     'showLabel': 'true', 'textColor': ctrl_plt['reverb'], 'textSize': '40',
                                     'trackForegroundColor': ctrl_plt['reverb'],
                                     'trackBackgroundColor': track_bg_plt[0],
                                     'tooltip': 'Reverb Wet Level',
                                     'minValue': '0', 'maxValue': '100',
                                     'value': f'{verb_dv}',
                                     'defaultValue': f'{verb_dv}'})

        prm = ('FX_REVERB_WET_LEVEL', 'FX_MIX')[bool(ir_samples)]
        Et.SubElement(knob, 'binding', attrib={'type': 'effect', 'level': 'instrument', 'position': '2',
                                               'parameter': prm, 'translation': 'linear',
                                               'translationOutputMin': '0', 'translationOutputMax': '1'})

        # Pull down menu to choose between available IR files
        if ir_samples:
            menu = Et.SubElement(tab, 'menu',
                                 attrib={'x': str(x), 'y': str(y + kw), 'width': str(kw), 'height': '32',
                                         'parameter': 'IR', 'value': '1'})
            for ir in sorted(ir_samples):
                ir_name = beautify_str(Path(ir).stem)
                option = Et.SubElement(menu, 'option', attrib={'name': ir_name})
                Et.SubElement(option, 'binding',
                              attrib={'type': 'effect', 'level': 'instrument', 'parameter': 'FX_IR_FILE',
                                      'position': '2', 'translation': 'fixed_value',
                                      'translationValue': str(Path(ir).as_posix())})

            Et.SubElement(effects, 'effect',
                          attrib={'type': 'convolution', 'mix': f'{verb_dv}', 'irFile': str(ir_samples[0])})
        else:
            Et.SubElement(effects, 'effect',
                          attrib={'type': 'reverb', 'wetLevel': f'{verb_dv}', 'roomSize': '0.8', 'damping': '0.75'})

        # Link Reverb to CC 19
        cc = Et.SubElement(midi, 'cc', attrib={'number': '19'})
        Et.SubElement(cc, 'binding',
                      attrib={'level': 'ui', 'type': 'control', 'parameter': 'VALUE', 'position': '3',
                              'translation': "linear", 'translationOutputMin': "0", 'translationOutputMax': "1"})

    # - ADSR Knobs -
    if adsr_knobs:
        prms = ['ENV_ATTACK', 'ENV_DECAY', 'ENV_SUSTAIN', 'ENV_RELEASE']
        tooltips = ['Attack', 'Decay', 'Sustain', 'Release']
        adsr_mx = [max_adsr_knobs, max_adsr_knobs, 1, max_adsr_knobs]
        adsr_margin = margin * 2.5
        w_m = w - adsr_margin * 2
        if add_grp_knobs:
            # To bottom and smaller when using group knobs
            adsr_spc = 64
            y = ui_h - adsr_spc * 1.125
        else:
            # To center and bigger when no group knob
            adsr_spc = 80
            y = cy - adsr_spc / 2
        for i, (knob_name, value, mx_value, prm, tooltip) in enumerate(zip('ADSR', adsr, adsr_mx, prms, tooltips)):
            x = (w_m / 4) * (i + .5) + adsr_margin - adsr_spc / 2

            knob = Et.SubElement(tab, 'labeled-knob',
                                 attrib={'x': str(x), 'y': str(y), 'width': str(adsr_spc), 'height': str(adsr_spc),
                                         'parameterName': knob_name, 'label': knob_name,
                                         'type': 'float',
                                         'showLabel': 'true', 'textColor': text_plt[0],
                                         'textSize': str(adsr_spc // 2),
                                         'trackForegroundColor': group_plt[0],
                                         'trackBackgroundColor': track_bg_plt[0],
                                         'tooltip': tooltip,
                                         'minValue': '0', 'maxValue': str(mx_value),
                                         'value': str(value),
                                         'defaultValue': str(value)})
            Et.SubElement(knob, 'binding',
                          attrib={'type': 'amp', 'level': 'instrument', 'position': '0', 'parameter': prm})

    # - Groups -
    groups = Et.SubElement(decentsampler, 'groups', attrib={'volume': '0dB', 'globalTuning': f'{tuning}'})

    # Velocity Track
    knob_w = 160
    knob_h = 48
    x = cx - 80
    y = 0
    ctrl = Et.SubElement(tab, 'control',
                         attrib={'x': str(x), 'y': str(y), 'width': str(knob_w), 'height': str(knob_h),
                                 'parameterName': '▼', 'label': '▼', 'style': 'linear_horizontal',
                                 'showLabel': 'true', 'textColor': other_plt['ampVelTrack'], 'textSize': '20',
                                 'trackForegroundColor': other_plt['ampVelTrack'],
                                 'trackBackgroundColor': track_bg_plt[0],
                                 'tooltip': 'Amplitude Velocity Tracking',
                                 'type': 'float', 'minValue': '0', 'maxValue': '1', 'value': str(vel_track),
                                 'defaultValue': str(vel_track)})
    Et.SubElement(ctrl, 'binding',
                  attrib={'type': 'amp', 'level': 'instrument', 'parameter': 'AMP_VEL_TRACK'})

    # Release Knob
    if enable_release:
        knob_h = 160

        if group_knobs_rows > 1:
            pos = math.ceil(len(instr.groups) / group_knobs_rows)
        else:
            pos = len(instr.groups)

        x_div = pos + 1
        x = ((w - margin * 2) / x_div) * (pos + .5) + margin - spc / 2
        y = cy - knob_h / 2

        ctrl = Et.SubElement(tab, 'control',
                             attrib={'x': str(round(x, 1)), 'y': str(y), 'width': str(spc), 'height': str(knob_h),
                                     'parameterName': '△', 'label': '△', 'style': 'linear_vertical',
                                     'showLabel': 'true', 'textColor': other_plt['ampVelTrack'], 'textSize': '32',
                                     'trackForegroundColor': other_plt['ampVelTrack'],
                                     'trackBackgroundColor': track_bg_plt[0],
                                     'tooltip': 'Release Volume',
                                     'type': 'float', 'minValue': '0', 'maxValue': '1', 'value': '1',
                                     'defaultValue': '1'})
        Et.SubElement(ctrl, 'binding',
                      attrib={'type': 'amp', 'level': 'tag', 'identifier': trig_tag_dict['release'],
                              'parameter': 'AMP_VOLUME'})

    # - Main loop -
    ctrls = []
    done = 0
    count = len(instr.samples) * len(rr_offset)
    btn_states = None

    ds_group, fk_rls_group, fk_leg_group = None, None, None
    grp_pos = 0

    for gt_idx, (grp, trg) in enumerate(instr.group_trigger):
        if isinstance(instr.limit, list):
            note_limit = instr.limit
        else:
            note_limit = instr.limit[grp]

        grp_df = instr.query_df(fltr={'group': grp, 'trigger': trg}, attr=None)
        samples = [instr.samples[i] for i in grp_df.index]
        grp_vels = grp_df['vel']

        use_vel = not grp_vels[grp_vels != 127].empty
        use_seq = not grp_df['seqPosition'][grp_df['seqPosition'] > 1].empty

        if progress is not None:
            progress.setMaximum(count)
            progress.setValue(0)
            progress.setTextVisible(True)
            progress.setFormat('%p%')

        # Disable Fake RR for group if sample number is too low
        grp_rro = ([0], rr_offset)[len(samples) > len(rr_offset)]

        grp_name = (grp, f'{grp}.{trg}')[trg != 'attack']
        print(f'Group: {grp_name}, Label: "{grp_label[grp]}"')

        grp_attrib = dict()
        grp_attrib['name'] = grp_name

        # Round-Robin mode
        if len(grp_rro) > 1 or use_seq:
            grp_attrib['seqMode'] = seq_mode or 'round_robin'
            if not use_seq:
                grp_attrib['seqLength'] = f'{len(grp_rro)}'
        else:
            grp_attrib['seqMode'] = 'always'

        # grp_attrib['tuning'] = f'{tuning}'

        # Per-group ADSR
        if adsr_knobs is False:
            for a, v in zip(['attack', 'decay', 'sustain', 'release'], adsr):
                if v is not None:
                    grp_attrib[a] = str(v)

        # Per-group ADR Release
        for a, v in zip(['attackCurve', 'decayCurve', 'releaseCurve'], adr_curve):
            if v is not None:
                grp_attrib[a] = str(v)

        if trg == 'attack' and fake_legato:
            grp_attrib['trigger'] = 'first'
            trig_tag = trig_tag_dict['first']
        else:
            grp_attrib['trigger'] = trg
            trig_tag = trig_tag_dict[trg]
        grp_attrib['tags'] = f'expression,{grp},{trig_tag}'

        if monophonic or fake_legato:
            grp_attrib['silencingMode'] = 'normal'
            if fake_legato:
                grp_attrib['silencedByTags'] = trig_tag_dict['legato']
            else:
                grp_attrib['silencedBy'] = grp

        # - Group creation -

        ds_group = Et.SubElement(groups, 'group', attrib=grp_attrib)
        grp_pos += 1

        # - Fake legato group -
        if fake_legato and trg == 'attack':
            fk_leg_grp_attrib = grp_attrib.copy()
            fk_leg_grp_attrib['name'] = f'{grp}.fake_legato'
            fk_leg_grp_attrib['trigger'] = 'legato'
            fk_leg_grp_attrib['tags'] = f'expression,{grp},{trig_tag_dict["legato"]}'

            if fk_leg_a is not None:
                fk_leg_grp_attrib['attack'] = str(fk_leg_a)
            if fk_leg_a_curve is not None:
                fk_leg_grp_attrib['attackCurve'] = str(fk_leg_a_curve)

            fk_leg_group = Et.SubElement(groups, 'group', attrib=fk_leg_grp_attrib)
            fk_leg_group.append(Et.Comment('Legato faked by re-using attack samples with a trimmed start'))
            grp_pos += 1

        # - Fake release group -
        if fake_release and trg == 'attack':
            fk_rls_grp_attrib = grp_attrib.copy()
            fk_rls_grp_attrib['name'] = f'{grp}.fake_release'
            fk_rls_grp_attrib['trigger'] = 'release'
            fk_rls_grp_attrib['tags'] = f'expression,{grp},{trig_tag_dict["release"]}'
            if fk_rls_volume != 0:
                fk_rls_grp_attrib['volume'] = f'{fk_rls_volume}dB'

            for a, v in zip(['attack', 'decay', 'sustain', 'release'], fk_rls_adsr):
                fk_rls_grp_attrib[a] = str(v)

            for a, v in zip(['attackCurve', 'decayCurve', 'releaseCurve'], fk_rls_adr_curve):
                fk_rls_grp_attrib[a] = str(v)

            # if fk_rls_mode == 'loop_end':
            #     fk_rls_grp_attrib['sustain'] = '0'

            # for attr in ['silencingMode', 'silencedByTags']:
            #     fk_rls_grp_attrib.pop(attr, None)

            fk_rls_group = Et.SubElement(groups, 'group', attrib=fk_rls_grp_attrib)
            fk_rls_group.append(Et.Comment('Release faked by re-using attack samples with a different envelope'))
            grp_pos += 1

            if fk_rls_mode == 'start':
                fk_rls_fx = Et.SubElement(fk_rls_group, 'effects')
                Et.SubElement(fk_rls_fx, 'effect',
                              attrib={'type': 'lowpass', 'resonance': f'{.7}', 'frequency': f'{fk_rls_cutoff}'})
            else:
                fk_rls_grp_attrib['sustain'] = '1'

        if len(grp_rro) > 1:
            comment = ' '.join([str(v) for v in grp_rro])
            ds_group.append(Et.Comment(f'Fake Round-Robin offset table {comment}'))
            if fk_rls_group:
                fk_rls_group.append(Et.Comment(f'Fake Round-Robin offset table {comment}'))
            if fk_leg_group:
                fk_leg_group.append(Et.Comment(f'Fake Round-Robin offset table {comment}'))

        # - Group Volume -
        if add_grp_knobs:
            ctrlname = grp_label[grp]
            slider_idx = instr.groups.index(grp)

            m_spc = spc

            if group_knobs_rows > 1:
                m_spc = spc * 1.25
                n_x = math.ceil(len(instr.groups) / group_knobs_rows)
                x_div = n_x + int(enable_release)
                row_n = slider_idx // n_x
                x = ((w - margin * 2) / x_div) * (slider_idx % n_x + .5) + margin - m_spc / 2 / group_knobs_rows
                y = cy + (row_n - 1) * spc / group_knobs_rows
            else:
                x_div = len(instr.groups) + int(enable_release)
                x = ((w - margin * 2) / x_div) * (slider_idx + .5) + margin - spc / 2
                y = cy - spc / 2

            if adsr_knobs:
                y -= adsr_spc / 4

            y -= btn_w // 2

            if ctrlname not in ctrls:
                ctrl_w = m_spc / group_knobs_rows
                ctrl = Et.SubElement(tab, 'control',
                                     attrib={'x': str(x), 'y': str(y),
                                             'width': str(ctrl_w), 'height': str(ctrl_w),
                                             'style': 'rotary_vertical_drag',
                                             'parameterName': shorten_str(grp), 'label': ctrlname,
                                             'showLabel': 'true', 'textColor': text_plt[0], 'textSize': str(ts),
                                             'trackForegroundColor': group_plt[slider_idx % len(group_plt)],
                                             'trackBackgroundColor': track_bg_plt[0],
                                             'tooltip': f'{beautify_str(grp)} Volume',
                                             'type': 'float', 'minValue': "0", 'maxValue': "1", 'value': '1',
                                             'defaultValue': '1'})
                Et.SubElement(ctrl, 'binding',
                              attrib={'type': 'amp', 'level': 'tag', 'identifier': grp, 'parameter': 'AMP_VOLUME'})
                ctrls.append(ctrlname)

                # Mute button
                btn = Et.SubElement(tab, 'button',
                                    attrib={'x': str(x + (ctrl_w - btn_w) // 2), 'y': str(y + ctrl_w - btn_w // 2),
                                            'style': 'image', 'width': str(btn_w), 'height': str(btn_w),
                                            'value': '1', 'tooltip': f'{ctrlname} On/Off'})
                btn_states = [
                    Et.SubElement(btn, 'state', attrib={'name': n, 'mainImage': p})
                    for n, p in zip(['off', 'on'], btn_paths)]

            btn_binds = 1
            if trg == 'attack':
                if fake_release:
                    btn_binds += 1
                if fake_legato:
                    btn_binds += 1

            for p in range(btn_binds):
                for st_idx, btn_state in enumerate(btn_states):
                    for tgl in range(2):
                        value = (1 - st_idx, st_idx)[tgl]
                        value_name = ('false', 'true')[value]
                        Et.SubElement(btn_state, 'binding',
                                      attrib={'type': 'general', 'level': 'group',
                                              'position': str(grp_pos + p - btn_binds),
                                              'parameter': 'ENABLED', 'translation': 'fixed_value',
                                              'translationValue': value_name})

        # Per-group Keyboard color
        if note_spread != 'none':
            lo, hi = note_limit
            if len(keyboard_plt) > 1:
                if lo > 0:
                    Et.SubElement(keyboard, 'color',
                                  attrib={'loNote': '0', 'hiNote': str(lo - 1), 'color': keyboard_plt[0]})
                if hi < 127:
                    Et.SubElement(keyboard, 'color',
                                  attrib={'loNote': str(hi + 1), 'hiNote': '127', 'color': keyboard_plt[0]})
            Et.SubElement(keyboard, 'color',
                          attrib={'loNote': str(lo), 'hiNote': str(hi), 'color': keyboard_plt[-1]})

        # Add Sample
        for i, o in enumerate(grp_rro):
            if len(grp_rro) > 1:
                print(f'  Fake RR Offset: {o}')
                ds_group.append(Et.Comment(f'Fake RR offset {o}'))
                if fk_rls_group:
                    fk_rls_group.append(Et.Comment(f'Fake RR offset {o}'))
                if fk_leg_group:
                    fk_leg_group.append(Et.Comment(f'Fake RR offset {o}'))

            for s, info in enumerate(samples):
                done += 1

                smp_attrib = {}
                note, pitch_fraction, loop_start, loop_end = info.note, info.pitchFraction, info.loopStart, info.loopEnd
                cues = info.cues or []
                vel = info.vel or 127
                seq = info.seqPosition or 1
                smp_len = info.params.nframes
                sr = info.params.framerate

                # Per-note keyboard color
                if note_spread == 'none':
                    Et.SubElement(keyboard, 'color',
                                  attrib={'loNote': str(note), 'hiNote': str(note), 'color': keyboard_plt})

                # Note mapping
                notes = instr.notes_per_group_trigger_vel[grp][trg][vel]
                lo_note, hi_note, mn, mx = iu.extend_note_range(note, notes, mode=note_spread, limit=note_limit)
                print(f'    {info.name} ({note}) {lo_note}-{hi_note}')

                # Offset sample for fake Round-robin
                if o != 0:
                    zones = instr.zones[grp][trg][vel][seq]

                    if len(grp_rro) > len(zones):
                        continue

                    zone_ids = zones.index

                    smp_idx = zones.tolist().index(info.name)
                    o_idx = smp_idx + o
                    if o_idx < 0:
                        o_idx = (smp_idx, smp_idx - o + 1)[rr_bounds]
                    elif o_idx > len(zones) - 1:
                        o_idx = (smp_idx, smp_idx - o - 1)[rr_bounds]

                    if o_idx != smp_idx:
                        o_smp = instr.samples[zone_ids[o_idx]]
                        info = o_smp
                        note, pitch_fraction, loop_start, loop_end = (info.note, info.pitchFraction,
                                                                      info.loopStart, info.loopEnd)
                        cues = info.cues or []
                        smp_len = info.params.nframes
                        sr = info.params.framerate
                    else:
                        o_smp = info

                    smp_attrib['path'] = str(Path(o_smp.path).relative_to(root_dir).as_posix())
                else:
                    smp_attrib['path'] = str(Path(info.path).relative_to(root_dir).as_posix())

                smp_attrib['rootNote'] = str(note)
                smp_attrib['loNote'] = str(lo_note)
                smp_attrib['hiNote'] = str(hi_note)

                # Pitch Fraction
                smp_tuning = None
                if pf_mode != 'off' and pitch_fraction is not None:
                    if abs(pitch_fraction) > .001:
                        smp_tuning = -pitch_fraction / 100
                    if smp_tuning is not None:
                        smp_attrib['tuning'] = f'{smp_tuning:.03f}'

                # Velocity mapping
                vels = instr.vels_per_group_trigger[grp][trg]
                if use_vel and len(vels) > 1:
                    lo_vel, hi_vel, _, _ = iu.extend_note_range(vel, vels, mode='down', limit=(1, 127))
                    smp_attrib['loVel'] = str(lo_vel)
                    smp_attrib['hiVel'] = str(hi_vel)

                # Set sequence position for Round-Robin
                if len(grp_rro) > 1:
                    smp_attrib['seqPosition'] = str(i + 1)
                elif info.seqPosition is not None:
                    smp_attrib['seqPosition'] = str(info.seqPosition)

                if use_seq:
                    seqs = instr.seqs_per_group_trigger_vel_note[grp][trg][vel][note]
                    seq_length = max(seqs)
                    smp_attrib['seqLength'] = str(seq_length)

                # Loop
                if loop and loop_start is not None and loop_end is not None:
                    smp_attrib['loopEnabled'] = '1'
                    smp_attrib['loopStart'] = str(loop_start)
                    smp_attrib['loopEnd'] = str(loop_end)
                else:
                    smp_attrib['loopEnabled'] = '0'

                # Cross-Fade
                if crossfade_mode in ['linear', 'equal_power'] and smp_attrib['loopEnabled'] == '1':
                    smp_attrib['loopCrossfadeMode'] = crossfade_mode
                    crosslen = (loop_end - loop_start) * crossfade
                    smp_attrib['loopCrossfade'] = str(int(crosslen))

                # Note Pan
                if note_pan != 0:
                    pan = linstep(mn, mx, (lo_note + hi_note) / 2)
                    pan = lerp(0, (pan * 2 - 1) * 100, note_pan)
                    smp_attrib['pan'] = str(round(pan, 1))

                # - Extra sample attributes -
                # Potential support for any smp_attrib offered by Decent Sampler
                # albeit with absolutely zero check to verify if the supplied values make sense or not

                attrlist = [attr for attr in dir(info) if attr in ds_smp_attrib]

                for attr in attrlist:
                    value = getattr(info, attr)
                    if value is not None and attr not in grp_attrib:
                        smp_attrib[attr] = str(value)

                # - Add Sample to group -

                Et.SubElement(ds_group, 'sample', attrib=smp_attrib)

                # Add fake release sample to its related group
                # Most attributes are copied from source sample
                if fake_release and trg == 'attack':
                    fk_rls_smp_attrib = smp_attrib.copy()
                    fk_rls_smp_tuning = fk_rls_tuning

                    if smp_tuning is not None:
                        fk_rls_smp_tuning = smp_tuning + fk_rls_tuning
                    if abs(fk_rls_smp_tuning) > .001:
                        fk_rls_smp_attrib['tuning'] = f'{fk_rls_smp_tuning:.03f}'

                    match fk_rls_mode:
                        case 'loop_end' | 'cue':
                            fk_rls_smp_attrib['loopEnabled'] = '0'
                            for attr in ['loopStart', 'loopEnd', 'loopCrossfadeMode', 'loopCrossfade']:
                                fk_rls_smp_attrib.pop(attr, None)
                            smp_start = (loop_end, cues[0])[fk_rls_mode == 'cue']
                            fk_rls_smp_attrib['start'] = str(smp_start)

                    Et.SubElement(fk_rls_group, 'sample', attrib=fk_rls_smp_attrib)

                # Add fake legato sample to its related group
                # Most attributes are copied from source sample
                if fake_legato and trg == 'attack':
                    fk_leg_smp_attrib = smp_attrib.copy()

                    # Trim start of the sample
                    smp_start = int(sr * fk_leg_start)
                    # TODO : check if necessary
                    if fk_leg_smp_attrib['loopEnabled'] == '1':
                        smp_start = min(smp_start, loop_start)
                    fk_leg_smp_attrib['start'] = str(smp_start)

                    Et.SubElement(fk_leg_group, 'sample', attrib=fk_leg_smp_attrib)

                if progress is not None:
                    progress.setValue(done)

    # Write XML
    basename = Path(root_dir).stem
    if add_suffix:
        basename += add_suffix
    filepath = Path.joinpath(Path(root_dir), f'{basename}.dspreset')

    if auto_increment:
        filepath = resolve_overwriting(filepath, mode='file', test_run=True)

    write_xml_to_file(decentsampler, str(filepath))

    if progress is not None:
        progress.setValue(count)
        progress.setFormat(f'{smp_count} sample(s) found.')

    print('')

    return filepath


# File function

def write_xml_to_file(xml_tree, output_file):
    xml_str = minidom.parseString(Et.tostring(xml_tree, encoding='utf-8')).toprettyxml(indent='  ')
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(xml_str)


def get_dspreset_dependencies(dspreset_file):
    """
    Get dependencies from a dspreset file
    :param str or Path dspreset_file:
    :return: List of files
    :rtype: set
    """
    root_dir = Path(dspreset_file).parent
    tree = Et.parse(dspreset_file)
    root = tree.getroot()
    result = set()
    # Retrieve all attribute values qualifying as valid file paths
    for elem in root.iter():
        result |= {v for v in elem.attrib.values() if Path(root_dir).joinpath(v).is_file()}
    return result


def create_dslibrary(root_dir):
    """
    Create a dslibrary file from a root directory archiving only files required by found dspreset(s)
    :param str or Path root_dir:
    :return: Created file path
    :rtype: str
    """
    # Get all dspreset files from root_dir
    dspreset_files = Path(root_dir).glob('*.dspreset')

    if not dspreset_files:
        return None

    # Get dependencies from each dspreset
    deps = set()
    for f in dspreset_files:
        deps.add(str((Path(f).relative_to(root_dir))))
        deps |= get_dspreset_dependencies(f)
    deps = sorted(list(deps))

    # Copy retrieved files to a temp directory
    base_name = Path(root_dir).name

    temp_dir = Path(tempfile.gettempdir()).joinpath(base_name)
    for f in deps:
        src = Path(root_dir).joinpath(f)
        if not src.is_file():
            continue
        dst = Path(temp_dir).joinpath(f)
        dst_dir = dst.parent
        if not dst_dir.exists():
            os.makedirs(dst_dir)
        shutil.copyfile(src=src, dst=dst)

    # Zip temp directory, rename to dslibrary and delete temp directory
    zip_basename = Path(root_dir).parent.joinpath(f'{base_name}')
    zip_path = shutil.make_archive(str(zip_basename), 'zip', root_dir=temp_dir)
    result_path = Path(root_dir).parent.joinpath(f'{base_name}.dslibrary')
    Path(zip_path).replace(result_path)
    shutil.rmtree(temp_dir, ignore_errors=True)

    return str(result_path)
