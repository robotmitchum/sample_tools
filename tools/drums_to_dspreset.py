# coding:utf-8
"""
    :module: drums_to_dspreset.py
    :description:
    :author: Michel 'Mitch' Pecqueur
    :date: 2025.03
"""

import getpass
import math
import os
import xml.etree.ElementTree as Et
from pathlib import Path

import numpy as np

import instrument_utils as iu
from color_utils import (adjust_palette, basic_background, blank_button, write_pil_image,
                         write_text, apply_symbol, hex_to_rgba, rgba_to_hex, plt_to_rgba)
from common_math_utils import lerp
from common_ui_utils import shorten_str, beautify_str
from file_utils import recursive_search, resolve_overwriting
from jsonFile import read_json
from smp_to_dspreset import write_xml_to_file

__ds_version__ = '1.12.11'
__version__ = '1.0.1'


def create_drums_dspreset(root_dir='', smp_subdir='Samples', data=None,

                          smp_fmt=('wav', 'flac'), ir_fmt=('wav', 'aif', 'flac'),
                          smp_attrib_cfg='smp_attrib_cfg.json',
                          pattern='{group}_{vel}_{seqPosition}', group_naming='keep',
                          override=True, loop=True,

                          bg_text=None, seq_mode='random',
                          text_font=('HelveticaNeueThin.otf', 24),
                          color_plt_cfg='plt_cfg/Dark_plt_cfg.json', plt_adjust=(0, 1, 1),

                          tuning_knobs=False, volume_knobs=True, pan_knobs=True,

                          attenuation=-3, vel_track=1.0,

                          use_reverb=True, reverb_wet=0.2, ir_subdir='IR',
                          multi_out=True,

                          add_suffix='', auto_increment=False,

                          worker=None, progress_callback=None, message_callback=None):
    """
    :param str root_dir:
    :param str smp_subdir:
    :param dict or None data:

    :param list smp_fmt:
    :param list ir_fmt:

    :param str smp_attrib_cfg: Path to a json file holding sample attribute configuration
    :param str color_plt_cfg: Path to a json file holding color configuration used to generate UI

    :param bool tuning_knobs: Add tuning knobs
    :param bool pan_knobs: Add pan knobs
    :param boll volume_knobs: Add volume knobs

    :param str pattern: Pattern used to figure the mapping
    :param str group_naming: 'keep', 'beautify', 'upper', 'lower','shorten'
    :param bool override: If True, info from sample name override metadata
    :param bool loop: Use sample loop

    :param str seq_mode: Sequence mode

    :param str bg_text:
    :param tuple text_font:

    :param str or Path color_plt_cfg:
    :param list plt_adjust:
    :param float attenuation:
    :param float vel_track:

    :param bool use_reverb:
    :param float reverb_wet:
    :param str ir_subdir:

    :param bool multi_out: Enable multiple output

    :param str add_suffix:
    :param bool auto_increment:

    :param Worker worker:
    :param progress_callback:
    :param message_callback:
    :return:
    """

    group_names = [grp.get('name', f'drum_{i + 1:02d}') for i, grp in enumerate(data['groups'])]

    current_dir = Path(__file__).parent
    base_dir = current_dir.parent

    # Fix shortened sample paths
    for grp in data.get('groups', []):
        grp_samples = []
        for smp in grp.get('samples', []):
            p = Path(smp)
            if not p.is_file():
                p = Path(root_dir) / f'{smp_subdir}/{smp}'
            grp_samples.append(p)
        grp['samples'] = grp_samples

    # smp_df, smp_info = iu.audio_files_to_df(files=files, pattern=pattern, transpose=0)
    # smp_info = {smp.name: smp for smp in smp_info}

    # List of sample attributes offered by Decent Sampler and *hopefully* supported by this tool
    # Attribute values can be provided in the sample name (with some limitations depending on how the name is formatted)
    # or using ID3 tags (flac only at the moment, it should support any attribute, but I didn't test everything...)

    num_attrs, ds_smp_attrib = ['vel', 'note', 'seqPosition'], []

    p = Path(smp_attrib_cfg)
    if not p.is_file():
        p = base_dir / smp_attrib_cfg
        if not p.is_file():
            print(f'{p} not found')
            return False

    smp_attrib_data = read_json(p) or {}
    if 'num_attrib' in smp_attrib_data:
        num_attrs = smp_attrib_data['num_attrib']
    if 'ds_smp_attrib' in smp_attrib_data:
        ds_smp_attrib = smp_attrib_data['ds_smp_attrib']

    exclude = ('backup_', 'ignore')
    # instr = iu.Instrument(root_dir=root_dir, smp_subdir=smp_subdir, files=files, smp_fmt=smp_fmt, exclude=exclude,
    #                       pattern=pattern, transpose=0,
    #                       override=override, extra_tags=ds_smp_attrib, num_attrib=num_attrs)
    #
    # print(instr.groups)

    # IR Samples
    ir_samples = []
    if ir_subdir:
        ir_samples = recursive_search(root_dir=Path(root_dir) / ir_subdir, input_ext=ir_fmt,
                                      exclude=exclude, relpath=root_dir)

    # Color Palette Config
    p = Path(color_plt_cfg)
    if not p.is_file():
        p = base_dir / color_plt_cfg
        if not p.is_file():
            print(f'{p} not found')
            return False

    plt_data = read_json(p) or {}
    if plt_adjust:
        plt_data = adjust_palette(plt_data=plt_data, adjust=plt_adjust)

    text_plt = list(plt_data['textColor'].values())
    track_bg_plt = list(plt_data['trackBackgroundColor'].values())
    ctrl_plt = plt_data['control']
    other_plt = plt_data['other']
    bg_plt = plt_data['backgroundRamp']
    group_plt = list(plt_data['group'].values())

    if 'keyboard' in plt_data:
        keyboard_plt = list(plt_data['keyboard'].values())
    else:
        keyboard_plt = [rgba_to_hex(rgba=[1, .625, .875, 1])]

    # Active keys color
    if len(keyboard_plt) > 1:
        active_keys_plt = keyboard_plt[1:]
    else:
        active_keys_plt = [keyboard_plt]

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
    resources_dir = Path(root_dir) / 'Resources'
    if not resources_dir.exists():
        os.makedirs(resources_dir, exist_ok=True)

    bg_colors = [hex_to_rgba(h)[1:] for h in bg_plt.values()]
    bg_img = basic_background(filepath=None, w=w, h=h, scl=2, overwrite=True, colors=bg_colors, gamma=1.0,
                              text=bg_text, text_xy=(8, top_band_h + 8), text_font=text_font,
                              text_color=plt_to_rgba(bg_text_plt[0]))

    info_w = 24
    blank_button_path = resources_dir / 'blank_button.png'
    blank_button_path = blank_button(str(blank_button_path), size=info_w, overwrite=True)
    blank_button_path = Path(blank_button_path).relative_to(root_dir)

    # - Group slider labels -
    grp_naming_func = {'beautify': beautify_str, 'upper': lambda x: x.upper(), 'lower': lambda x: x.lower(),
                       'shorten': shorten_str, 'keep': lambda x: x}

    grp_label = {g: grp_naming_func[group_naming](g) for g in group_names}

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
                       attrib={'width': str(w), 'height': str(h), 'bgImage': '', 'layoutMode': 'relative',
                               'bgColor': "00000000"})
    tab = Et.SubElement(ui, 'tab', attrib={'name': 'main'})
    keyboard = Et.SubElement(ui, 'keyboard')

    # Information / Credits
    info_text, info_tooltip = 'Info', ''

    info_filepath = Path(root_dir) / 'INFO.txt'
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

    symbol_path = current_dir / 'symbols/info.png'
    bg_img = apply_symbol(bg_img, symbol_path=symbol_path, pos_xy=(w - info_w - 8, top_band_h + 8),
                          size=(info_w, info_w), scl=2, color=plt_to_rgba(bg_text_plt[-1], alpha=.5))

    Et.SubElement(tab, 'button',
                  attrib={'style': 'image', 'x': str(w - info_w - 8), 'y': str(8), 'width': str(info_w),
                          'height': str(info_w), 'mainImage': str(blank_button_path.as_posix()),
                          'tooltip': info_tooltip})

    # - Add some headroom to avoid saturation -
    Et.SubElement(effects, 'effect', attrib={'type': 'gain', 'level': f'{attenuation}'})

    # - Default controls -
    mx_len = np.median([len(name) for name in group_names])  # Max Word Length
    kw = 96  # Knob Width
    spc = 80  # Group knob width/height
    ts = round(min((spc / mx_len) * 3, spc / 2))  # Text size
    margin = 120  # Margin for group sliders from canvas left border

    # - "Expression" (Volume) -
    x = 8
    sw = 32
    ew, eh = 48, 160 - sw
    y = cy - eh // 2 + sw

    symbol_path = current_dir / 'symbols/expression.png'
    bg_img = apply_symbol(bg_img, symbol_path=symbol_path,
                          pos_xy=(x + ew // 2 - sw // 2, top_band_h + y - sw),
                          size=(sw, sw), scl=2, color=plt_to_rgba(ctrl_plt['expression']))

    vol_ctrl = Et.SubElement(tab, 'control',
                             attrib={'x': str(x), 'y': str(y), 'width': str(ew), 'height': str(eh),
                                     'parameterName': 'expression', 'style': 'linear_vertical',
                                     'showLabel': 'false', 'textColor': ctrl_plt['expression'], 'textSize': '40',
                                     'trackForegroundColor': ctrl_plt['expression'],
                                     'trackBackgroundColor': track_bg_plt[0],
                                     'tooltip': 'Expression',
                                     'type': 'float', 'minValue': '0', 'maxValue': '1', 'value': '1',
                                     'defaultValue': '1'})
    Et.SubElement(vol_ctrl, 'binding',
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

    x = 56
    sw = 32
    ew, eh = 48, 160 - sw
    y = cy - eh // 2 + sw

    symbol_path = current_dir / 'symbols/modulation.png'
    bg_img = apply_symbol(bg_img, symbol_path=symbol_path,
                          pos_xy=(x + ew // 2 - sw // 2, top_band_h + y - sw),
                          size=(sw, sw), scl=2, color=plt_to_rgba(ctrl_plt['modulation']))

    vol_ctrl = Et.SubElement(tab, 'control',
                             attrib={'x': str(x), 'y': str(y), 'width': str(ew), 'height': str(eh),
                                     'parameterName': 'modulation', 'style': 'linear_vertical',
                                     'showLabel': 'false', 'textColor': ctrl_plt['modulation'], 'textSize': '40',
                                     'trackForegroundColor': ctrl_plt['modulation'],
                                     'trackBackgroundColor': track_bg_plt[0],
                                     'tooltip': 'Modulation (Low Pass Filter)',
                                     'type': 'float', 'minValue': '0', 'maxValue': '1', 'value': '1',
                                     'defaultValue': '1'})
    bind = Et.SubElement(vol_ctrl, 'binding',
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
        rw = 96
        rh = rw
        x = w - rw - 8
        y = cy - rh / 2
        verb_dv = reverb_wet * 100

        symbol_path = current_dir / 'symbols/reverb.png'
        sw = 32
        bg_img = apply_symbol(bg_img, symbol_path=symbol_path,
                              pos_xy=(w - rw / 2 - sw / 2 - 8, top_band_h + cy - sw // 2),
                              size=(sw, sw), scl=2, color=plt_to_rgba(ctrl_plt['reverb']))

        knob = Et.SubElement(tab, 'labeled-knob',
                             attrib={'x': str(x), 'y': str(y), 'width': str(rw), 'height': str(rh),
                                     'parameterName': 'reverb', 'type': 'percent',
                                     'showLabel': 'false',
                                     'textColor': ctrl_plt['reverb'], 'textSize': '40',
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

    # - Groups -
    tuning = 0
    groups = Et.SubElement(decentsampler, 'groups', attrib={'volume': '0dB', 'globalTuning': f'{tuning}'})

    # Velocity Track
    knob_w = 160
    knob_h = 32
    x = cx - knob_w // 2
    y = 0

    symbol_path = current_dir / 'symbols/velocity.png'
    sw = knob_h // 3
    bg_img = apply_symbol(bg_img, symbol_path=symbol_path,
                          pos_xy=(cx - sw // 2, top_band_h + knob_h // 4 - sw // 2),
                          size=(sw, sw), scl=2, color=plt_to_rgba(other_plt['ampVelTrack']))

    vel_ctrl = Et.SubElement(tab, 'control',
                             attrib={'x': str(x), 'y': str(y), 'width': str(knob_w), 'height': str(knob_h),
                                     'parameterName': 'ampVelTrack', 'label': 'amplitude velocity tracking',
                                     'style': 'linear_horizontal', 'showLabel': 'false',
                                     'trackForegroundColor': other_plt['ampVelTrack'],
                                     'trackBackgroundColor': track_bg_plt[0],
                                     'tooltip': 'Amplitude Velocity Tracking',
                                     'type': 'float', 'minValue': '0', 'maxValue': '1', 'value': str(vel_track),
                                     'defaultValue': str(vel_track)})
    Et.SubElement(vel_ctrl, 'binding',
                  attrib={'type': 'amp', 'level': 'instrument', 'parameter': 'AMP_VEL_TRACK'})

    active_keys = set()
    note_idx = 0

    # - Main loop -
    smp_count = 0
    done = 0
    count = len(data['groups'])

    for g, grp in enumerate(data['groups']):
        if progress_callback is not None:
            progress_callback.emit(0)
            message_callback.emit('%p%')

        grp_attrib, smp_attrib = {}, {}

        grp_name = group_names[g]
        print(f'Group: {grp_name}')

        grp_attrib['name'] = grp_name
        grp_attrib['tags'] = 'expression'
        grp_samples = grp['samples']
        label_text = grp.get('label', shorten_str(grp_name))

        smp_df, smp_info = iu.audio_files_to_df(files=grp_samples, pattern=pattern)
        smp_info = {info.name: info for info in smp_info}

        seqs = smp_df['seqPosition']
        use_seq = not seqs[seqs > 1].empty

        # Choke
        choke_indices = grp.get('choke', [])
        # choke_indices.append(g)  # Self-Choke
        choke_names = [group_names[c] for c in sorted(list(set(choke_indices)))]
        grp_attrib['silencedBy'] = ','.join(choke_names)
        grp_attrib['silencingMode'] = 'fast'

        # Round-Robin mode
        if use_seq:
            grp_attrib['seqMode'] = seq_mode or 'round_robin'
        else:
            grp_attrib['seqMode'] = 'always'

        # Note and tuning
        note = grp.get('note', g + 36)
        smp_attrib['rootNote'] = str(note)
        smp_attrib['loNote'] = str(note)
        smp_attrib['hiNote'] = str(note)

        # Per-drum keyboard color
        active_keys.add(note)
        plt_idx = note_idx % len(active_keys_plt)
        note_idx += 1
        Et.SubElement(keyboard, 'color',
                      attrib={'loNote': str(note), 'hiNote': str(note), 'color': active_keys_plt[plt_idx]})

        # ADR Curve
        for a, v in zip(['attackCurve', 'decayCurve', 'releaseCurve'], [-100, -100, -100]):
            grp_attrib[a] = str(v)

        # Multi-Output
        if multi_out:
            grp_attrib['output1Target'] = 'MAIN_OUTPUT'
            grp_attrib['output1Volume'] = '1.0'
            grp_attrib['output2Target'] = f'AUX_STEREO_OUTPUT_{g + 1}'
            grp_attrib['output2Volume'] = '1.0'

        # Tuning, Volume, Pan
        tuning = grp.get('tuning', 0)
        volume = grp.get('volume', 1)
        pan = grp.get('pan', 0)

        if not tuning_knobs and tuning != 0:
            grp_attrib['tuning'] = str(tuning)
        if not volume_knobs and volume != 1:
            grp_attrib['volume'] = str(volume)
        if not pan_knobs and pan != 0:
            grp_attrib['pan'] = str(pan)

        # Create group
        group = Et.SubElement(groups, 'group', attrib=grp_attrib)
        tp_w = 40

        x_div = len(group_names)
        ctrl_w, ctrl_h = 12, 128
        x = ((w - margin * 2) / x_div) * (g + .5) + margin
        y = cy - tp_w

        if tuning_knobs:
            knob_name = f'{grp_name} tuning'
            tuning_ctrl = Et.SubElement(tab, 'control',
                                        attrib={'x': str(x - tp_w // 2), 'y': str(y),
                                                'width': str(tp_w), 'height': str(tp_w),
                                                'style': 'rotary_vertical_drag',
                                                'parameterName': shorten_str(knob_name), 'label': knob_name,
                                                'showLabel': 'false', 'textColor': text_plt[0], 'textSize': str(ts),
                                                'trackForegroundColor': group_plt[g % len(group_plt)],
                                                'trackBackgroundColor': track_bg_plt[0],
                                                'tooltip': f'{beautify_str(knob_name)}',
                                                'type': 'float', 'minValue': "-36", 'maxValue': "36",
                                                'value': str(tuning), 'defaultValue': str(tuning)})
            Et.SubElement(tuning_ctrl, 'binding',
                          attrib={'type': 'amp', 'level': 'group', 'position': str(g), 'parameter': 'GROUP_TUNING'})

        if pan_knobs:
            knob_name = f'{grp_name} pan'
            pan_y = y + (0, tp_w)[tuning_knobs]
            pan_ctrl = Et.SubElement(tab, 'control',
                                     attrib={'x': str(x - tp_w // 2), 'y': str(pan_y),
                                             'width': str(tp_w), 'height': str(tp_w),
                                             'style': 'rotary_vertical_drag',
                                             'parameterName': shorten_str(knob_name), 'label': knob_name,
                                             'showLabel': 'false', 'textColor': text_plt[0], 'textSize': str(ts),
                                             'trackForegroundColor': group_plt[g % len(group_plt)],
                                             'trackBackgroundColor': track_bg_plt[0],
                                             'tooltip': f'{beautify_str(knob_name)}',
                                             'type': 'float', 'minValue': "-100", 'maxValue': "100",
                                             'value': str(pan), 'defaultValue': str(pan)})
            Et.SubElement(pan_ctrl, 'binding',
                          attrib={'type': 'amp', 'level': 'group', 'position': str(g), 'parameter': 'PAN'})

        if volume_knobs:
            knob_name = f'{grp_name} volume'
            vol_y = y + (0, tp_w)[tuning_knobs] + (0, tp_w)[pan_knobs]
            vol_h = ctrl_h - (0, tp_w)[tuning_knobs] - (0, tp_w)[pan_knobs]
            vol_ctrl = Et.SubElement(tab, 'control',
                                     attrib={'x': str(x - ctrl_w // 2), 'y': str(vol_y),
                                             'width': str(ctrl_w), 'height': str(vol_h),
                                             'style': 'linear_bar_vertical',
                                             'parameterName': shorten_str(knob_name), 'label': knob_name,
                                             'showLabel': 'false', 'textColor': text_plt[0], 'textSize': str(ts),
                                             'trackForegroundColor': group_plt[g % len(group_plt)],
                                             'trackBackgroundColor': track_bg_plt[0],
                                             'tooltip': f'{beautify_str(knob_name)}',
                                             'type': 'float', 'minValue': "0", 'maxValue': "1",
                                             'value': str(volume), 'defaultValue': str(volume)})
            Et.SubElement(vol_ctrl, 'binding',
                          attrib={'type': 'amp', 'level': 'group', 'position': str(g), 'parameter': 'AMP_VOLUME'})

        lbl_h = 16
        if any([tuning_knobs, pan_knobs, volume_knobs]):
            bg_img = write_text(bg_img, text=label_text, text_xy=(x, top_band_h + y - lbl_h),
                                text_font=(text_font[0], lbl_h), text_color=plt_to_rgba(text_plt[0]),
                                align='center', anchor='mt', stroke_width=0, scl=2)

        # Add samples

        for smp in sorted(grp_samples):

            smp_path = Path(smp).relative_to(root_dir)
            info = smp_info[smp_path.stem]

            print(f'\tSample: {smp_path.name}')
            smp_attrib['path'] = str(smp_path)

            ratio = pow(2, -tuning / 12)
            duration = round(info.params.nframes / info.params.framerate * ratio, 3)

            # ADSR
            for a, v in zip(['attack', 'decay', 'sustain', 'release'], [0, duration, 1, duration]):
                smp_attrib[a] = str(v)

            # - Velocity and round-robin mapping -
            vel, seq = info.vel or 127, info.seqPosition or 1

            vels = smp_df[smp_df['seqPosition'] == seq]['vel']
            use_vel = not vels[vels != 127].empty

            # Velocity mapping
            if use_vel and len(vels) > 1:
                lo_vel, hi_vel, _, _ = iu.extend_note_range(vel, vels.to_list(), mode='down', limit=(1, 127))
                smp_attrib['loVel'] = str(lo_vel)
                smp_attrib['hiVel'] = str(hi_vel)

            # Round-Robin
            if use_seq:
                smp_attrib['seqPosition'] = str(seq)
                smp_attrib['seqLength'] = str(max(seqs))

            Et.SubElement(group, 'sample', attrib=smp_attrib)
            smp_count += 1

        done += 1
        if progress_callback is not None:
            percent = int(done / count * 100)
            progress_callback.emit(percent)

    # Color inactive keys
    if len(keyboard_plt) > 1 and len(active_keys) < 128:
        inactive_keys = np.delete(np.arange(128), np.array(list(active_keys)))
        diff = np.diff(inactive_keys)
        idx = np.argwhere(diff > 1).reshape(-1)
        idx = np.append(np.append(idx, idx + 1), np.array([0, len(inactive_keys) - 1]))
        inactive_ranges = inactive_keys[np.sort(idx)].reshape(-1, 2)
        for r in inactive_ranges:
            lo, hi = r.tolist()
            Et.SubElement(keyboard, 'color',
                          attrib={'loNote': str(lo), 'hiNote': str(hi), 'color': keyboard_plt[0]})

    # Write XML and background image
    basename = Path(root_dir).stem
    bg_basename = 'bg'
    if add_suffix:
        basename += add_suffix
        bg_basename += add_suffix
    filepath = Path(root_dir) / f'{basename}.dspreset'
    bg_path = Path(root_dir) / f'Resources/{bg_basename}.jpg'

    if auto_increment:
        filepath = resolve_overwriting(filepath, mode='file', test_run=True)
        version = Path(filepath).stem.split('_')[-1]
        bg_path = Path(root_dir) / f'Resources/{bg_basename}_{version}.jpg'

    rel_bg_path = bg_path.relative_to(root_dir).as_posix()
    ui.set('bgImage', rel_bg_path)

    write_xml_to_file(decentsampler, str(filepath))
    write_pil_image(bg_path, im=bg_img)

    if progress_callback is not None:
        progress_callback.emit(100)
        message_callback.emit(f'{smp_count} sample(s) found')

    print('')

    return filepath
