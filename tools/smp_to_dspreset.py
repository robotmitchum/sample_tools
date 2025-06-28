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

import instrument_utils as iu
from bitmap_knobs import *
from color_utils import (adjust_palette, basic_background, blank_button, write_pil_image,
                         write_text, limit_font_size, apply_symbol, hex_to_rgba, rgba_to_hex, plt_to_rgba)
from common_math_utils import linstep, lerp, clamp
from common_ui_utils import shorten_str, beautify_str
from estimate_offset import sampleset_offset
from file_utils import recursive_search, resolve_overwriting
from jsonFile import read_json

__ds_version__ = '1.12.16'
__version__ = '1.4.2'


def create_dspreset(root_dir: str, smp_subdir: str = 'Samples',
                    smp_fmt: list[str] = ('wav', 'flac'), ir_fmt: list[str] = ('wav', 'aif', 'flac'),

                    smp_attrib_cfg: str = 'smp_attrib_cfg.json',

                    pattern: str = '{group}_{note}_{trigger}', group_naming: str = 'keep',
                    override: bool = False, loop: bool = True,
                    transpose: int = 0, tuning: float = 0, seq_mode: str = 'random',
                    pad_vel: bool = False,

                    adsr: tuple[float, float, float, float] = (0.001, 0, 1, .25),
                    adr_curve: tuple[float | None, float | None, float | None] = (None, None, None),

                    fake_release: bool = False, fk_rls_mode: str = 'start',
                    fk_rls_volume: float = -24, fk_rls_tuning: float = 0, fk_rls_cutoff: float = 1000,
                    fk_rls_adsr: tuple[float, float, float, float] = (.001, .05, 0, .05),
                    fk_rls_adr_curve: tuple[float | None, float | None, float | None] = (None, None, None),

                    fake_legato: bool = False,
                    fk_leg_start: float = .1, fk_leg_a: float = .1, fk_leg_a_curve: float | None = None,

                    note_spread: str = 'mid', limit: bool = True, note_limit_mode: str = 'shared',
                    rr_offset: tuple[int] | None = None, rr_bounds: bool = True,
                    pf_mode: str = 'off', pf_th: float = 5,
                    crossfade_mode: str = 'linear', crossfade: float = .05,

                    attenuation: float = -6, vel_track: float = 1.0, note_pan: float = 0,

                    monophonic: bool = False,

                    bg_text: str or None = None,
                    text_font: tuple[str, int] = ('HelveticaNeueThin.otf', 24),

                    color_plt_cfg: str = 'plt_cfg/Dark_plt_cfg.json',
                    plt_adjust: tuple[float, float, float] = (0, 1, 1),
                    group_knobs_rows: int = 1, no_solo_grp_knob: bool = True,
                    adsr_knobs: bool = True, max_adsr_knobs: float = 10,
                    use_reverb: bool = True, reverb_wet: float = 0.2, ir_subdir: str = 'IR',
                    knob_scl: float = 1.0,

                    add_suffix: str = '', auto_increment: bool = True,

                    multi_out: bool = True,

                    estimate_delay: bool = False,

                    worker=None, progress_callback=None, message_callback=None):
    """
    Create a Decent Sampler Preset from audio samples

    :param text_font: font path, font size
    :param root_dir: Instrument root directory
    :param smp_subdir: Samples subdirectory name
    :param ir_subdir: Impulse Response subdirectory name

    :param smp_fmt: Accepted sample format extensions
    :param ir_fmt: Accepted IR file format extensions

    :param smp_attrib_cfg: Path to a json file holding sample attribute configuration
    :param color_plt_cfg: Path to a json file holding color configuration used to generate UI

    :param pattern: Pattern used to figure the mapping
    :param group_naming: 'keep', 'beautify', 'upper', 'lower','shorten'
    :param override: If True, info from sample name override metadata
    :param loop: Use sample loop

    :param transpose: Transpose the whole mapping
    :param tuning: Tuning adjustment at group level
    :param seq_mode: Round-Robin mode - 'round_robin','random', 'true_random' or 'always' (off)

    :param pad_vel: Re-use/duplicate note samples for velocity layers with fewer note samples

    :param adsr: ADSR envelope (in s, except sustain)
    :param adr_curve: ADR Curve (-100 log, 0 lin, 100 exp)

    :param fake_release: Create fake release from attack group and samples
    :param fk_rls_mode: 'start' 'loop_end' 'cue'
    :param fk_rls_volume: Release attenuation
    :param fk_rls_tuning: Release pitch shifting
    :param fk_rls_cutoff: Low pass filter value in Hz
    :param fk_rls_adsr: ADSR envelope of release
    :param fk_rls_adr_curve: ADR Curve of release

    :param fake_legato: Create fake legato from attack group and samples
    :param fk_leg_start: Start of legato sample in s
    :param fk_leg_a: Attack of legato envelope
    :param fk_leg_a_curve: Attack curve of legato envelope

    :param note_spread: 'up' 'mid' 'down' or None
    :param limit: Limit note range to bounding samples or extend bounds
    Extended bound can be given as a list
    :param note_limit_mode: 'shared' or 'group'

    :param rr_offset: Fake Round-Robin offsets
    :param rr_bounds: Steal notes further to make fake RR work on bounding notes

    :param pf_mode: Pitch fraction mode, 'off', 'on', 'mean_scale', 'on_rand', 'on_threshold', 'on_threshold_rand'
    :param pf_th: Pitch fraction threshold/random value

    :param crossfade_mode: 'linear' 'equal_power' or 'off'
    :param crossfade: Percent of loop length

    :param attenuation: Volume attenuation to prevent saturation/clipping
    :param vel_track: Velocity Tracking Amplitude
    :param note_pan: Auto pan sample from left to right depending on note number
    :param monophonic: Set instrument to monophonic

    :param bg_text: Override text written on background image otherwise use directory name
    :param group_knobs_rows: Number of rows for group knobs
    :param no_solo_grp_knob: Do not generate any group knob if there is only one
    :param adsr_knobs: Add ASDR knobs to UI
    :param max_adsr_knobs: Max length in s for ASDR knobs
    :param plt_adjust: Global Hue Saturation Value adjustment of color palette

    :param use_reverb: Add reverb knob to UI
    :param reverb_wet: Default reverb wet level

    :param knob_scl: Knob render factor

    :param add_suffix: Add suffix to created file
    :param auto_increment: Increment file to avoid overwriting

    :param multi_out: Enable multiple output

    :param estimate_delay: Estimate sample set delay and append information to info tool tip

    :param Worker worker:
    :param progress_callback:
    :param message_callback:

    :return: Created file
    :rtype: str
    """

    # - Retrieve instrument samples -

    # Sample attributes Config

    # List of sample attributes offered by Decent Sampler and *hopefully* supported by this tool
    # Attribute values can be provided in the sample name (with some limitations depending on how the name is formatted)
    # or using ID3 tags (flac only at the moment, it should support any attribute, but I didn't test everything...)

    num_attrs, ds_smp_attrib = ['vel', 'note', 'seqPosition'], []
    smp_attrib_data = read_json(smp_attrib_cfg) or {}
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
        if message_callback:
            progress_callback.emit(0)
            message_callback.emit(f'Your samples must be located in a "{smp_subdir}" sub-directoy')
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

    # - Path management -

    current_dir = Path(__file__).parent
    base_dir = current_dir.parent

    resources_dir = Path(root_dir) / 'Resources'
    if not resources_dir.exists():
        os.makedirs(resources_dir, exist_ok=True)

    basename = Path(root_dir).stem
    file_suffix = ('', add_suffix)[bool(add_suffix)]

    filepath = Path(root_dir) / f'{basename}{file_suffix}.dspreset'

    if auto_increment:
        filepath = resolve_overwriting(filepath, mode='file', test_run=True)
        version = Path(filepath).stem.split('_')[-1]
        file_suffix += f'_{version}'

    bg_path = resources_dir / f'bg{file_suffix}.jpg'
    rel_bg_path = bg_path.relative_to(root_dir).as_posix()

    # IR Samples
    ir_samples = []
    if ir_subdir:
        ir_samples = recursive_search(root_dir=Path.joinpath(Path(root_dir), ir_subdir), input_ext=ir_fmt,
                                      exclude=exclude, relpath=root_dir)

    # Color Palette Config
    plt_data = read_json(color_plt_cfg) or {}
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
        keyboard_plt = [rgba_to_hex(rgba=(1, .625, .875, 1))]

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
    row_spc_adjust = (1, 1.33)[group_knobs_rows > 1]

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

    # Create mute buttons
    btn_paths = []
    btn_w = 16
    if add_grp_knobs:
        for i, plt in enumerate([track_bg_plt[0], mute_btn_plt[0]]):
            btn_path = resources_dir / f'grp_btn{i}.png'
            btn_path = round_button(btn_path, size=btn_w, bg_color=plt, scl=2)
            btn_path = Path(btn_path).relative_to(root_dir)
            btn_paths.append(str(btn_path.as_posix()))
    # btn_w /= row_spc_adjust

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
    decentsampler.append(Et.Comment('"https://github.com/robotmitchum/sample_tools"'))
    tags = Et.SubElement(decentsampler, 'tags')
    effects = Et.SubElement(decentsampler, 'effects')
    midi = Et.SubElement(decentsampler, 'midi')

    # - Instrument UI -
    ui = Et.SubElement(decentsampler, 'ui',
                       attrib={'width': str(w), 'height': str(h), 'bgImage': str(rel_bg_path), 'layoutMode': 'relative',
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
    else:
        info_tooltip = f'Samplist - {beautify_str(getpass.getuser())}\n'
        info_tooltip += f"UI - {beautify_str(Path(__file__).stem)} {__version__}\n"

    if estimate_delay:
        delay = sampleset_offset(input_path=instr.df['path'].tolist(), verbose=True)
        info_tooltip += f'Negative Delay : {-round(delay)} ms\n'

    info_tooltip += '\n'

    symbol_path = current_dir / 'symbols/info.png'
    bg_img = apply_symbol(bg_img, symbol_path=symbol_path, pos_xy=(w - info_w - 8, top_band_h + 8),
                          size=(info_w, info_w), scl=2, color=plt_to_rgba(bg_text_plt[-1], alpha=.5))

    Et.SubElement(tab, 'button',
                  attrib={'style': 'image', 'x': str(w - info_w - 8), 'y': str(8), 'width': str(info_w),
                          'height': str(info_w), 'mainImage': str(blank_button_path.as_posix()),
                          'tooltip': info_tooltip})

    # Auto RR offset
    if isinstance(rr_offset, int):
        rr_offset = iu.rr_ofs_from_count(count=clamp(rr_offset, 1, 7))

    if len(rr_offset) > 1:
        instr.set_zones()

    print(f'- {instr.root_dir} : {instr.name} - \n')
    print(f'RR Offsets: {rr_offset}')
    print(f'Note limits: {instr.limit}')

    # - Add some headroom to avoid saturation -
    Et.SubElement(effects, 'effect', attrib={'type': 'gain', 'level': f'{attenuation}'})

    # - Default controls -

    mx_len = np.median([len(lbl) for lbl in grp_label.values()])  # Max Word Length
    kw = 96  # Knob Width
    spc = 80  # Group knob width/height
    ts = round(min((spc / mx_len) * 3, spc / 2))  # Text size
    margin = 120  # Margin for group sliders from canvas left border

    # - "Expression" (Volume) -
    ex = 48
    sw = 32
    ew, eh = 16, 160 - sw
    x = 8 + ex // 2
    y = cy - eh // 2 + sw - 4

    symbol_path = current_dir / 'symbols/expression.png'
    bg_img = apply_symbol(bg_img, symbol_path=symbol_path, pos_xy=(x - sw // 2, top_band_h + y - sw),
                          size=(sw, sw), scl=2, color=plt_to_rgba(ctrl_plt['expression']))

    frames = 128
    expr_path = resources_dir / f'expr_slider{file_suffix}.png'
    linear_slider(filepath=expr_path, shape=(ew, eh), scl=knob_scl,
                  bg_r=2, fg_r=2, dot_r=5,
                  bg_color=track_bg_plt[0], fg_color=ctrl_plt['expression'], dot_color=None,
                  frames=frames)

    expr_ctrl = Et.SubElement(tab, 'control',
                              attrib={'x': str(x - ew // 2), 'y': str(y), 'width': str(ew), 'height': str(eh),
                                      'parameterName': 'expression', 'style': 'custom_skin_vertical_drag',

                                      'customSkinImage': expr_path.relative_to(root_dir).as_posix(),
                                      'customSkinNumFrames': str(frames), 'customSkinImageOrientation': 'horizontal',

                                      'showLabel': 'false', 'textColor': ctrl_plt['expression'],
                                      'textSize': '40',
                                      'trackForegroundColor': ctrl_plt['expression'],
                                      'trackBackgroundColor': track_bg_plt[0],
                                      'tooltip': 'Expression',
                                      'type': 'float', 'minValue': '0', 'maxValue': '1', 'value': '1',
                                      'defaultValue': '1'})

    Et.SubElement(expr_ctrl, 'binding',
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

    ex = 48
    sw = 32
    ew, eh = 16, 160 - sw
    x = 56 + ex // 2
    y = cy - eh // 2 + sw - 4

    symbol_path = current_dir / 'symbols/modulation.png'
    bg_img = apply_symbol(bg_img, symbol_path=symbol_path, pos_xy=(x - sw // 2, top_band_h + y - sw),
                          size=(sw, sw), scl=2, color=plt_to_rgba(ctrl_plt['modulation']))

    mod_path = resources_dir / f'mod_slider{file_suffix}.png'
    linear_slider(filepath=mod_path, shape=(ew, eh), scl=knob_scl,
                  bg_r=2, fg_r=2, dot_r=5,
                  bg_color=track_bg_plt[0], fg_color=ctrl_plt['modulation'], dot_color=None,
                  frames=frames)

    mod_ctrl = Et.SubElement(tab, 'control',
                             attrib={'x': str(x - ew // 2), 'y': str(y), 'width': str(ew), 'height': str(eh),
                                     'parameterName': 'modulation', 'style': 'custom_skin_vertical_drag',

                                     'customSkinImage': mod_path.relative_to(root_dir).as_posix(),
                                     'customSkinNumFrames': str(frames), 'customSkinImageOrientation': 'horizontal',

                                     'showLabel': 'false', 'textColor': ctrl_plt['modulation'], 'textSize': '40',
                                     'trackForegroundColor': ctrl_plt['modulation'],
                                     'trackBackgroundColor': track_bg_plt[0],
                                     'tooltip': 'Modulation (Low Pass Filter)',
                                     'type': 'float', 'minValue': '0', 'maxValue': '1', 'value': '1',
                                     'defaultValue': '1'})

    bind = Et.SubElement(mod_ctrl, 'binding',
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
        rx = 96
        rw = 80
        rh = rw
        x = w - 8 - rx // 2
        y = cy + 8
        verb_dv = reverb_wet * 100

        sw = 32
        symbol_path = current_dir / 'symbols/reverb.png'
        bg_img = apply_symbol(bg_img, symbol_path=symbol_path,
                              pos_xy=(x - sw // 2, top_band_h + y - sw // 2),
                              size=(sw, sw), scl=2, color=plt_to_rgba(ctrl_plt['reverb']))

        frames = 49
        verb_path = resources_dir / f'verb_knob{file_suffix}.png'
        rotary_knob(filepath=verb_path, size=rw, scl=knob_scl,
                    bg_r=2, fg_r=2, dot_r=5,
                    bg_color=track_bg_plt[0], fg_color=ctrl_plt['reverb'], dot_color=None,
                    frames=frames)

        verb_knob = Et.SubElement(tab, 'labeled-knob',
                                  attrib={'x': str(x - rw / 2), 'y': str(y - rh // 2), 'width': str(rw),
                                          'height': str(rh),
                                          'parameterName': 'reverb', 'type': 'percent',
                                          'style': 'custom_skin_vertical_drag',

                                          'customSkinImage': verb_path.relative_to(root_dir).as_posix(),
                                          'customSkinNumFrames': str(frames), 'customSkinImageOrientation': 'vertical',

                                          'showLabel': 'false',
                                          'textColor': ctrl_plt['reverb'], 'textSize': '40',
                                          'trackForegroundColor': ctrl_plt['reverb'],
                                          'trackBackgroundColor': track_bg_plt[0],
                                          'tooltip': 'Reverb Wet Level',
                                          'minValue': '0', 'maxValue': '100',
                                          'value': f'{verb_dv}',
                                          'defaultValue': f'{verb_dv}'})

        prm = ('FX_REVERB_WET_LEVEL', 'FX_MIX')[bool(ir_samples)]
        Et.SubElement(verb_knob, 'binding', attrib={'type': 'effect', 'level': 'instrument', 'position': '2',
                                                    'parameter': prm, 'translation': 'linear',
                                                    'translationOutputMin': '0', 'translationOutputMax': '1'})

        # Pull down menu to choose between available IR files
        if ir_samples:
            mw, mh = 96, 32
            menu = Et.SubElement(tab, 'menu',
                                 attrib={'x': str(x - mw // 2), 'y': str(y + rh // 2), 'width': str(mw),
                                         'height': str(mh),
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

        if add_grp_knobs:
            # To bottom and smaller when using group knobs
            adsr_spc = 48
            y = ui_h - adsr_spc - 8
        else:
            # To center and bigger when no group knob
            adsr_spc = 64
            y = cy - adsr_spc / 2

        frames = 61
        adsr_path = resources_dir / f'adsr_knob{file_suffix}.png'
        rotary_knob(filepath=adsr_path, size=adsr_spc, scl=knob_scl,
                    bg_r=2, fg_r=2, dot_r=4,
                    bg_color=track_bg_plt[0], fg_color=group_plt[0], dot_color=None,
                    frames=frames)

        for i, (knob_name, value, mx_value, prm, tooltip) in enumerate(zip('ADSR', adsr, adsr_mx, prms, tooltips)):
            x = cx + (adsr_spc * (i - 1.5))

            lbl_h = adsr_spc / 4
            bg_img = write_text(bg_img, text=knob_name, text_xy=(x, top_band_h + y + adsr_spc / 2),
                                text_font=(text_font[0], lbl_h), text_color=plt_to_rgba(text_plt[0]),
                                align='center', anchor='mm', stroke_width=0, scl=2)

            adsr_knob = Et.SubElement(tab, 'labeled-knob',
                                      attrib={'x': str(x - adsr_spc / 2), 'y': str(y),
                                              'width': str(adsr_spc), 'height': str(adsr_spc),
                                              'parameterName': knob_name, 'type': 'percent',
                                              'style': 'custom_skin_vertical_drag',

                                              'customSkinImage': adsr_path.relative_to(root_dir).as_posix(),
                                              'customSkinNumFrames': str(frames),
                                              'customSkinImageOrientation': 'vertical',

                                              'showLabel': 'false',
                                              'textColor': text_plt[0], 'textSize': str(adsr_spc // 2),
                                              'trackForegroundColor': group_plt[0],
                                              'trackBackgroundColor': track_bg_plt[0],
                                              'minValue': '0', 'maxValue': str(mx_value),
                                              'value': f'{value}', 'defaultValue': f'{value}',
                                              'tooltip': tooltip})

            Et.SubElement(adsr_knob, 'binding',
                          attrib={'type': 'amp', 'level': 'instrument', 'position': '0', 'parameter': prm})

    # - Groups -
    groups = Et.SubElement(decentsampler, 'groups', attrib={'volume': '0dB', 'globalTuning': f'{tuning}'})

    # Velocity Track
    knob_w = 160
    knob_h = 16
    x = cx
    y = 16

    sw = 16
    symbol_path = current_dir / 'symbols/velocity.png'
    bg_img = apply_symbol(bg_img, symbol_path=symbol_path, pos_xy=(x - sw // 2, top_band_h + y // 2 - sw // 2),
                          size=(sw, sw), scl=2, color=plt_to_rgba(other_plt['ampVelTrack']))

    frames = 61
    vel_path = resources_dir / f'vel_slider{file_suffix}.png'
    linear_slider(filepath=vel_path, shape=(knob_w, knob_h), scl=knob_scl,
                  bg_r=2, fg_r=2, dot_r=5,
                  bg_color=track_bg_plt[0], fg_color=other_plt['ampVelTrack'], dot_color=None,
                  frames=frames)

    vel_ctrl = Et.SubElement(tab, 'control',
                             attrib={'x': str(x - knob_w // 2), 'y': str(y - knob_h // 2),
                                     'width': str(knob_w), 'height': str(knob_h),
                                     'parameterName': 'ampVelTrack', 'label': 'amplitude velocity tracking',
                                     'style': 'custom_skin_horizontal_drag', 'showLabel': 'false',

                                     'customSkinImage': vel_path.relative_to(root_dir).as_posix(),
                                     'customSkinNumFrames': str(frames), 'customSkinImageOrientation': 'vertical',

                                     'trackForegroundColor': other_plt['ampVelTrack'],
                                     'trackBackgroundColor': track_bg_plt[0],
                                     'tooltip': 'Amplitude Velocity Tracking',
                                     'type': 'float', 'minValue': '0', 'maxValue': '1', 'value': str(vel_track),
                                     'defaultValue': str(vel_track)})

    Et.SubElement(vel_ctrl, 'binding', attrib={'type': 'amp', 'level': 'instrument', 'parameter': 'AMP_VEL_TRACK'})

    # Release Knob
    if enable_release:
        knob_h = 80

        if add_grp_knobs:
            avt_h = 40  # Amplitude Velocity Tracking Height
            gk_h = 120 / row_spc_adjust
            gk_cy = gk_h / 2 + avt_h

            n_x = math.ceil(len(instr.groups) / max(1, group_knobs_rows))

            sw = 16

            x = ((w - margin * 2 - ew) / n_x) * ((n_x - 1) + .5) + margin + ew
            x += spc // 2
            y = gk_cy - knob_h / 2
        else:
            if adsr_knobs:
                adsr_spc = 64
                x = cx + adsr_spc * 1.5
                x += spc // 2
            else:
                x = cx
            y = cy - knob_h / 2

        symbol_path = current_dir / 'symbols/release.png'
        bg_img = apply_symbol(bg_img, symbol_path=symbol_path,
                              pos_xy=(int(x) - sw // 2, top_band_h + y - sw),
                              size=(sw, sw), scl=2, color=plt_to_rgba(other_plt['ampVelTrack']))

        frames = 61
        rls_path = resources_dir / f'rls_slider{file_suffix}.png'
        linear_slider(filepath=rls_path, shape=(ew, knob_h), scl=knob_scl,
                      bg_r=2, fg_r=2, dot_r=5,
                      bg_color=track_bg_plt[0], fg_color=other_plt['ampVelTrack'], dot_color=None,
                      frames=frames)

        ctrl = Et.SubElement(tab, 'control',
                             attrib={'x': str(int(x) - ew // 2), 'y': str(y),
                                     'width': str(ew), 'height': str(knob_h),
                                     'parameterName': 'release', 'style': 'custom_skin_vertical_drag',

                                     'customSkinImage': rls_path.relative_to(root_dir).as_posix(),
                                     'customSkinNumFrames': str(frames),
                                     'customSkinImageOrientation': 'horizontal',

                                     'showLabel': 'false', 'textColor': other_plt['ampVelTrack'],
                                     'textSize': '40',
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

    active_keys, active_key_groups = set(), set()
    note_idx = 0
    grp_font_size_limit = None

    grp_knob_paths = {}

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

        if progress_callback is not None:
            progress_callback.emit(0)
            message_callback.emit('%p%')

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
                grp_attrib['silencedByTags'] = grp

        # - Multi-Output -
        slider_idx = instr.groups.index(grp)

        if multi_out:
            grp_attrib['output1Target'] = 'MAIN_OUTPUT'
            grp_attrib['output1Volume'] = '1.0'
            grp_attrib['output2Target'] = f'AUX_STEREO_OUTPUT_{slider_idx + 1}'
            grp_attrib['output2Volume'] = '1.0'

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

            if fk_rls_mode != 'start':
                fk_rls_grp_attrib['sustain'] = '1'

            # for attr in ['silencingMode', 'silencedByTags']:
            #     fk_rls_grp_attrib.pop(attr, None)

            fk_rls_group = Et.SubElement(groups, 'group', attrib=fk_rls_grp_attrib)
            fk_rls_group.append(Et.Comment('Release faked by re-using attack samples with a different envelope'))
            grp_pos += 1

            if fk_rls_mode == 'start':
                fk_rls_fx = Et.SubElement(fk_rls_group, 'effects')
                Et.SubElement(fk_rls_fx, 'effect',
                              attrib={'type': 'lowpass', 'resonance': f'{.7}', 'frequency': f'{fk_rls_cutoff}'})

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

            avt_h = 40

            gk_h = 120 / row_spc_adjust

            gk_cy = gk_h / 2 + avt_h

            n_x = math.ceil(len(instr.groups) / max(1, group_knobs_rows))
            row_n = slider_idx // n_x
            x_div = n_x

            ctrl_w = min((w - margin * 2) / x_div, gk_h / group_knobs_rows, 80)

            er = (0, ew)[enable_release]

            x = ((w - margin * 2 - er / 2) / x_div) * (slider_idx % n_x + .5) + margin - ctrl_w / 2 - er / 2
            y = gk_cy + (row_n * ctrl_w * row_spc_adjust) - (ctrl_w / 2) * group_knobs_rows

            if ctrlname not in ctrls:
                lbl_h = ctrl_w / 4

                if grp_font_size_limit is None:
                    grp_font_size_limit = limit_font_size(texts=list(grp_label.keys()), text_font=(text_font[0], lbl_h),
                                                          max_length=ctrl_w - 8)
                lbl_h = grp_font_size_limit

                bg_img = write_text(bg_img, text=ctrlname, text_xy=(x + ctrl_w / 2, top_band_h + y),
                                    text_font=(text_font[0], lbl_h),
                                    text_color=plt_to_rgba(text_plt[0]),
                                    align='center', anchor='md', stroke_width=0, scl=2)

                frames = 49
                grp_plt_idx = slider_idx % len(group_plt)

                if grp_plt_idx not in grp_knob_paths:
                    p = resources_dir / f'grp_knob{file_suffix}.png'
                    rotary_knob(filepath=p, size=int(ctrl_w), scl=knob_scl,
                                bg_r=2, fg_r=2, dot_r=4, gap=90,
                                bg_color=track_bg_plt[0], fg_color=group_plt[grp_plt_idx], dot_color=None,
                                frames=frames)
                    grp_knob_paths[grp_plt_idx] = p.relative_to(root_dir).as_posix()

                ctrl = Et.SubElement(tab, 'control',
                                     attrib={'x': str(x), 'y': str(y),
                                             'width': str(ctrl_w), 'height': str(ctrl_w),
                                             'style': 'custom_skin_vertical_drag',
                                             'parameterName': shorten_str(grp), 'label': ctrlname,

                                             'customSkinImage': grp_knob_paths[grp_plt_idx],
                                             'customSkinNumFrames': str(frames),
                                             'customSkinImageOrientation': 'vertical',

                                             'showLabel': 'false', 'textColor': text_plt[0], 'textSize': str(ts),
                                             'trackForegroundColor': group_plt[grp_plt_idx],
                                             'trackBackgroundColor': track_bg_plt[0],
                                             'tooltip': f'{beautify_str(grp)} Volume',
                                             'type': 'float', 'minValue': "0", 'maxValue': "1", 'value': '1',
                                             'defaultValue': '1'})

                Et.SubElement(ctrl, 'binding',
                              attrib={'type': 'amp', 'level': 'tag', 'identifier': grp, 'parameter': 'AMP_VOLUME'})
                ctrls.append(ctrlname)

                # Mute button
                btn_w = (ctrl_w / 80) * 16
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

        # Per-group keyboard color
        if note_spread != 'none' and grp not in active_key_groups:
            plt_idx = instr.groups.index(grp) % len(active_keys_plt)
            lo, hi = note_limit
            active_keys |= set(range(lo, hi + 1))
            Et.SubElement(keyboard, 'color',
                          attrib={'loNote': str(lo), 'hiNote': str(hi),
                                  'color': active_keys_plt[plt_idx]})
            active_key_groups.add(grp)

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
                if note_spread == 'none' and note not in active_keys:
                    active_keys.add(note)
                    plt_idx = note_idx % len(active_keys_plt)
                    note_idx += 1
                    Et.SubElement(keyboard, 'color',
                                  attrib={'loNote': str(note), 'hiNote': str(note), 'color': active_keys_plt[plt_idx]})

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
                    fk_rls_smp_tuning = (smp_tuning, 0)[smp_tuning is None]

                    match fk_rls_mode:
                        case 'start':
                            fk_rls_smp_tuning += fk_rls_tuning
                        case 'loop_end' | 'cue':
                            fk_rls_smp_attrib['loopEnabled'] = '0'
                            for attr in ['loopStart', 'loopEnd', 'loopCrossfadeMode', 'loopCrossfade']:
                                fk_rls_smp_attrib.pop(attr, None)
                            smp_start = None
                            if fk_rls_mode == 'cue' and cues:
                                smp_start = cues[0]
                            elif loop_end:
                                smp_start = loop_end
                            if smp_start:
                                fk_rls_smp_attrib['start'] = str(smp_start)

                    if abs(fk_rls_smp_tuning) > .001:
                        fk_rls_smp_attrib['tuning'] = f'{fk_rls_smp_tuning:.03f}'

                    Et.SubElement(fk_rls_group, 'sample', attrib=fk_rls_smp_attrib)

                # Add fake legato sample to its related group
                # Most attributes are copied from source sample
                if fake_legato and trg == 'attack':
                    fk_leg_smp_attrib = smp_attrib.copy()

                    # Trim start of the sample
                    smp_start = min(int(sr * fk_leg_start), smp_len - 1)

                    if fk_leg_smp_attrib['loopEnabled'] == '1':
                        smp_start = min(smp_start, loop_start)

                    fk_leg_smp_attrib['start'] = str(smp_start)

                    Et.SubElement(fk_leg_group, 'sample', attrib=fk_leg_smp_attrib)

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

    write_xml_to_file(decentsampler, str(filepath))
    write_pil_image(bg_path, im=bg_img)

    if progress_callback is not None:
        progress_callback.emit(100)
    message_callback.emit(f'{smp_count} sample(s) found')

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
        result |= {v for v in elem.attrib.values() if root_dir.joinpath(v).is_file()}

    # Also include text files and preference files
    for ext in ('txt', 'smp2ds', 'drds'):
        files = root_dir.glob(f'*.{ext}')
        result |= {f.relative_to(root_dir).as_posix() for f in files}

    return result


def create_dslibrary(root_dir):
    """
    Create a dslibrary file from a root directory archiving only files required by found dspreset(s)
    :param str or Path root_dir:
    :return: Created file path
    :rtype: str
    """
    # Get all dspreset files from root_dir
    dspreset_files = [f for f in Path(root_dir).glob('*.dspreset')]

    if not dspreset_files:
        return None

    # Get dependencies from each dspreset
    deps = set()
    for f in dspreset_files:
        deps.add(Path(f).relative_to(root_dir).as_posix())
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
