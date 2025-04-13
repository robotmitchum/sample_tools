# coding:utf-8
"""
    :module: rename_tool_UI.py
    :description: Tool to rename/convert audio samples according to note
    while preserving some metadata useful for sampled/virtual instruments

    Support import/export for :
    - wav : through riff 'smpl' chunk, metadata are properly recognized by Kontakt or LoopAuditoneer for example
    - flac : through custom ID3 tags which are recognized by these tools and are very easy to manage
    (using foreign metadata inherited from wav is possible but is cumbersome to create
    and is in practice recognized by virtually nothing, so I gave up on this idea.)

    - aiff unsupported for export : both media chunks and tags are handled differently
    (besides I don't really use this format)

    :author: Michel 'Mitch' Pecqueur
    :date: 2024.06

    NOTE : Under windows, when experiencing pauses with video players in a web browser (YT for example)
    or any other interference with any software currently running
    replace the portaudio dll used by sounddevice found in
    .../Lib/site-packages/_sounddevice_data/portaudio-binaries
    by a non-asio dll from this repository :
    https://github.com/spatialaudio/portaudio-binaries

    Sound device is used to preview sounds by double-clicking on the item
"""

import importlib
import traceback
from functools import partial
from pathlib import Path

from PyQt5 import QtWidgets, QtGui, Qt, QtCore

import sample_utils as st
from UI import rename_tool as gui
from audio_player import play_notification
from base_tool_UI import BaseToolUi, launch
from common_ui_utils import add_ctx, add_insert_ctx, get_user_directory, resource_path, style_widget
from file_utils import move_to_subdir

# from simple_logger import SimpleLogger

__version__ = '1.1.1'


class RenameToolUi(gui.Ui_rename_tool_mw, BaseToolUi):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle(f'Rename Sample Tool v{__version__}')

        # log_path = self.base_dir / 'rename_tool_log.txt'
        # self.logger = SimpleLogger(log_path)

        self.groupname_le.setText('')
        self.src_pattern_le.setText('{group}_{note}_{vel}')
        self.tgt_pattern_le.setText('{group}_{noteName}_v{vel}')

        app_icon = QtGui.QIcon()
        img_file = resource_path(self.current_dir / 'UI/icons/rename_tool_64.png')
        app_icon.addFile(img_file, QtCore.QSize(64, 64))
        self.setWindowIcon(app_icon)

        self.get_defaults()

        self.update_message("Rename audio files and update their 'smpl' chunk information")

    def setup_connections(self):
        super().setup_connections()

        # Output path widgets
        default_dir = get_user_directory()
        desktop_dir = get_user_directory('Desktop')
        add_ctx(self.output_path_l, values=['', default_dir, desktop_dir],
                names=['Clear', 'Default directory', 'Desktop'])
        self.set_output_path_tb.clicked.connect(self.output_path_l.browse_path)

        # Base name widgets
        add_ctx(self.groupname_le, ['', 'Instrument', 'Sample'])
        add_ctx(self.repstr_le, ['', '001,v063', '001,v079 002,v047', '001,seq002', 'r,release'])

        # Pattern Widgets
        add_ctx(self.src_pattern_le,
                values=['', '{group}_{note}', '{group}_{note}_{trigger}', '{group}_{note}_{vel}',
                        '{group}_{note}_{seqPosition}', '{group}_{note}_{repstr}'], trigger=self.src_pattern_pb)
        smp_attrib = ['group', 'note', 'pitchFraction', 'vel', 'trigger', 'seqPosition', 'repstr', '_']
        src_fields = [f'{{{at}}}' for at in smp_attrib]
        add_insert_ctx(self.src_pattern_le, values=src_fields)

        add_ctx(self.tgt_pattern_le,
                values=['', '{group}_{note}', '{group}_{noteName}', '{group}_{noteName}_{trigger}',
                        '{group}_{noteName}_v{vel}', '{group}_{noteName}_{seqPosition}', '{group}_{noteName}_{repstr}'],
                trigger=self.tgt_pattern_pb)
        tgt_attrib = ['group', 'note', 'noteName', 'pitchFraction', 'vel', 'dyn', 'trigger', 'seqPosition', 'repstr']
        tgt_fields = [f'{{{at}}}' for at in tgt_attrib]
        add_insert_ctx(self.tgt_pattern_le, values=tgt_fields)

        # Prefix/Suffix widgets
        add_ctx(self.prefix_le, ['', 'Instrument_', 'Sample_'])
        add_ctx(self.suffix_le, ['', '_v063', '_v127', '_attack', '_release', '_seq1'])

        # Note/Loop widgets

        self.pitch_detect_cb.stateChanged.connect(lambda state: self.pitch_mode_cmb.setEnabled(state == 2))

        self.pitch_fraction_cmb.currentTextChanged.connect(
            lambda state: self.pitchfraction_dsb.setEnabled(state == 'override'))
        self.pitchfraction_dsb.setContextMenuPolicy(3)

        add_ctx(self.transpose_sb, [-12, 0, 12])

        # Process buttons
        self.process_sel_pb.clicked.connect(partial(self.do_process, mode='sel'))
        self.process_sel_pb.setFixedHeight(24)
        style_widget(self.process_sel_pb, properties={'background-color': 'rgb(95,95,95)', 'border-radius': 8})

        self.process_pb.clicked.connect(partial(self.do_process, mode='batch'))
        self.process_pb.setFixedHeight(24)
        style_widget(self.process_pb, properties={'border-radius': 8})

        # Custom events

    def batch_rename(self, worker, progress_callback, message_callback, files, test_run):
        """
        Batch process files given files

        :param Worker or None worker:
        :param function or None progress_callback:
        :param function or None message_callback:

        :param list files: Files to process
        :param bool test_run: True (simulate process)
        :return: Resulting names

        :rtype: list
        """
        count = len(files)

        # - Get UI settings -

        group_name = self.groupname_le.text()

        rep_str = self.repstr_le.text()
        rep_str = [item.split(',') for item in rep_str.split()]

        src_pattern = self.src_pattern_le.text()
        tgt_pattern = self.tgt_pattern_le.text()

        prefix = self.prefix_le.text()
        suffix = self.suffix_le.text()

        # Note/Loop settings
        pitch_detect = (None, self.pitch_mode_cmb.currentText())[self.pitch_detect_cb.isChecked()]
        force_pitch_from_name = self.force_pitch_name_cb.isChecked()
        transpose = self.transpose_sb.value()

        pitch_fraction_mode = self.pitch_fraction_cmb.currentText()
        pitch_fraction_ovr = (None, self.pitchfraction_dsb.value())[pitch_fraction_mode == 'override']

        use_loop = self.use_loop_cb.isChecked()

        output_path = self.output_path_l.fullPath()

        # File settings
        bit_depth = self.bitdepth_cmb.currentText()
        if bit_depth != 'same':
            bit_depth = eval(bit_depth)
        else:
            bit_depth = None

        ext = self.format_cmb.currentText()

        # Progress bar init
        self.progress_pb.setMaximum(len(files))
        self.progress_pb.setTextVisible(True)

        if progress_callback is not None:
            progress_callback.emit(0)
            message_callback.emit('%p%')

        result = []
        temp_files = None

        try:
            # Move original files to temp folder
            temp_files = move_to_subdir(files, sub_dir=None, test_run=test_run)

            for i, (f, tmp_f) in enumerate(zip(files, temp_files)):
                if worker.is_stopped():
                    return result

                if ext == 'same':
                    output_ext = Path(f).suffix[1:]
                else:
                    output_ext = ext

                # With test_run, files are not yet moved so, we need to use original location
                input_file = (str(tmp_f), f)[test_run]

                # output_dir must reflect original location if output_path is not provided
                output_dir = output_path or str(Path(f).parent)

                new_name = st.rename_sample(input_file=input_file, output_dir=output_dir, output_ext=output_ext,
                                            check_list=result, bit_depth=bit_depth,
                                            group_name=group_name, rep_str=rep_str,
                                            src_pattern=src_pattern, tgt_pattern=tgt_pattern,
                                            prefix=prefix, suffix=suffix,
                                            extra_tags=None,
                                            force_pitch_from_name=force_pitch_from_name,
                                            transpose=transpose,
                                            detect_pitch=pitch_detect,
                                            pitch_fraction_mode=pitch_fraction_mode,
                                            pitch_fraction_override=pitch_fraction_ovr,
                                            use_loop=use_loop, test_run=test_run)

                result.append(new_name)
                if progress_callback is not None:
                    progress_callback.emit(i + 1)

        except Exception as e:
            # self.logger.log_exception(f'An error occurred: {e}')
            traceback.print_exc()

        done = len(result)

        if done < count:
            if message_callback is not None:
                message_callback.emit('Some file(s) could not be processed - Please check settings')
            play_notification(audio_file=self.current_dir / 'process_error.flac')
        else:
            if test_run:
                if message_callback is not None:
                    message_callback.emit(f'{done} of {count} file(s)')
            else:
                # Delete temp folder
                temp_dirs = {Path(f).parent for f in temp_files}
                for d in temp_dirs:
                    d.rmdir()
                if message_callback is not None:
                    message_callback.emit(f'{done} of {count} file(s) processed')
                play_notification(audio_file=self.current_dir / 'process_complete.flac')

        if not test_run:
            if count == self.files_lw.count():
                self.add_lw_items(result)
            else:
                self.del_lw_items()
                res = self.get_lw_items()
                res.extend(result)
                self.add_lw_items(res)
            self.last_file = result[-1]

        return result

    def do_process(self, mode):
        count = self.files_lw.count()
        if not count:
            return False

        if mode == 'batch':
            files = self.get_lw_items()
        else:
            files = self.get_sel_lw_items()

        if not files:
            return False

        importlib.reload(st)

        result = []
        try:
            self.as_worker(partial(self.batch_rename, files=files, test_run=True))
            self.event_loop.exec_()  # Wait for result
            result = self.worker_result
            self.event_loop.quit()
        except Exception as e:
            print(f'Error: {e}')
            traceback.print_exc()

        if len(result) < len(files):
            return False

        # Max name length for files and projected result
        f_len, r_len = [max([len(Path(item).name) for item in items]) for items in [files, result]]

        sep = '  ►  '
        report = [f'{Path(in_f).name : <{f_len}}{sep}{Path(out_f).name}' for in_f, out_f in zip(files, result)]

        dialog = InfoDialog(info_list=report, parent=self)
        dialog.accept_cmd = partial(self.as_worker, partial(self.batch_rename, files=files, test_run=False))
        dialog.exec_()


class InfoDialog(QtWidgets.QDialog):
    def __init__(self, info_list, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Process Info')
        self.accept_cmd = None

        self.current_dir = Path(__file__).parent

        pos = QtGui.QCursor.pos()
        w, h = 800, 800
        self.setGeometry(pos.x() - w // 2, pos.y() - h, w, h)

        self.info_dialog_lyt = QtWidgets.QVBoxLayout(self)

        self.info_dialog_lw = QtWidgets.QListWidget(self)
        self.info_dialog_lw.setFrameShape(Qt.QFrame.NoFrame)

        # Set a mono font to the list widget to simplify white spaces handling
        font_path = resource_path(self.current_dir / 'RobotoMono-Medium.ttf')
        font_id = QtGui.QFontDatabase.addApplicationFont(font_path)
        font_family = QtGui.QFontDatabase.applicationFontFamilies(font_id)[0]

        # monospaced_font = Qt.QFont('DejaVu Sans Mono', 11)
        custom_font = Qt.QFont(font_family, 11)
        self.info_dialog_lw.setFont(custom_font)
        self.info_dialog_lw.setUniformItemSizes(True)

        for item in info_list:
            self.info_dialog_lw.addItem(item)

        self.info_dialog_bb = QtWidgets.QDialogButtonBox(self)
        self.info_dialog_bb.setOrientation(Qt.Qt.Horizontal)
        self.info_dialog_bb.setStandardButtons(Qt.QDialogButtonBox.Cancel | Qt.QDialogButtonBox.Apply)
        self.info_dialog_bb.setCenterButtons(True)

        cancel_pb = self.info_dialog_bb.button(Qt.QDialogButtonBox.Cancel)
        apply_pb = self.info_dialog_bb.button(Qt.QDialogButtonBox.Apply)

        cancel_pb.setDefault(True)
        apply_pb.setText('Accept')
        apply_pb.setStyleSheet('QPushButton{background-color: rgb(159, 95, 95);color: rgb(255, 255, 255);}')

        apply_pb.clicked.connect(self.accept)
        self.info_dialog_bb.rejected.connect(self.close)

        self.info_dialog_lyt.addWidget(self.info_dialog_lw)
        self.info_dialog_lyt.addWidget(self.info_dialog_bb)
        self.setLayout(self.info_dialog_lyt)

    def accept(self):
        if self.accept_cmd is not None:
            self.accept_cmd()
        self.close()


def run(mw=RenameToolUi, parent=None):
    window = mw(parent=parent)
    return window.run()


if __name__ == "__main__":
    launch(mw=RenameToolUi, app_id=f'mitch.RenameTool.{__version__}')
