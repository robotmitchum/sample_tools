# coding:utf-8
"""
    :module: audio_playback.py
    :description: Audio player class using sound device able to handle looped audio
    :author: Michel 'Mitch' Pecqueur
    :date: 2024.07
"""

import os
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from PyQt5.QtCore import QObject, pyqtSignal

from common_ui_utils import resource_path


class AudioPlayerSignals(QObject):
    message = pyqtSignal(str)


class AudioPlayer:
    def __init__(self):
        self.is_playing = threading.Event()
        self.stream = None
        self.signals = AudioPlayerSignals()
        self.tuning = 0
        self.pan = 0
        self.volume = 1

    def play(self, data: np.ndarray, sr: int, loop_start: int, loop_end: int):
        if not self.is_playing.is_set():
            callback = AudioCallback(data, loop_start, loop_end, player=self)
            self.stream = sd.OutputStream(samplerate=sr, channels=data.ndim, callback=callback)
            self.is_playing.set()

            if loop_start is None or loop_start is None:
                self.signals.message.emit(f'▶ Press space bar to stop')
            else:
                self.signals.message.emit(f'▶ {loop_start}-{loop_end}    Press space bar to stop')

            with self.stream:
                while self.is_playing.is_set():
                    sd.sleep(100)
                if loop_start is None or loop_start is None:
                    try:
                        self.signals.message.emit(f'■ Press space bar to play')
                    except RuntimeError as e:
                        print(f'Error: {e}')

    def stop(self):
        if self.is_playing.is_set():
            if self.stream is not None:
                # self._stream.stop()
                self.stream.close()
                self.stream = None
            self.is_playing.clear()
        self.signals.message.emit('■ Press space bar to play')


class AudioCallback:
    """
    Audio callback supporting looping with sounddevice
    """

    def __init__(self, audio_data: np.ndarray, loop_start: int, loop_end: int, player: AudioPlayer) -> None:
        self.player = player
        if loop_start is None or loop_end is None:
            self.loop_size = 0
            self.loop_start = 0
            self.loop_end = len(audio_data) - 1
        else:
            self.loop_start = max(loop_start, 0)
            self.loop_end = min(loop_end, len(audio_data) - 1)  # Sanitize incorrect loop end
            self.loop_size = self.loop_end - self.loop_start + 1

        self.buffer = audio_data.reshape(-1, audio_data.ndim)[:self.loop_end + 1]
        self.buffer_size = self.loop_end + 1

        self.current_pos = 0

    def __call__(self, outdata: np.ndarray, frames: int, time, status) -> None:
        if status:
            print(status)

        if self.loop_size > 0:
            remaining = self.buffer_size - self.current_pos

            if frames <= remaining:
                outdata[:frames] = self.buffer[self.current_pos:self.current_pos + frames]
                self.current_pos += frames
            else:
                outdata[:remaining] = self.buffer[self.current_pos:]
                frames_needed = frames - remaining
                loops_needed = frames_needed // self.loop_size
                extra_frames = frames_needed % self.loop_size

                # Fill in the full loops needed
                for i in range(loops_needed):
                    outdata[remaining + i * self.loop_size:remaining + (i + 1) * self.loop_size] = self.buffer[
                                                                                                   self.loop_start:]
                # Fill in the extra frames needed
                outdata[remaining + loops_needed * self.loop_size:] = self.buffer[self.loop_start:][:extra_frames]
                self.current_pos = self.loop_start + extra_frames

        else:
            remaining = self.buffer_size - self.current_pos
            if frames <= remaining:
                outdata[:frames] = self.buffer[self.current_pos:self.current_pos + frames]
                self.current_pos += frames
            elif remaining > 0:
                buf = self.buffer[self.current_pos:self.current_pos + frames]
                pad = np.zeros(shape=(frames - remaining, buf.shape[1]))
                outdata[:frames] = np.concatenate((buf, pad), axis=0)
                self.player.is_playing.clear()
                raise sd.CallbackStop


def play_notification(audio_file: str | Path):
    """
    Simple sound playback
    :param audio_file:
    """
    if os.path.isfile(resource_path(audio_file)):
        data, sr = sf.read(audio_file)
        sd.play(data, sr)
