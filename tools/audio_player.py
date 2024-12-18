# coding:utf-8
"""
    :module: audio_playback.py
    :description: Audio player class using sound device able to handle looped audio
    :author: Michel 'Mitch' Pecqueur
    :date: 2024.07
"""

import os
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf
from common_ui_utils import resource_path


class AudioPlayer:
    def __init__(self):
        self.is_playing = threading.Event()
        self.stream = None

    def play(self, data, sr, loop_start, loop_end, msg=None):
        if not self.is_playing.is_set():
            callback = AudioCallback(data, loop_start, loop_end, player=self)
            self.stream = sd.OutputStream(samplerate=sr, channels=data.ndim, callback=callback)
            self.is_playing.set()
            if msg:
                if loop_start is None or loop_start is None:
                    msg(f'▶ Press space bar to stop')
                else:
                    msg(f'▶ {loop_start}-{loop_end}    Press space bar to stop')
            with self.stream:
                while self.is_playing.is_set():
                    sd.sleep(100)
                if msg:
                    if loop_start is None or loop_start is None:
                        msg(f'■ Press space bar to play')

    def stop(self, msg=None):
        if self.is_playing.is_set():
            if self.stream is not None:
                # self._stream.stop()
                self.stream.close()
                self.stream = None
            self.is_playing.clear()
        if msg:
            msg('■ Press space bar to play')


class AudioCallback:
    """
    Audio callback supporting looping with sounddevice
    """

    def __init__(self, audio_data, loop_start, loop_end, player):
        self.player = player
        if loop_start is None or loop_end is None:
            self.loop_size = 0
            self.loop_start = 0
            self.loop_end = len(audio_data) - 1
        else:
            self.loop_start = loop_start
            self.loop_end = loop_end
            self.loop_size = loop_end - loop_start + 1

        self.buffer = audio_data.reshape(-1, audio_data.ndim)[:self.loop_end + 1]
        self.buffer_size = self.loop_end + 1

        self.current_pos = 0

    def __call__(self, outdata, frames, time, status):
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


def play_notification(audio_file):
    """
    Simple sound playback
    :param str or Path audio_file:
    :return:
    """
    if os.path.isfile(resource_path(audio_file)):
        data, sr = sf.read(audio_file)
        sd.play(data, sr)
