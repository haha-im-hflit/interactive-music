import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(".."))

from imslib.audio import Audio
from imslib.core import BaseWidget, run
from imslib.gfxutil import topleft_label
from imslib.writer import AudioWriter
from lab1_tremolo import Tremolo


class SineGenerator(object):
    def __init__(self, initial_freq, sample_rate=44100):
        super(SineGenerator, self).__init__()
        self.frame = 0
        self.sample_rate = float(sample_rate)
        self.freq = float(initial_freq)
        self.gain = 0.2

        # Exercise 5: set this to 0.001, 0.01, or 0.1 while experimenting.
        self.simulated_delay_s = 0.0

    def set_freq(self, f):
        self.freq = float(f)

    def set_gain(self, g):
        self.gain = float(np.clip(g, 0, 1))

    def generate(self, num_frames, num_channels):
        # Exercise 2
        print(f"generate: num_frames={num_frames}, num_channels={num_channels}")

        # Exercise 3
        frames = np.arange(self.frame, self.frame + num_frames, dtype=np.float64)
        theta = 2.0 * np.pi * frames * self.freq / self.sample_rate
        mono = (self.gain * np.sin(theta)).astype(np.float32)
        self.frame += num_frames

        # Exercise 5
        if self.simulated_delay_s > 0.0:
            time.sleep(self.simulated_delay_s)

        if num_channels == 1:
            output = mono
        else:
            # Audio engine expects interleaved output for >1 channels.
            output = np.repeat(mono, num_channels)

        return output, True


class MainWidget(BaseWidget):
    def __init__(self):
        super(MainWidget, self).__init__()

        # Create the main Audio object. There should only be one global object.
        self.audio = Audio(1)

        # Add functionality for writing audio output to file for debugging.
        self.writer = AudioWriter("data")
        self.audio.add_listen_func(self.writer.add_audio)

        # Main generator plus effect processor for Exercise 8.
        self.sine_gen = SineGenerator(440)
        self.tremolo = Tremolo(self.sine_gen)

        # Exercise 2: hook up the generator to Audio.
        self.audio.set_generator(self.sine_gen)

        # Create a label used to display state on screen.
        self.info = topleft_label()
        self.add_widget(self.info)

    def _is_recording(self):
        """
        Handle small API differences across imslib versions.
        Returns True if AudioWriter is currently recording.
        """
        candidate_attrs = ("is_recording", "recording", "_recording")
        for name in candidate_attrs:
            if hasattr(self.writer, name):
                value = getattr(self.writer, name)
                try:
                    return bool(value() if callable(value) else value)
                except TypeError:
                    return bool(value)

        candidate_methods = ("is_active", "active", "get_recording_status")
        for name in candidate_methods:
            if hasattr(self.writer, name):
                fn = getattr(self.writer, name)
                if callable(fn):
                    try:
                        return bool(fn())
                    except TypeError:
                        pass
        return False

    # on_update() gets called very often (about 60 times per second).
    def on_update(self):
        # Exercise 2
        self.audio.on_update()

        self.info.text = "Info:\n"

        # Exercise 4
        self.info.text += f"audio CPU load: {self.audio.get_cpu_load():.2f} ms\n"
        self.info.text += f"gain: {self.sine_gen.gain:.2f}\n"
        self.info.text += f"frequency: {self.sine_gen.freq:.2f} Hz\n"

        # Exercise 7
        rec_text = "ON" if self._is_recording() else "OFF"
        self.info.text += f"recording: {rec_text}"

    def on_key_down(self, keycode, modifiers):
        print("key-down", keycode, modifiers)

        if keycode[1] == "t":
            # Exercise 8a
            self.audio.set_generator(self.tremolo)

        if keycode[1] == "up":
            self.sine_gen.set_gain(self.sine_gen.gain + 0.05)

        elif keycode[1] == "down":
            self.sine_gen.set_gain(self.sine_gen.gain - 0.05)

        elif keycode[1] == "left":
            # Exercise 6: move down by one half-step.
            self.sine_gen.set_freq(self.sine_gen.freq * (2.0 ** (-1.0 / 12.0)))

        elif keycode[1] == "right":
            # Exercise 6: move up by one half-step.
            self.sine_gen.set_freq(self.sine_gen.freq * (2.0 ** (1.0 / 12.0)))

        elif keycode[1] == "z":
            # First press starts recording, second press stops and writes .wav.
            self.writer.toggle()

    def on_key_up(self, keycode):
        print("key-up", keycode)

        # Exercise 8b
        if keycode[1] == "t":
            self.audio.set_generator(self.sine_gen)


run(MainWidget())
