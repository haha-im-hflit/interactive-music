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


SEMITONE_RATIO = 2 ** (1 / 12)


class SineGenerator(object):
    def __init__(self, initial_freq, sample_rate=44100):
        super(SineGenerator, self).__init__()
        self.frame = 0
        self.freq = float(initial_freq)
        self.gain = 0.2
        self.sample_rate = int(sample_rate)

        # Exercise 5: set to 0.001, 0.01, or 0.1 to hear performance issues.
        self.processing_delay_s = 0.0

    def set_freq(self, f):
        self.freq = float(f)

    def set_gain(self, g):
        self.gain = float(np.clip(g, 0, 1))

    def generate(self, num_frames, num_channels):
        # Exercise 2
        print(f"generate: num_frames={num_frames}, num_channels={num_channels}")

        # Exercise 3
        frames = np.arange(self.frame, self.frame + num_frames, dtype=np.float64)
        theta = 2.0 * np.pi * self.freq * frames / self.sample_rate
        output = (self.gain * np.sin(theta)).astype(np.float32)
        self.frame += num_frames

        # Exercise 5
        if self.processing_delay_s > 0:
            time.sleep(self.processing_delay_s)

        return (output, True)


class MainWidget(BaseWidget):
    def __init__(self):
        super(MainWidget, self).__init__()

        self.audio = Audio(1)

        self.writer = AudioWriter("data")
        self.audio.add_listen_func(self.writer.add_audio)

        self.sine_gen = SineGenerator(440)
        self.tremolo = Tremolo(self.sine_gen)

        # Exercise 2
        self.audio.set_generator(self.sine_gen)

        self.info = topleft_label()
        self.add_widget(self.info)

    def _is_recording(self):
        # AudioWriter implementations vary across lab versions, so check a few names.
        for attr in ("recording", "active", "is_recording"):
            if not hasattr(self.writer, attr):
                continue
            value = getattr(self.writer, attr)
            if callable(value):
                try:
                    value = value()
                except TypeError:
                    pass
            return bool(value)
        return False

    # on_update() gets called very often (about 60 times per second)
    def on_update(self):
        # Exercise 2
        self.audio.on_update()

        # Exercise 4 + 7
        cpu_ms = self.audio.get_cpu_load()
        recording_text = "ON" if self._is_recording() else "OFF"
        self.info.text = (
            "Info:\n"
            f"audio cpu: {cpu_ms:.2f} ms\n"
            f"gain: {self.sine_gen.gain:.2f}\n"
            f"frequency: {self.sine_gen.freq:.2f} Hz\n"
            f"recording: {recording_text}"
        )

    def on_key_down(self, keycode, modifiers):
        print("key-down", keycode, modifiers)

        if keycode[1] == "t":
            # Exercise 8
            self.audio.set_generator(self.tremolo)

        if keycode[1] == "up":
            self.sine_gen.set_gain(self.sine_gen.gain + 0.05)

        elif keycode[1] == "down":
            self.sine_gen.set_gain(self.sine_gen.gain - 0.05)

        elif keycode[1] == "left":
            # Exercise 6
            self.sine_gen.set_freq(self.sine_gen.freq / SEMITONE_RATIO)

        elif keycode[1] == "right":
            # Exercise 6
            self.sine_gen.set_freq(self.sine_gen.freq * SEMITONE_RATIO)

        elif keycode[1] == "z":
            self.writer.toggle()

    def on_key_up(self, keycode):
        print("key-up", keycode)

        # Exercise 8
        if keycode[1] == "t":
            self.audio.set_generator(self.sine_gen)


run(MainWidget())