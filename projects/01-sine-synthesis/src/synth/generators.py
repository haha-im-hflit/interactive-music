

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .utils import midi_to_hz

class Generator:
    """
    Generator interface expected by the Audio engine.

    generate(num_frames, num_channels) returns:
      - audio: numpy float array length num_frames * num_channels
      - continue_flag: bool (True to keep generating next time)
    """

    def generate(self, num_frames: int, num_channels: int) -> Tuple[np.ndarray, bool]:
        raise NotImplementedError


class NoteGenerator(Generator):
    """
    Part 1 — NoteGenerator

    Instantiated with:
      - pitch: MIDI integer (69 -> A4 -> 440 Hz)
      - gain: 0.0 .. 1.0
      - wave: default "sine" (Part 1 supports only sine; Part 4 expands)
    note_off() stops the note by causing generate() to return continue_flag False.
    """

    def __init__(
        self,
        pitch: int,
        gain: float,
        wave: str = "sine",
        sample_rate: int = 44100,
    ):
        self.pitch = int(pitch)
        self.gain = float(gain)
        self.wave = str(wave)
        self.sample_rate = int(sample_rate)

        if self.wave != "sine":
            raise ValueError("Part 1 only supports wave='sine'.")

        self.freq = float(midi_to_hz(self.pitch))

        # Phase accumulator so the tone is continuous across generate() calls
        self._phase = 0.0  # radians
        self._on = True

    def note_off(self) -> None:
        """Turn off the note. Next generate() returns (silence, False)."""
        self._on = False

    def generate(self, num_frames: int, num_channels: int) -> Tuple[np.ndarray, bool]:
        # Part 1 is mono only
        assert num_channels == 1, "This assignment assumes mono output (num_channels == 1)."

        if not self._on:
            return np.zeros(num_frames, dtype=np.float32), False

        # radians per sample
        omega = 2.0 * math.pi * self.freq / self.sample_rate

        # sample indices within this block
        n = np.arange(num_frames, dtype=np.float64)

        # sine wave
        block = np.sin(self._phase + omega * n)

        # advance phase (wrap to keep it bounded)
        self._phase = (self._phase + omega * num_frames) % (2.0 * math.pi)

        # apply gain
        out = (self.gain * block).astype(np.float32)
        return out, True
