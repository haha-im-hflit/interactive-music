import numpy as np


class Tremolo(object):
    """
    Tremolo effect generator.

    This wraps another generator, modulates its amplitude, and returns
    the modified signal.
    """

    def __init__(self, input_gen, rate_hz=5.0, depth=0.7, sample_rate=44100):
        super(Tremolo, self).__init__()
        self.input_gen = input_gen
        self.rate_hz = float(rate_hz)
        self.depth = float(np.clip(depth, 0.0, 1.0))
        self.sample_rate = float(sample_rate)
        self.frame = 0

    def set_input_generator(self, input_gen):
        self.input_gen = input_gen

    def set_rate(self, rate_hz):
        self.rate_hz = float(max(rate_hz, 0.0))

    def set_depth(self, depth):
        self.depth = float(np.clip(depth, 0.0, 1.0))

    def generate(self, num_frames, num_channels):
        dry, continue_flag = self.input_gen.generate(num_frames, num_channels)

        frames = np.arange(self.frame, self.frame + num_frames, dtype=np.float64)
        theta = 2.0 * np.pi * frames * self.rate_hz / self.sample_rate
        lfo = 0.5 * (1.0 + np.sin(theta))

        # Scale between (1 - depth) and 1.0 so full silence is avoided by default.
        amp = (1.0 - self.depth) + self.depth * lfo
        self.frame += num_frames

        if num_channels == 1:
            mod = amp.astype(np.float32)
        else:
            mod = np.repeat(amp, num_channels).astype(np.float32)

        wet = np.asarray(dry, dtype=np.float32) * mod
        return wet, continue_flag
